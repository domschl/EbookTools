import logging
import os
import sys
import termios
import re
import threading
import queue
from dataclasses import dataclass

import sdl2
import sdl2.ext
import sdl2.sdlttf


@dataclass()
class Pad:
    screen_pos_x: int
    screen_pos_y: int
    width: int
    height: int
    left_border: int
    bottom_border: int
    cur_x: int
    cur_y: int
    buffer: list[str]
    buf_x: int
    buf_y: int
    screen: list[str]
    schema: dict[str, int]

@dataclass()
class InputEvent:
    cmd: str
    msg: str


class Repl():
    def __init__(self, engine:str="TEXT"):
        # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
        self.log: logging.Logger = logging.getLogger("Repl")
        valid_engines = ["TEXT", "SDL2"]
        if engine not in valid_engines:
            self.log.error(f"Unknown engine {engine}, use one of {valid_engines}")
            exit(1)
        self.engine:str = engine
        self.default_schema: dict[str, int] = {
            'fg': 15,
            'bg': 243,
            'lb': 247,
            'bb': 55
            }
        self.pads: list[Pad] = []
        self.cur_x: int
        self.cur_y: int
        self.input_loop_active:bool = False
        self.key_reader_active:bool = False
        self.cur_x, self.cur_y = self.get_cursor_pos()

        self.input_translation_mode: str = "simple"
        self.input_queue:queue.Queue[InputEvent] = queue.Queue()
        if engine == "TEXT" or engine == "SDL2":
            self.key_queue:queue.Queue[bytearray] = queue.Queue()
            self.key_reader_active = True
            self.key_thread: threading.Thread = threading.Thread(target=self.key_reader, daemon=True)
            self.key_thread.start()
            self.input_loop_active = True
            self.input_thread: threading.Thread = threading.Thread(target=self.input_loop, daemon=True)
            self.input_thread.start()
        elif engine == "SDL2":
            sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]
            sdl2.sdlttf.TTF_Init()  # pyright: ignore[reportUnknownMemberType]
        else:
            self.log.error(f"State error {engine}")
            exit(1)

    def get_ansi_char(self) -> str | None:
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        term = termios.tcgetattr(fd)
        ch: str | None = None
        try:
            term[3] &= ~(termios.ICANON | termios.ECHO | termios.IGNBRK | termios.BRKINT)
            termios.tcsetattr(fd, termios.TCSAFLUSH, term)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        return ch
        
    def key_reader(self):
        while self.key_reader_active is True:
            inp = self.get_ansi_char()
            if isinstance(inp, str):
                bytes = bytearray(inp, encoding='UTF-8')
                self.key_queue.put_nowait(bytes)

    def input_loop(self):
        esc_state: bool = False
        esc_code = ""
        term_char:str = ""
        tinp: InputEvent = InputEvent("", "")
        while self.input_loop_active is True:
            try:
                inp = self.key_queue.get(timeout=0.01)
            except queue.Empty:
                if esc_state is True:
                    tinp = InputEvent("esc", "")
                    self.input_queue.put_nowait(tinp)
                esc_state = False
                esc_code = ""
                term_char = ""
                continue
            self.key_queue.task_done()
            if len(inp) > 0:
                if self.input_translation_mode == "simple":
                    if esc_state is True:
                        esc_code += chr(inp[0])
                        if len(esc_code) == 2:
                            if esc_code == "[A":
                                tinp = InputEvent("up", "")
                            elif esc_code == "[B":
                                tinp = InputEvent("down", "")
                            elif esc_code == "[C":
                                tinp = InputEvent("right", "")
                            elif esc_code == "[D":
                                tinp = InputEvent("left", "")
                            elif esc_code == "[F":
                                tinp = InputEvent("end", "")
                            elif esc_code == "[H":
                                tinp = InputEvent("home", "")
                            elif esc_code == "OP":
                                tinp = InputEvent("F1", "")
                            elif esc_code == "OQ":
                                tinp = InputEvent("F2", "")
                            elif esc_code == "OR":
                                tinp = InputEvent("F3", "")
                            elif esc_code == "OS":
                                tinp = InputEvent("F4", "")
                            elif esc_code[0] == "[" and esc_code[1] in "123456":
                                term_char = '~'
                            else:
                                tinp = InputEvent("err", "ESC-"+esc_code)
                            if tinp.cmd != "":
                                self.input_queue.put_nowait(tinp)
                                tinp = InputEvent("", "")
                                esc_code = ""
                                esc_state = False
                        if term_char != '' and esc_code.endswith(term_char):
                            if esc_code == "[5~":  # PgUp
                                tinp = InputEvent("PgUp", "")
                            elif esc_code == "[6~":
                                tinp = InputEvent("PgDown", "")
                            elif esc_code == "[5;2~":
                                tinp = InputEvent("Start", "")
                            elif esc_code == "[6;2~":
                                tinp = InputEvent("End", "")
                            else:
                                tinp = InputEvent("EscSeq", esc_code)
                            self.input_queue.put_nowait(tinp)
                            tinp = InputEvent("", "")
                            esc_code = ""
                            esc_state = False
                            term_char = ""
                    else:
                        if inp == bytearray([0x7f]):  # BSP
                            tinp = InputEvent("bsp", "")
                        elif inp == bytearray([27]):  # ESC
                            esc_state = True
                            continue
                        elif inp == bytearray([0x05]):  # Ctrl-E
                            tinp = InputEvent("end", "")
                        elif inp == bytearray([0x0a]):
                            tinp = InputEvent("nl", "")
                        elif inp == bytearray([0x01]):  # ^A
                            tinp = InputEvent("home", "")
                        elif inp == bytearray([0x06]):  # ^F
                            tinp = InputEvent("right", "")
                        elif inp == bytearray([0x02]):  # ^B
                            tinp = InputEvent("left", "")
                        elif inp == bytearray([14]):  # ^N
                            tinp = InputEvent("down", "")
                        elif inp == bytearray([16]):  # ^P
                            tinp = InputEvent("up", "")
                        elif inp == bytearray([24]):  # ^X
                            tinp = InputEvent("exit", "")
                        else:
                            tinp = InputEvent("char", inp.decode('utf-8'))
                        # print(f"<Q:{tinp}>", end="")
                        # _ = sys.stdout.flush()
                        self.input_queue.put_nowait(tinp)
                else:
                    pass
                # print(f"<QE: {inp}>")
                # _ = sys.stdout.flush()
                    
        
    def get_cursor_pos(self) -> tuple[int, int]:
        if self.input_loop_active is False:
            _ = sys.stdout.write("\x1b[6n")
            _ = sys.stdout.flush()
            res = ""
            while res.endswith('R') is False:
                t = self.get_ansi_char()
                if t is not None:
                    res += t
            mt = re.match(r".*\[(?P<y>\d*);(?P<x>\d*)R", res)
            if mt is not None:
                x = int(mt.group("x"))
                y = int(mt.group("y"))
                return (x, y)
            else:
                return (-1, -1)
        else:
            return (-1, -1)
            
    def hide_cursor(self):
        print('\033[?25l', end="")
        _ = sys.stdout.flush()

    def show_cursor(self, padIndex: int | None =None):
        if padIndex is None:
            print('\033[?25h', end="")
        else:
            pad = self.pads[padIndex]
            print(f"\033[{pad.screen_pos_y+pad.cur_y};{pad.screen_pos_x + pad.cur_x}H\033[?25h", end="")
        _ = sys.stdout.flush()
      
    def print_at(self, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False):
        cols, rows = os.get_terminal_size()
        if scroll is False:
            if x>=cols or y>=rows:
                if flush is True:
                    _ = sys.stdout.flush()
                return
        nmsg = ""
        for c in msg:
            if ord(c)<32:
                continue
            else:
                nmsg +=c
        if x+len(nmsg) > cols:
            nmsg = nmsg[:cols-x]
        print(f"\033[{y};{x}H{nmsg}", end="")
        if flush is True:
            _ = sys.stdout.flush()

    def init_screen(self, size_x:int =0, size_y:int=0) -> bool:
        if self.engine == "TEXT":
            cols, rows = os.get_terminal_size()
            if size_x == 0 or size_x > cols:
                size_x = cols
            if size_y == 0 or size_y > rows:
                size_y = rows
            
            if self.cur_y + size_y >= rows:
                for _ in range(size_y - 1):
                    print()
                self.cur_y -= size_y + self.cur_y - rows
            return True
        elif self.engine == "SDL2":
            self.log.error("NOT IMPLEMENTED")
            return False
        else:
            self.log.error(f"Bad engine {self.engine} at init")
        return False

    def create_pad(self, buffer:list[str], height: int, width:int = 0, offset_y:int = 0, offset_x:int = 0, left_border:int=0, bottom_border:int=0, schema: dict[str, int] | None = None) -> int:
        if schema is None:
            schema = self.default_schema
        pad: Pad = Pad(
            screen_pos_x = self.cur_x + offset_x + left_border,
            screen_pos_y = self.cur_y + offset_y,
            width = width-left_border,
            height = height-bottom_border,
            left_border = left_border,
            bottom_border = bottom_border,
            cur_x = 0,
            cur_y = 0,
            schema = schema,
            screen = [' ' * width] * height,
            buffer = buffer,
            buf_x = 0,
            buf_y = 0
            )
        self.pads.append(pad)
        pad_index = len(self.pads)-1
        self.display_screen(pad_index)
        return pad_index
    
    def pad_print_at(self, pad_index:int, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False, border:bool=False):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]
        if border is False:
            self.print_at(msg, y+pad.screen_pos_y, x+pad.screen_pos_x, flush=flush, scroll=scroll)
        else:
            self.print_at(msg, y+pad.screen_pos_y, x+pad.screen_pos_x-pad.left_border, flush=flush, scroll=scroll)

    def display_screen(self, pad_index:int, set_cursor:bool = True, update_from_buffer:bool=True):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]

        if update_from_buffer is True:
            for i in range(pad.height):
                if i+pad.buf_y < len(pad.buffer):
                    pad.screen[i] = buffer[i+pad.buf_y][pad.buf_x:pad.buf_x+pad.width]
                    pad.screen[i] += ' ' * (pad.width - len(pad.screen[i]))
                else:
                    pad.screen[i] = ' ' * pad.width
        print(f"\033[48;5;{pad.schema['bg']}m", end="")
        print(f"\033[38;5;{pad.schema['fg']}m", end="")
        for i in range(pad.height):
            self.pad_print_at(pad_index, pad.screen[i], i, 0)
        if pad.left_border > 0:
            print(f"\033[48;5;{pad.schema['lb']}m", end="")
            for i in range(pad.height):
                self.pad_print_at(pad_index, f"  {i+pad.buf_y:3d} ", i, 0, border=True)
        if pad.bottom_border > 0:
            print(f"\033[48;5;{pad.schema['bb']}m", end="")
            for i in range(pad.height, pad.height+pad.bottom_border):
                status_msg = ' ' * pad.left_border + f"Doms editor ({pad.cur_y+pad.buf_y},{pad.cur_x+pad.buf_x})"
                gl = pad.left_border + pad.width
                status_msg = status_msg[:gl]
                status_msg += ' ' * (gl - len(status_msg))
                self.pad_print_at(pad_index, status_msg, i, 0, border=True)
        if set_cursor is True:
            self.pad_print_at(pad_index, "", pad.cur_y, pad.cur_x)
        _ = sys.stdout.flush()

    def pad_move(self, pad_id:int, dx:int | None = None, dy:int | None = None, x:int | None = None, y: int | None = None) -> bool:
        changed: bool = False
        if pad_id>= len(self.pads):
            return changed
        pad = self.pads[pad_id]
        if x is None:
            if dx is not None:
                if dx < 0:
                    if pad.cur_x + dx >=0:
                        pad.cur_x += dx
                        changed=True
                    elif pad.buf_x + pad.cur_x + dx >= 0:
                        pad.buf_x += dx
                        if pad.buf_x < 0:
                            pad.buf_x = 0
                            pad.cur_x = 0
                        changed = True
                    else:
                        pad.buf_x = 0
                        pad.cur_x = 0
                        changed = True
                elif dx > 0:
                    len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
                    if pad.buf_x + pad.cur_x < len_x:
                        if pad.cur_x < pad.width:
                            pad.cur_x += dx
                        else:
                            pad.buf_x += dx
                        changed = True
                    else:
                        pass  # EOL, don't expand
        else:
            if x == -1:
                len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
                len_w = len_x - pad.width
                if len_w < 0:
                    len_w = 0
                pad.buf_x = len_w
                pad.cur_x = len_x - pad.buf_x
                changed = True
            elif x == 0:
                pad.buf_x = 0
                pad.cur_x = 0
                changed = True
            else:
                len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
                if x > len_x:
                    x= len_x
                if x <= pad.width:
                    pad.buf_x = 0
                    pad.cur_x = x
                else:
                    pad.buf_x = x
                    pad.cur_x = 0
                changed = True
        if y is None:
            if dy is not None:
                if dy < 0:
                    if pad.cur_y + dy >=0:
                        pad.cur_y += dy
                        changed=True
                    elif pad.buf_y + pad.cur_y + dy >= 0:
                        pad.buf_y += dy
                        if pad.buf_y < 0:
                            pad.buf_y = 0
                            pad.cur_y = 0
                        changed = True
                    else:
                        pad.buf_y = 0
                        pad.cur_y = 0
                        changed = True
                elif dy > 0:
                    if pad.buf_y + pad.cur_y < len(pad.buffer) - dy:
                        if pad.cur_y < pad.height - 1:
                            pad.cur_y += dy
                        else:
                            pad.buf_y += dy
                        changed = True
        else:
            if y == -1:
                len_y = len(pad.buffer)
                len_h = len_y - pad.height
                if len_h < 0:
                    len_h = 0
                pad.buf_y = len_h
                pad.cur_y = len_y - pad.buf_y
                changed = True
            elif y == 0:
                pad.buf_y = 0
                pad.cur_y = 0
                changed = True
            else:
                len_y = len(pad.buffer)
                if y > len_y:
                    y= len_y
                if y <= pad.height:
                    pad.buf_y = 0
                    pad.cur_y = y
                else:
                    pad.buf_y = y
                    pad.cur_y = 0
                changed = True
        len_x = len(pad.buffer[pad.buf_y+pad.cur_y])
        delta = len_x - (pad.buf_x + pad.cur_x)
        if delta < 0:
            if pad.cur_x + delta >= 0:
                pad.cur_x += delta
            else:
                pad.buf_x += delta
                if pad.buf_x < 0:
                    pad.buf_x = 0
                    pad.cur_x = 0
            changed = True
        if pad.cur_x == pad.width:
            pad.buf_x += 1
            pad.cur_x -= 1
            changed = True
        while pad.cur_y >= pad.height:
            pad.buf_y += 1
            if pad.cur_y > 0:
                pad.cur_y -= 1
        if pad.cur_y >= pad.height:
            print(f"Pad_y: {pad.cur_y} error")
            exit(1)
        return changed

    def create_editor(self, buffer: list[str], height: int, width:int = 0, offset_y:int =0, offset_x:int =0, schema: dict[str, int] | None=None, line_no:bool=False, status_line:bool=False, debug:bool=False) -> int:
        left_border:int = 0
        bottom_border:int = 0
        if line_no is True:
            left_border = 6
        if status_line is True:
            bottom_border = 1
        pad_id = self.create_pad(buffer, height, width, offset_y, offset_x, left_border, bottom_border, schema)
        self.show_cursor(pad_id)
        esc: bool = False
        pad = self.pads[pad_id]
        while esc is False:
            try:
                tinp:InputEvent | None = self.input_queue.get(timeout=0.02)
            except queue.Empty:
                tinp = None
            if debug is True and tinp is not None:
                hex_msg = f"{bytearray(tinp.msg, encoding='utf-8')}"
                print(f"[{tinp.cmd},{tinp.msg},{hex_msg}]")
                self.input_queue.task_done()
            else:
                if tinp is not None:
                    if tinp.cmd == "bsp":
                        if pad.cur_x + pad.buf_x > 0:
                            _ = self.pad_move(pad_id, dx = -1)
                            pad.buffer[pad.buf_y+pad.cur_y] = pad.buffer[pad.buf_y+pad.cur_y][:pad.buf_x+pad.cur_x] + pad.buffer[pad.buf_y+pad.cur_y][pad.buf_x+pad.cur_x+1:]
                        else:
                            if pad.cur_y + pad.buf_y > 0:
                                cur_idx = pad.cur_y+pad.buf_y
                                cur_line = pad.buffer[cur_idx]
                                _ = self.pad_move(pad_id, dy = -1)
                                _ = self.pad_move(pad_id, x = -1)
                                cur_idx_new = pad.cur_y+pad.buf_y
                                pad.buffer[cur_idx_new] += cur_line
                                del pad.buffer[cur_idx]
                        self.display_screen(pad_id)
                    elif tinp.cmd == 'exit':
                        esc = True
                    elif tinp.cmd == "nl":
                        cur_ind = pad.cur_y+pad.buf_y
                        cur_pos = pad.cur_x + pad.buf_x
                        if cur_ind < len(pad.buffer):
                            cur_line: str = pad.buffer[cur_ind]
                        else:
                            print("error cur_line invl")
                            cur_line = ""
                            exit(1)
                        left = cur_line[:cur_pos]
                        right = cur_line[cur_pos:]
                        pad.buffer[cur_ind]=left
                        if cur_ind == len(pad.buffer) -1:
                            pad.buffer.append(right)
                        else:
                            pad.buffer.insert(cur_ind+1, right)
                        _ = self.pad_move(pad_id, dy=1, x=0)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "up":
                        _ = self.pad_move(pad_id, dy = -1)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "down":
                        _ = self.pad_move(pad_id, dy = 1)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "left":
                        _ = self.pad_move(pad_id, dx = -1)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "right":
                        _ = self.pad_move(pad_id, dx = 1)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "home":
                        _ = self.pad_move(pad_id, x=0)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "end":
                        _ = self.pad_move(pad_id, x= -1)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "PgUp":
                        _ = self.pad_move(pad_id, dy = -pad.height)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "PgDown":
                        _ = self.pad_move(pad_id, dy = pad.height)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "Start":
                        _ = self.pad_move(pad_id, x=0, y=0)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "End":
                        llen = len(pad.buffer) - 1
                        y = llen + pad.height
                        if y > llen:
                            y = llen
                        _ = self.pad_move(pad_id, y=y)
                        _ = self.pad_move(pad_id, x= -1)
                        self.display_screen(pad_id)
                    elif tinp.cmd == "err":
                        print()
                        print(tinp.msg)
                        exit(1)
                    elif tinp.cmd == "char":
                        cur_ind = pad.cur_y+pad.buf_y
                        cur_line = pad.buffer[cur_ind]
                        if ord(tinp.msg[0]) >= 32:
                            left = cur_line[:pad.buf_x+pad.cur_x]
                            right = cur_line[pad.buf_x+pad.cur_x:]
                            pad.buffer[cur_ind] = left + tinp.msg + right
                            _ = self.pad_move(pad_id, dx = 1)
                        self.display_screen(pad_id)
                    else:
                        print(f"Bad state: cmd={tinp.cmd}, msg={tinp.msg}")
                        exit(1)
                    self.input_queue.task_done()
                
        self.display_screen(pad_id, False)
        return pad_id


if __name__ == "__main__":
    repl = Repl(engine="TEXT")
    if repl.init_screen(40,100) is False:
        repl.log.error("Init failed.")
        exit(1)
    buffer: list[str] = ["That", "is", "the", "initial", "long", "text"]
    id = repl.create_editor(buffer, 40,100, 1, 3, None, True, True)
    # print()
    # print(buffer)
