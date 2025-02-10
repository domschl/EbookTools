import logging
import os
import sys
import termios
# import tty
import time
import re
import threading
import queue
from dataclasses import dataclass
from typing import TypedDict


class Pad(TypedDict):
    screen_pos_x: int
    screen_pos_y: int
    width: int
    height: int
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
    def __init__(self, lines: int=5):
        # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
        self.log: logging.Logger = logging.getLogger("Repl")
        self.default_schema: dict[str, int] = {
            'fg': 15,
            'bg': 243,
            }
        self.pads: list[Pad] = []
        self.cur_x: int
        self.cur_y: int
        self.input_loop_active:bool = False
        self.key_reader_active:bool = False
        self.cur_x, self.cur_y = self.get_cursor_pos()

        self.input_translation_mode: str = "simple"
        self.input_queue:queue.Queue[InputEvent] = queue.Queue()
        self.key_queue:queue.Queue[bytearray] = queue.Queue()
        self.key_reader_active = True
        self.key_thread: threading.Thread = threading.Thread(target=self.key_reader, daemon=True)
        self.key_thread.start()
        self.input_loop_active = True
        self.input_thread: threading.Thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

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
        tinp: InputEvent
        while self.input_loop_active is True:
            try:
                inp = self.key_queue.get(timeout=0.01)
            except queue.Empty:
                if esc_state is True:
                    tinp = InputEvent("esc", "")
                    self.input_queue.put_nowait(tinp)
                esc_state = False
                esc_code = ""
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
                            else:
                                tinp = InputEvent("err", "ESC-"+esc_code)
                            if tinp.cmd != "":
                                self.input_queue.put_nowait(tinp)
                            esc_code = ""
                            esc_state = False
                    else:
                        if inp[0] == 0x7f:  # BSP
                            tinp = InputEvent("bsp", "")
                        elif inp[0] == 27:  # ESC
                            esc_state = True
                            continue
                        elif inp[0] == 0x05:  # Ctrl-E
                            tinp = InputEvent("exit", "")
                        elif inp[0] == ord('\n'):
                            tinp = InputEvent("nl", "")
                        else:
                            tinp = InputEvent("char", chr(inp[0]))
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
            print(f"\033[{pad['screen_pos_y']+pad['cur_y']};{pad['screen_pos_x'] + pad['cur_x']}H\033[?25h", end="")
        _ = sys.stdout.flush()
      
    def print_at(self, msg: str, y:int, x:int, flush:bool = False):
        print(f"\033[{y};{x}H{msg}", end="")
        if flush is True:
            _ = sys.stdout.flush()


    def create_pad(self, height: int, width:int = 0, offset_y:int = 0, offset_x:int = 0, schema: dict[str, int] | None = None) -> int:
        cols, rows = os.get_terminal_size()
        if width + offset_x > cols:
            width = cols - offset_x
        if height + offset_y > rows:
            height = rows - offset_y
        if self.cur_y + offset_y + height >= rows:
            for _ in range(height + offset_y - 2):
                print()
            self.cur_y -= height + self.cur_y + offset_y - rows
        if schema is None:
            schema = self.default_schema
        pad: Pad = {
            'screen_pos_x': self.cur_x + offset_x,
            'screen_pos_y': self.cur_y + offset_y,
            'width': width,
            'height': height,
            'cur_x': 0,
            'cur_y': 0,
            'schema': schema,
            'screen': [' ' * width] * height,
            'buffer': [],
            'buf_x': 0,
            'buf_y': 0
            }
        self.pads.append(pad)
        pad_index = len(self.pads)-1
        self.display_screen(pad_index)
        return pad_index
    
    def pad_print_at(self, pad_index:int, msg: str, y:int, x:int, flush:bool = False):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]
        self.print_at(msg, y+pad['screen_pos_y'], x+pad['screen_pos_x'], flush=flush)

    def display_screen(self, pad_index:int):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]
        print(f"\033[48;5;{pad['schema']['bg']}m")
        print(f"\033[38;5;{pad['schema']['fg']}m")
        for i in range(pad['height']):
            self.pad_print_at(pad_index, pad['screen'][i], i, 0)
        _ = sys.stdout.flush()
        
    def create_editor(self, height: int, width:int = 0, offset_y:int =0, offset_x:int =0, schema: dict[str, int] | None=None, debug:bool=False) -> int:
        pad_id = self.create_pad(height, width, offset_y, offset_x, schema)
        self.show_cursor(pad_id)
        esc: bool = False
        pad = self.pads[pad_id]
        while esc is False:
            try:
                tinp:InputEvent | None = self.input_queue.get(timeout=0.02)
            except queue.Empty:
                tinp = None
            # bytes = f"{bytearray(c.encode('utf-8'))}"
            # self.print_at(bytes, 0, 0, True)
            if debug is True and tinp is not None:
                hex_msg = f"{bytearray(tinp.msg, encoding='utf-8')}"
                print(f"[{tinp.cmd},{tinp.msg},{hex_msg}]")
                self.input_queue.task_done()
            else:
                if tinp is not None:
                    if tinp.cmd == "bsp":
                        if pad['cur_x'] > 0:
                            pad['cur_x'] -= 1
                            self.pad_print_at(pad_id, " ", pad['cur_y'], pad['cur_x'], True)
                            self.pad_print_at(pad_id, "", pad['cur_y'], pad['cur_x'], True)
                        else:
                            if pad['cur_y'] > 0:
                                pad['cur_y'] -= 1
                                pad['cur_x'] = pad['width'] -1
                    elif tinp.cmd == 'exit':
                        esc = True
                    elif tinp.cmd == "nl":
                        pad['cur_x'] = 0
                        if pad['cur_y'] + 1 < pad['height']:
                            pad['cur_y'] += 1
                        self.pad_print_at(pad_id, "", pad['cur_y'], pad['cur_x'], flush=True)
                    elif tinp.cmd == "up":
                        if pad['cur_y'] > 0:
                            pad['cur_y'] -= 1
                            self.pad_print_at(pad_id, "", pad['cur_y'], pad['cur_x'], flush=True)
                    elif tinp.cmd == "down":
                        if pad['cur_y'] < pad['height'] - 1:
                            pad['cur_y'] += 1
                            self.pad_print_at(pad_id, "", pad['cur_y'], pad['cur_x'], flush=True)
                    elif tinp.cmd == "left":
                        if pad['cur_x'] > 0:
                            pad['cur_x'] -= 1
                            self.pad_print_at(pad_id, "", pad['cur_y'], pad['cur_x'], flush=True)
                    elif tinp.cmd == "right":
                        if pad['cur_x'] < pad['width'] - 1:
                            pad['cur_x'] += 1
                            self.pad_print_at(pad_id, "", pad['cur_y'], pad['cur_x'], flush=True)
                    elif tinp.cmd == "err":
                        print()
                        print(tinp.msg)
                        exit(1)
                    elif tinp.cmd == "char":
                        self.pad_print_at(pad_id, tinp.msg, pad['cur_y'], pad['cur_x'], True)
                        pad['cur_x'] += 1
                        if pad['cur_x'] == pad['width']:
                            pad['cur_x'] = 0
                            if pad['cur_y'] + 1 < pad['height']:
                                pad['cur_y'] += 1
                    else:
                        print(f"Bad state: cmd={tinp.cmd}, msg={tinp.msg}")
                        exit(1)
                    self.input_queue.task_done()
                
        self.display_screen(pad_id)
        return pad_id


if __name__ == "__main__":
    repl = Repl()
    id = repl.create_editor(7,40, 1, 3)
    print()
    # repl.print_at("Done\n",6, 0)
    # repl.fill_pad('x')
   #  time.sleep(1)
    # repl.fill_pad('y')
    # time.sleep(1)
    # repl.fill_pad('z')
    # time.sleep(1)
    # repl.show_cursor()
    # c:str = repl.get_char()
