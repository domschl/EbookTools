import logging
import os
import sys
import termios
import tty
import time
import re
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


class Repl():
    def __init__(self, lines: int=5):
        # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
        self.log: logging.Logger = logging.getLogger("Repl")
        self.default_schema: dict[str, int] = {
            'fg': 15,
            'bg': 243,
            }
        self.pads: list[Pad] = []
        
    def get_cursor_pos(self) -> tuple[int, int]:
        old_stdin = termios.tcgetattr(sys.stdin)
        attr = termios.tcgetattr(sys.stdin)
        attr[3] = attr[3] & ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, attr)
        try:
            _ = ""
            _ = sys.stdout.write("\x1b[6n")
            _ = sys.stdout.flush()
            seq:str = ""
            while not (seq := seq + sys.stdin.read(1)).endswith('R'):
                time.sleep(0.01)
            res = re.match(r".*\[(?P<y>\d*);(?P<x>\d*)R", seq)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_stdin)
        if res is not None:
            xs, ys =  (res.group("x"), res.group("y"))
            try:
                x: int = int(xs)
                y: int = int(ys)
            except Exception as _:
                return (-1, -1)
            return x, y
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

    def get_char(self):
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        try:
            _ = tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        return ch

    def create_pad(self, height: int, width:int = 0, offset_y:int = 0, offset_x:int = 0, schema: dict[str, int] | None = None) -> int:
        cols, rows = os.get_terminal_size()
        cur_x, cur_y = self.get_cursor_pos()
        if width + offset_x > cols:
            width = cols - offset_x
        if height + offset_y > rows:
            height = rows - offset_y
        if cur_y + offset_y + height >= rows:
            for _ in range(height + offset_y - 2):
                print()
            cur_y -= height + cur_y + offset_y - rows
        if schema is None:
            schema = self.default_schema
        pad: Pad = {
            'screen_pos_x': cur_x + offset_x,
            'screen_pos_y': cur_y + offset_y,
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
        
    def create_editor(self, height: int, width:int = 0, offset_y:int =0, offset_x:int =0, schema: dict[str, int] | None=None) -> int:
        pad_id = self.create_pad(height, width, offset_y, offset_x, schema)
        self.show_cursor(pad_id)
        esc: bool = False
        pad = self.pads[pad_id]
        while esc is False:
            c = self.get_char()
            bytes = f"{bytearray(c.encode('utf-8'))}"
            self.print_at(bytes, 0, 0, True)
            if c[0] == chr(0x7f):
                if pad['cur_x'] > 0:
                    pad['cur_x'] -= 1
                    self.pad_print_at(pad_id, " ", pad['cur_y'], pad['cur_x'], True)
                    self.pad_print_at(pad_id, "", pad['cur_y'], pad['cur_x'], True)
                else:
                    if pad['cur_y'] > 0:
                        pad['cur_y'] -= 1
                        pad['cur_x'] = pad['width'] -1
            elif c[0] == 'q':
                esc = True
            elif c[0] == chr(13):
                pad['cur_x'] = 0
                if pad['cur_y'] + 1 < pad['height']:
                    pad['cur_y'] += 1
            else:
                self.pad_print_at(pad_id, c, pad['cur_y'], pad['cur_x'], True)
                pad['cur_x'] += 1
                if pad['cur_x'] == pad['width']:
                    pad['cur_x'] = 0
                    if pad['cur_y'] + 1 < pad['height']:
                        pad['cur_y'] += 1
                        
            
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
