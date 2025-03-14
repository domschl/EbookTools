import logging
import os
import sys
import termios
import re
import threading
import queue
from dataclasses import dataclass
from abc import abstractmethod
from typing import Protocol, override, cast

import sdl2
import sdl2.ext
import sdl2.sdlttf
import ctypes

@dataclass()
class InputEvent:
    cmd: str
    msg: str

        
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
    schema: dict[str, list[int]]


class ReplIO(Protocol):
    @abstractmethod
    def __init__(self, que:queue.Queue[InputEvent]):
        pass

    @abstractmethod
    def exit(self):
        pass
    
    @abstractmethod
    def cursor_hide(self):
        pass

    @abstractmethod
    def cursor_show(self):
        pass
    
    @abstractmethod
    def cursor_start_offset_get(self) -> tuple[int, int]:
        pass
    
    @abstractmethod
    def canvas_update_size(sel0f) -> tuple[int, int]:
        pass

    @abstractmethod
    def canvas_init(self, size_x:int =0, size_y:int=0) -> bool:
        pass

    @abstractmethod
    def canvas_print_at(self, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False):
        pass

    @abstractmethod
    def canvas_render_start(self):
        pass

    @abstractmethod
    def canvas_render_show(self):
        pass

    @abstractmethod
    def event_loop_tick(self):
        pass

    @abstractmethod
    def color_set(self, fg:list[int], bg:list[int] | None):
        pass
    

class TextReplIO(ReplIO):
    def __init__(self, que:queue.Queue[InputEvent]):
        self.log: logging.Logger = logging.getLogger("TextReplIO")
        self.input_queue: queue.Queue[InputEvent] = que
        self.cur_x_offset: int
        self.cur_y_offset: int
        self.cols: int
        self.rows: int
        self.cols, self.rows = self.canvas_update_size()
        self.fg_color: list[int] = [0xff, 0xff, 0xff, 0xff]
        self.bg_color: list[int] = [0, 0, 0, 0xff]

        self.input_loop_active:bool = False
        self.key_reader_active:bool = False
        self.cur_x_offset, self.cur_y_offset = self.get_cursor_pos()
        self.key_queue:queue.Queue[bytearray] = queue.Queue()
        self.key_reader_active = True
        self.key_thread: threading.Thread = threading.Thread(target=self.key_reader, daemon=True)
        self.key_thread.start()
        self.input_loop_active = True
        self.input_thread: threading.Thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

    @override
    def exit(self):
        self.key_reader_active = False
        self.input_loop_active = False

    @override
    def event_loop_tick(self):
        _, _ = self.canvas_update_size()
        
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

    @override
    def color_set(self, fg:list[int], bg:list[int] | None):
        self.fg_color = fg
        if bg is not None:
            self.bg_color = bg

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
        
    @override
    def cursor_start_offset_get(self) -> tuple[int, int]:
        return self.cur_x_offset, self.cur_y_offset

    @override
    def canvas_update_size(self) -> tuple[int, int]:
        self.cols, self.rows = os.get_terminal_size()
        return (self.cols, self.rows)
    
    @override
    def canvas_init(self, size_x:int =0, size_y:int=0) -> bool:
        if size_x == 0 or size_x > self.cols:
            size_x = self.cols
        if size_y == 0 or size_y > self.rows:
            size_y = self.rows

        if self.cur_y_offset + size_y >= self.rows:
            for _ in range(size_y - 1):
                print()
            self.cur_y_offset -= size_y + self.cur_y_offset - self.rows
        return True

    @override
    def canvas_print_at(self, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False):
        cols, rows = os.get_terminal_size()
        print(f"\033[38;2;{self.fg_color[0]};{self.fg_color[1]};{self.fg_color[2]}m")  # Set foreground color as RGB.
        print(f"\033[48;2;{self.bg_color[0]};{self.bg_color[1]};{self.bg_color[2]}m")  # Set background color as RGB.
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

    @override
    def canvas_render_start(self):
        return

    @override
    def canvas_render_show(self):
        _ = sys.stdout.flush()
        return
    
    @override
    def cursor_hide(self):
        print('\033[?25l', end="")
        _ = sys.stdout.flush()

    @override
    def cursor_show(self):
        print('\033[?25h', end="")

class Sdl2ReplIO(ReplIO):
    def __init__(self, que:queue.Queue[InputEvent]):
        self.log: logging.Logger = logging.getLogger("TextReplIO")
        self.input_queue: queue.Queue[InputEvent] = que
        self.cur_x_offset: int = 0
        self.cur_y_offset: int = 0
        self.cur_pos_x: int = 0
        self.cur_pos_y: int = 0
        self.cur_active: bool = True
        self.fg_color: list[int] = [0xff, 0xff, 0xff, 0xff]
        self.bg_color: list[int] = [0, 0, 0, 0xff]

        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]
        sdl2.sdlttf.TTF_Init()  # pyright: ignore[reportUnknownMemberType]
        self.event_loop_active: bool = True
        WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
        window = sdl2.ext.Window("SDL2 Text Example", size=(WINDOW_WIDTH, WINDOW_HEIGHT),
                                 flags = (sdl2.SDL_WINDOW_ALLOW_HIGHDPI |  sdl2.SDL_RENDERER_ACCELERATED)) # pyright: ignore[reportAny] # sdl2.SDL_WINDOW_RESIZABLE)) #  | sdl2.SDL_WINDOW_METAL |
        window.show()  # pyright: ignore[reportUnknownMemberType]
        self.renderer:sdl2.ext.Renderer = sdl2.ext.Renderer(window)

        rw: ctypes.c_int = ctypes.c_int(0)
        rh: ctypes.c_int = ctypes.c_int(0)
        #prw = ctypes.POINTER(ctypes.c_int(rw))
        #prh: ctypes.POINTER(ctypes.c_int)
        sdl2.SDL_GetRendererOutputSize(self.renderer.sdlrenderer, rw, rh);  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        if rw.value != WINDOW_WIDTH:
            widthScale = rw.value / WINDOW_WIDTH
            heightScale = rh.value / WINDOW_HEIGHT

            if widthScale != heightScale:
                self.log.warning("WARNING: width scale != height scale")
            else:
                print(f"Scale: {widthScale}")

            # sdl2.SDL_RenderSetScale(self.renderer.sdlrenderer, widthScale, heightScale);
            
        # font_path = "./Resources/IosevkaNerdFontMono-Regular.ttf"
        font_path = "./Resources/BabelStoneTibetan.ttf"
        if os.path.exists(font_path) is False:
            self.log.error(f"Font {font_path} does not exist")
        # sdl2.ext.RenderSetScale(self.renderer,2,2)
        self.font_mag:int = 2
        self.dpi:int = 144
        font_size = 8 * self.font_mag
        self.font: sdl2.sdlttf.TTF_Font = sdl2.sdlttf.TTF_OpenFontDPI(font_path.encode('utf-8'), font_size, self.dpi, self.dpi)  # pyright: ignore[reportUnknownMemberType] # , reportUnannotatedClassAttribute]
        sdl2.sdlttf.TTF_SetFontHinting(self.font, sdl2.sdlttf.TTF_HINTING_LIGHT_SUBPIXEL)  # pyright: ignore[reportAny, reportUnknownMemberType]
        rect = self.render_text("a", 0, 0)
        if rect is not None:
            self.char_width: int = rect.w  # pyright: ignore[reportAny]
            self.char_height: int = rect.h  # pyright: ignore[reportAny]
            print(f"Char-sizes: {self.char_width}, {self.char_height}")
        else:
            self.log.error("Cannot determine character dimensions!")
        self.line_spacing_extra:int = 0
        script = "Tibt".encode('utf-8')
        sdl2.sdlttf.TTF_SetFontScriptName(self.font, script)  # pyright:ignore[reportUnknownMemberType]

    @override
    def exit(self):
        sdl2.sdlttf.TTF_CloseFont(self.font)  # pyright: ignore[reportUnknownMemberType]
        sdl2.sdlttf.TTF_Quit()  # pyright: ignore[reportUnknownMemberType]
        sdl2.SDL_Quit()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

    @override
    def color_set(self, fg:list[int], bg:list[int] | None):
        self.fg_color = fg
        if bg is not None:
            self.bg_color = bg

    @override
    def canvas_init(self, size_x:int =0, size_y:int=0) -> bool:
        return True
    
    # Function to render text
    def render_text(self, text:str, x:int, y:int) -> sdl2.SDL_Rect | None:
        if text == "":
            return
        color_fg = sdl2.SDL_Color(self.fg_color[0], self.fg_color[1], self.fg_color[2])
        color_bg = sdl2.SDL_Color(self.bg_color[0], self.bg_color[1], self.bg_color[2])
        
        # Surface = sdl2.sdlttf.TTF_RenderUTF8_Solid(self.font, text.encode(), color)
        surface = sdl2.sdlttf.TTF_RenderUTF8_LCD(self.font, text.encode(), color_fg, color_bg)  # pyright:ignore[reportUnknownMemberType, reportUnknownVariableType]
        texture = sdl2.SDL_CreateTextureFromSurface(self.renderer.sdlrenderer, surface)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAttributeAccessIssue]
        rect = sdl2.SDL_Rect(x, y, surface.contents.w // self.font_mag, surface.contents.h // self.font_mag)  # pyright: ignore[reportUnknownMemberType]
        sdl2.SDL_FreeSurface(surface)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        sdl2.SDL_RenderCopy(self.renderer.sdlrenderer, texture, None, rect)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        sdl2.SDL_DestroyTexture(texture)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        return rect

    @override
    def canvas_print_at(self, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False):
        _ = self.render_text(msg, x*self.char_width, y*(self.char_height + self.line_spacing_extra))
        self.cur_pos_x = x + len(msg)
        self.cur_pos_y = y
        
    @override
    def event_loop_tick(self):
        events: list[sdl2.SDL_Event] = cast(list[sdl2.SDL_Event], sdl2.ext.get_events())  # pyright: ignore[reportUnknownMemberType]
        for event in events:
            if event.type == sdl2.SDL_QUIT:  # pyright: ignore[reportAny]
                msg = InputEvent("exit", "")
                self.input_queue.put_nowait(msg)
                continue
            if event.type == sdl2.SDL_KEYDOWN:  # pyright: ignore[reportAny]
                key_sym:int = event.key.keysym.sym  # pyright: ignore[reportAny]
                key_code:int = event.key.keysym.scancode  # pyright: ignore[reportAny]
                key_mod:int = event.key.keysym.mod  # pyright: ignore[reportAny]
                
                print(f"{hex(key_sym)} {key_sym} {key_code} {key_mod}")
                if key_code == 82: # up
                    msg = InputEvent("up", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_code == 81: # down
                    msg = InputEvent("down", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_code == 80:  # left
                    msg = InputEvent("left", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_code == 79: # up
                    msg = InputEvent("right", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_sym == 8:
                    msg = InputEvent("bsp", "")
                    self.input_queue.put_nowait(msg)
                    continue
                if key_sym == 13:
                    msg = InputEvent("nl", "")
                    self.input_queue.put_nowait(msg)
                    continue
                continue
            if event.type == sdl2.SDL_TEXTINPUT:  # pyright: ignore[reportAny]
                text_char:str = event.text.text.decode('utf-8')  # pyright: ignore[reportAny]
                _text_type:int = event.text.type  # pyright: ignore[reportAny]
                msg = InputEvent("char", text_char)
                self.input_queue.put_nowait(msg)
                continue
        #self.renderer.present()

    @override
    def canvas_render_start(self):
        self.renderer.clear()  # pyright: ignore[reportUnknownMemberType]
        return

    @override
    def canvas_render_show(self):
        if self.cur_active is True:
            sdl2.SDL_SetRenderDrawColor(self.renderer.sdlrenderer,self.fg_color[0],self.fg_color[1],self.fg_color[2],self.fg_color[3])  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            sdl2.SDL_RenderDrawLine(self.renderer.sdlrenderer, self.cur_pos_x * self.char_width, self.cur_pos_y * self.char_height, self.cur_pos_x * self.char_width, (self.cur_pos_y + 1) * self.char_height)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.renderer.present()  # pyright: ignore[reportUnknownMemberType]
        return

    @override
    def cursor_start_offset_get(self) -> tuple[int, int]:
        return self.cur_x_offset, self.cur_y_offset
    
    @override
    def cursor_show(self):
        self.cur_active = True
        return

    @override
    def cursor_hide(self):
        self.cur_active = False
        pass

    @override
    def canvas_update_size(self) -> tuple[int, int]:
        self.cols:int
        self.rows:int
        self.cols, self.rows = os.get_terminal_size()
        return (self.cols, self.rows)


class Repl():
    def __init__(self, engine:str="TEXT"):
        # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
        self.log: logging.Logger = logging.getLogger("Repl")
        valid_engines = ["TEXT", "SDL2"]
        self.default_schema: dict[str, list[int]] = {
            'fg': [240,240,240,0xff],
            'bg': [15,15,15,0xff],
            'lb': [0,32,120,0xff],
            'bb': [20,20,160,0xff],
            }
        self.schema: dict[str, list[int]] = self.default_schema
        self.editor_esc: bool = False
        self.pads: list[Pad] = []
        if engine not in valid_engines:
            self.log.error(f"Unknown engine {engine}, use one of {valid_engines}")
            exit(1)
        self.engine:str = engine
        self.input_queue:queue.Queue[InputEvent] = queue.Queue()
        if self.engine == "TEXT":
            self.repl: ReplIO = TextReplIO(self.input_queue)
        else:
            self.repl = Sdl2ReplIO(self.input_queue)

    def pad_print_at(self, pad_index:int, msg: str, y:int, x:int, flush:bool = False, scroll:bool=False, border:bool=False):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]
        if border is False:
            self.repl.canvas_print_at(msg, y+pad.screen_pos_y, x+pad.screen_pos_x, flush=flush, scroll=scroll)
        else:
            self.repl.canvas_print_at(msg, y+pad.screen_pos_y, x+pad.screen_pos_x-pad.left_border, flush=flush, scroll=scroll)

    def pad_create(self, buffer:list[str], height: int, width:int = 0, offset_y:int = 0, offset_x:int = 0, left_border:int=0, bottom_border:int=0, schema: dict[str, list[int]] | None = None) -> int:
        if schema is None:
            self.schema = self.default_schema
        cur_x_offset, cur_y_offset = self.repl.cursor_start_offset_get()
        if schema is None:
            schema = self.schema
        pad: Pad = Pad(
            screen_pos_x = cur_x_offset + offset_x + left_border,
            screen_pos_y = cur_y_offset + offset_y,
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
        self.pad_display(pad_index)
        return pad_index
    
    def pad_get(self, padIndex: int) -> Pad | None:
        if padIndex >=0 and padIndex<len(self.pads):
            return self.pads[padIndex]
        else:
            self.log.error(f"Invalid padIndex: {padIndex}")
            return None

    def pad_display(self, pad_index:int, set_cursor:bool = True, update_from_buffer:bool=True):
        if pad_index >= len(self.pads):
            return
        pad = self.pads[pad_index]

        self.repl.canvas_render_start()

        self.repl.color_set(self.schema['fg'], self.schema['bg'])
        if update_from_buffer is True:
            for i in range(pad.height):
                if i+pad.buf_y < len(pad.buffer):
                    pad.screen[i] = buffer[i+pad.buf_y][pad.buf_x:pad.buf_x+pad.width]
                    pad.screen[i] += ' ' * (pad.width - len(pad.screen[i]))
                else:
                    pad.screen[i] = ' ' * pad.width
        for i in range(pad.height):
            self.pad_print_at(pad_index, pad.screen[i], i, 0)
        if pad.left_border > 0:
            self.repl.color_set(self.schema['fg'], self.schema['lb'])
            for i in range(pad.height):
                self.pad_print_at(pad_index, f"  {i+pad.buf_y:3d} ", i, 0, border=True)
        if pad.bottom_border > 0:
            self.repl.color_set(self.schema['fg'], self.schema['bb'])
            for i in range(pad.height, pad.height+pad.bottom_border):
                status_msg = ' ' * pad.left_border + f"Doms editor ({pad.cur_y+pad.buf_y},{pad.cur_x+pad.buf_x})"
                gl = pad.left_border + pad.width
                status_msg = status_msg[:gl]
                status_msg += ' ' * (gl - len(status_msg))
                self.pad_print_at(pad_index, status_msg, i, 0, border=True)
        if set_cursor is True:
            self.pad_print_at(pad_index, "", pad.cur_y, pad.cur_x)
        self.repl.canvas_render_show()

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

    def create_editor(self, buffer: list[str], height: int, width:int = 0, offset_y:int =0, offset_x:int =0, schema: dict[str, list[int]] | None=None, line_no:bool=False, status_line:bool=False, debug:bool=False) -> int:
        tinp: InputEvent | None
        left_border:int = 0
        bottom_border:int = 0
        if line_no is True:
            left_border = 6
        if status_line is True:
            bottom_border = 1
        pad_id = self.pad_create(buffer, height, width, offset_y, offset_x, left_border, bottom_border, schema)
        self.repl.cursor_show()
        self.editor_esc = False
        pad = self.pad_get(pad_id)
        print("Starting editor loop")
        while self.editor_esc is False and pad is not None:
            try:
                tinp = self.input_queue.get(timeout=0.02)
            except queue.Empty:
                tinp = None
                self.repl.event_loop_tick()
                self.pad_display(pad_id)
                continue
            if debug is True:
                hex_msg = f"{bytearray(tinp.msg, encoding='utf-8')}"
                print(f"[{tinp.cmd},{tinp.msg},{hex_msg}]")
                self.input_queue.task_done()
            else:
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
                    self.pad_display(pad_id)
                elif tinp.cmd == 'exit':
                    self.editor_esc = True
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
                    self.pad_display(pad_id)
                elif tinp.cmd == "up":
                    _ = self.pad_move(pad_id, dy = -1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "down":
                    _ = self.pad_move(pad_id, dy = 1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "left":
                    _ = self.pad_move(pad_id, dx = -1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "right":
                    _ = self.pad_move(pad_id, dx = 1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "home":
                    _ = self.pad_move(pad_id, x=0)
                    self.pad_display(pad_id)
                elif tinp.cmd == "end":
                    _ = self.pad_move(pad_id, x= -1)
                    self.pad_display(pad_id)
                elif tinp.cmd == "PgUp":
                    _ = self.pad_move(pad_id, dy = -pad.height)
                    self.pad_display(pad_id)
                elif tinp.cmd == "PgDown":
                    _ = self.pad_move(pad_id, dy = pad.height)
                    self.pad_display(pad_id)
                elif tinp.cmd == "Start":
                    _ = self.pad_move(pad_id, x=0, y=0)
                    self.pad_display(pad_id)
                elif tinp.cmd == "End":
                    llen = len(pad.buffer) - 1
                    y = llen + pad.height
                    if y > llen:
                        y = llen
                    _ = self.pad_move(pad_id, y=y)
                    _ = self.pad_move(pad_id, x= -1)
                    self.pad_display(pad_id)
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
                    self.pad_display(pad_id)
                else:
                    print(f"Bad state: cmd={tinp.cmd}, msg={tinp.msg}")
                    exit(1)
                self.input_queue.task_done()
                
        self.pad_display(pad_id, False)
        print("Exit edit-loop")
        return pad_id


if __name__ == "__main__":
    repl = Repl(engine="TEXT")
    # repl = Repl(engine="SDL2")
    if repl.repl.canvas_init(10,60) is False:
        repl.log.error("Init failed.")
        exit(1)
    buffer: list[str] = ["That", "is", "the", "initial", "long", "text"]
    id = repl.create_editor(buffer, 10,60, 1, 3, None, True, True)
    print("Exit")
    repl.repl.exit()
