from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import sdl2
import sdl2.ext

# From: <https://gist.github.com/gmarull/dcc8218385014559c1ca46047457c364>
# matplotlib: force computer modern font set
plt.rc('mathtext', fontset='cm')  # pyright: ignore[reportUnknownMemberType]


def tex2svg(formula: str, fontsize:int=12, dpi:int=200) -> bytes:

    fig: Figure = plt.figure(figsize=(0.01, 0.01))  # pyright: ignore[reportUnknownMemberType]
    _ = fig.text(0, 0, r'${}$'.format(formula), fontsize=fontsize)  # pyright: ignore[reportUnknownMemberType]

    output: BytesIO = BytesIO()
    fig.savefig(output, dpi=dpi, transparent=True, format='svg',  # pyright: ignore[reportUnknownMemberType]
                bbox_inches='tight', pad_inches=0.0, frameon=False)
    plt.close(fig)

    _ = output.seek(0)
    svg_bytes: bytes = output.read()  
    return svg_bytes

def svgtext(renderer: sdl2.ext.Renderer, svg_bytes: bytes) -> sdl2.ext.Texture:
    # sdl2.ext.init()
    # window = sdl2.ext.Window("SVG Display", size=svg_surface.get_size())
    # renderer = sdl2.ext.Renderer(window)
    svg_surface = sdl2.ext.image.load_svg(BytesIO(svg_bytes))
    texture = sdl2.ext.Texture(renderer, svg_surface)
    sdl2.SDL_FreeSurface(svg_surface)
    # renderer.clear()
    # renderer.copy(texture)
    # sdl2.SDL_DestroyTexture(texture)
    # renderer.present()
    return texture
