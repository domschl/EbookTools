def sanitized_md_filename(name: str) -> str:
    bad_chars = "\\/:*?\"<>|.`'\n\r\t[]{}()&^%$#@!~"
    for char in bad_chars:
        name = name.replace(char, "_")
    name = name.replace("__", "_")
    name = name.replace(" _ ", ", ")
    name = name.replace("_ ", ", ")
    name = name.replace(" _", " ")
    return name


def progress_bar_string(
        progress: int, max_progress:int, bar_length: int=20, start_bracket: str | None="⦃", end_bracket: str | None="⦄"
) -> str:
    """Create a Unicode progress bar string

    This creates a string of length bar_length with a Unicode progress bar using
    fractional Unicode block characters. The returned string is always of constant
    length and is suitable for printing to a terminal or notebook.

    This pretty much obsoletes the `tqdm` or similar package for simple progress bars.

    :param progress: current progress
    :param max_progress: maximum progress
    :param bar_length: length of the progress bar
    :param start_bracket: Unicode string to use as the start bracket, None for no bracket
    :param end_bracket: Unicode string to use as the end bracket, None for no bracket
    :return: Unicode progress bar string of length `bar_length`
    """
    progress_frac = progress / max_progress
    num_blocks = int(bar_length * progress_frac)
    rem = bar_length * progress_frac - num_blocks
    blocks = " ▏▎▍▌▋▊▉█"
    remainder_index = int(rem * len(blocks))
    bar = blocks[-1] * num_blocks
    if remainder_index > 0:
        bar += blocks[remainder_index]
    bar += " " * (bar_length - len(bar))
    if start_bracket is not None:
        bar = start_bracket + bar
    if end_bracket is not None:
        bar += end_bracket
    return bar
