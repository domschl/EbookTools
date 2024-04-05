def sanitized_md_filename(name):
    bad_chars = "\\/:*?\"<>|.`'\n\r\t[]{}()&^%$#@!~"
    for char in bad_chars:
        name = name.replace(char, "_")
    name = name.replace("__", "_")
    name = name.replace(" _ ", ", ")
    name = name.replace("_ ", ", ")
    name = name.replace(" _", " ")
    return name
