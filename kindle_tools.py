import os
import logging
import sys
import datetime

from kindle_tools_localization import kindle_kw_locales


class KindleTools:
    """Class to manage the Kindle clippings file

    Kindle clippings for notes, highlights, and bookmarks are exported as JSON.

    Note that Kindle clipping format is difficult to parse since it is very locale-specific.
    You can add languages to the kindle_kw_locales dictionary in kindle_tools_localization.py.
    Currently, only English and German are supported.
    """

    def __init__(self):
        self.log = logging.getLogger("KindleTools")
        self.mount_folder = None
        self.clippings = []

    def check_for_connected_kindle(self):
        if sys.platform == "darwin":
            mount_folder = "/Volumes/Kindle"
        elif sys.platform == "linux":
            user = os.environ.get("USER")
            mount_folder = f"/run/media/{user}/Kindle"
        else:
            self.log.error(f"Unsupported platform: {sys.platform}")
            return False
        if os.path.exists(mount_folder):
            self.log.info("Kindle connected")
            self.mount_folder = mount_folder
            return True
        else:
            self.log.error("Kindle not connected")
            return False

    def get_clippings_text(self, local_file=None):
        if local_file:
            clippings_file = os.path.expanduser(local_file)
        else:
            if not self.mount_folder:
                self.log.error(
                    "No Kindle connected, cannot get automatic clippings file"
                )
                return None
            clippings_file = os.path.join(
                self.mount_folder, "documents", "My Clippings.txt"
            )
        if not os.path.exists(clippings_file):
            self.log.error(f"Clippings file not found: {clippings_file}")
            return None
        # text_clippings = None
        with open(clippings_file, "rb") as f:
            # Kindle stupidly uses UTF-8 with BOM markers for every single entry!
            clippings_text = f.read().decode("utf-8-sig").replace("\ufeff", "")
        return clippings_text

    def get_clipping_locale(self, clipping_text):
        for locale_name in kindle_kw_locales:
            locale = kindle_kw_locales[locale_name]
            for line in clipping_text.split("\n"):
                if (
                    line.startswith(locale["info_line_start"])
                    and line.find(locale["info_line_date_start"]) > 0
                    and (
                        line.find(locale["info_line_type_loc_sep"]) > 0
                        or line.find(locale["info_line_page_sep"]) > 0
                    )
                ):
                    return locale_name, locale
        return None, None

    def parse_clippings(self, clippings_text):
        self.clippings = []
        for clipping in clippings_text.split("=========="):
            clipping = clipping.strip()
            if not clipping or clipping == "":
                continue
            print(clipping)
            lines = clipping.split("\n")
            title_author = ""
            header_lines = 0
            for line in lines:
                line = line.strip()
                if line == "" or line.startswith("- "):
                    break
                header_lines += 1
                if title_author == "":
                    title_author = line
                else:
                    title_author = title_author + " " + line
            author_separator = title_author.rfind("(")
            if author_separator > 0:
                title = title_author[:author_separator].strip()
                author = title_author[author_separator:].strip()
                # remove brackets from author and change last, first to first last
                author = author.replace("(", "").replace(")", "")
                author_parts = author.split(",")
                if len(author_parts) == 2:
                    author = author_parts[1].strip() + " " + author_parts[0].strip()
                else:
                    self.log.warning(f"Could not parse author: {author}, using as-is")
            else:
                title = title_author
                author = None
            locale_name, locale = self.get_clipping_locale(clipping)
            if locale is None:
                self.log.warning(f"Could not determine locale for clipping: {title}")
                self.log.info(f"Clipping text: {clipping}")
                return None

            # get the type of clipping, location, and date
            clipping_type_location_date = lines[header_lines].strip()
            for transform in locale["info_line_transforms"]:
                clipping_type_location_date = clipping_type_location_date.replace(
                    transform[0], transform[1]
                )
            # separate the type, location, and date
            type_location_date = clipping_type_location_date.split(" | ")
            clipping_location = None
            if len(type_location_date) == 2:
                # No page number
                page_no = None
                type_location = type_location_date[0].strip()
                date = (
                    type_location_date[1]
                    .replace(locale["info_line_date_start"], "")
                    .strip()
                )
                if locale["info_line_type_loc_sep"] in type_location:
                    # separate the type and location
                    type_location_parts = type_location.split(
                        locale["info_line_type_loc_sep"]
                    )
                    clipping_type = (
                        type_location_parts[0]
                        .replace(locale["info_line_start"], "")
                        .strip()
                    )
                    clipping_location = type_location_parts[1].strip()
                else:
                    type_page_parts = type_location.split(locale["info_line_page_sep"])
                    clipping_type = (
                        type_page_parts[0]
                        .replace(locale["info_line_start"], "")
                        .strip()
                    )
                    page_no = type_page_parts[1].strip()
            elif len(type_location_date) == 3:
                # With page number
                date = (
                    type_location_date[2]
                    .replace(locale["info_line_date_start"], "")
                    .strip()
                )
                type_page = type_location_date[0].strip()
                comps = type_page.split(locale["info_line_page_sep"])
                page_no = comps[1].strip()
                type_location = (
                    type_location_date[1]
                    .strip()
                    .replace(locale["info_line_page_sep"], "")
                )
                clipping_type = comps[0].replace(locale["info_line_start"], "").strip()
                clipping_location = (
                    type_location_date[1]
                    .strip()
                    .replace(locale["info_line_type_loc_sep"], "")
                )
            else:
                self.log.error(f"Could not parse type, location, date: {type_location_date}")
                return None
            # get the text of the clipping
            clipping_text = "\n".join(lines[header_lines + 1:]).strip()
            # convert date to ISO format
            if locale_name == "en":
                # Tuesday, 28 March 2023 13:48:18
                date_parts = date.split(" ")
                day = int(date_parts[1])
                month = locale["months"].index(date_parts[2]) + 1
                year = date_parts[3]
                time = date_parts[4]
                iso_date_local = f"{year}-{month:02d}-{day:02d}T{time}"
                # convert to datetime and set timezone to local time
                dt_local = datetime.datetime.strptime(
                    iso_date_local, "%Y-%m-%dT%H:%M:%S"
                ).astimezone(datetime.datetime.now().astimezone().tzinfo)
                # convert to UTC
                dt_utc = dt_local.astimezone(datetime.timezone.utc)
            elif locale_name == "de":
                # Dienstag, 28. März 2023 13:48:18
                print(date)
                date_parts = date.split(" ")
                print(date_parts)
                day = int(date_parts[1][:-1])
                month = locale["months"].index(date_parts[2]) + 1
                year = date_parts[3]
                time = date_parts[4]
                iso_date_local = f"{year}-{month:02d}-{day:02d}T{time}"
                # convert to datetime and set timezone to local time
                dt_local = datetime.datetime.strptime(
                    iso_date_local, "%Y-%m-%dT%H:%M:%S"
                ).astimezone(datetime.datetime.now().astimezone().tzinfo)
                # convert to UTC
                dt_utc = dt_local.astimezone(datetime.timezone.utc)

                locale_en = kindle_kw_locales["en"]
                if clipping_type == locale["info_bookmark"]:
                    clipping_type = locale_en["info_bookmark"]
                elif clipping_type == locale["info_highlight"]:
                    clipping_type = locale_en["info_highlight"]
                elif clipping_type == locale["info_note"]:
                    clipping_type = locale_en["info_note"]
            else:
                self.log.error(f"Unsupported locale: {locale['locale']}")
                return None
            # convert dt_utc to ISO format
            iso_date_utc = dt_utc.strftime("%Y-%m-%dT%H:%M:%S%z")
            self.clippings.append(
                {
                    "title": title,
                    "author": author,
                    "type": clipping_type,
                    "location": clipping_location,
                    "page": page_no,
                    "date": iso_date_utc,
                    "text": clipping_text,
                }
            )
        self.log.info(f"Found {len(self.clippings)} clippings")
        return self.clippings
