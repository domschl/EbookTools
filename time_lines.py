import logging
import os
import re


class TimeLines:
    def __init__(self, book_text_lib):
        self.log = logging.getLogger("IndraTools")
        self.book_text_lib = book_text_lib
        self.events = []
        self.text_lib = {}
        self._read_text_books(self.book_text_lib)

    def _read_text_books(self, book_text_lib):
        self.log.info(f"Reading all books from {book_text_lib}")
        n_books = 0
        for root, dirs, files in os.walk(book_text_lib):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    self.text_lib[file_path] = {}
                    with open(file_path, 'r') as f:
                        # print(f"Reading: {file_path}")
                        self.text_lib[file_path]['text'] = f.read()
                        n_books += 1
        self.log.info(f"{n_books} books read")
        date_regex = r"\b(18|19|20)\d{2}\b"
        n_dates = 0
        self.log.info("Searching year-date occurrences...")
        for book_path in self.text_lib:
            text = self.text_lib[book_path]['text']
            # Find all occurences of date_regex in text:
            dates = [(match.start(), match.group()) for match in re.finditer(date_regex, text)]
            self.text_lib[book_path]['dates'] = dates
            n_dates += len(dates)
            if len(dates) > 0:
                for sample in dates:
                    snip = text[sample[0]-30:sample[0]+10].replace("\n", " ")
                    self.events.append((sample[1], sample[0], snip))
                    n_dates += 1
        self.log.info(f"{n_dates} dates found")
        return n_books, n_dates

    def add_notes_table_events(notes_events):
        pass
