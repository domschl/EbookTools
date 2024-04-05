import os
import logging
from ebook_utils import sanitized_md_filename


class MdTools:
    def __init__(self, notes_books_folder):
        self.log = logging.getLogger("MdTools")
        self.notes_books_folder = notes_books_folder

    def get_books(self):
        self.books = []
        for root, dirs, files in os.walk(self.notes_books_folder):
            for file in files:
                if file.endswith(".md"):
                    self.books.append(os.path.join(root, file))
        self.log.info(f"Found {len(self.books)} existing markdown notes on books")
        return self.books


def md_filename(self, name):
    name = sanitized_md_filename(name)
    md_note_filename = os.path.join(self.notes_books_folder, name + ".md")
    return md_note_filename

    def match_md_note_filename(self, md_note_filename):
        for book in self.books:
            if md_note_filename == book:
                self.log.info(f"exact match: {md_note_filename}")
                return md_note_filename
        for book in books:
            if (
                md_note_filename[:-3] == book[: len(md_note_filename[:-3])]
                or md_note_filename[: len(book[:-3])] == book[:-3]
            ):
                self.log.info(
                    f"Warning: Book filename {md_note_filename} does not match {book} exactly"
                )
                return book
        self.log.info(f"no match: {md_note_filename}")
        return None
