import logging
import re


class TimeLines:
    def __init__(self):
        self.log = logging.getLogger("IndraTools")
        self.events = []

    def add_book_events(self, calibre_lib_entries):
        date_regex = r"\b(18|19|20)\d{2}\b"
        n_dates = 0
        n_books = 0
        for entry in calibre_lib_entries:
            if 'doc_text' not in entry:
                continue
            n_books += 1
            text = entry['doc_text']
            # Find all occurences of date_regex in text:
            dates = [(match.start(), match.group()) for match in re.finditer(date_regex, text)]
            # self.text_lib[book_path]['dates'] = dates
            n_dates += len(dates)
            if len(dates) > 0:
                for sample in dates:
                    snip = text[sample[0]-30:sample[0]+10].replace("\n", " ")
                    self.events.append((sample[1], sample[0], snip))
                    n_dates += 1
        self.log.info(f"{n_dates} dates found")
        return n_books, n_dates

    def add_notes_table_events(self, notes, indra, do_timeline, format, timespec, domains, keywords):
        event_cnt = 0
        skipped_cnt = 0
        for note_name in notes:
            note = notes[note_name]
            for table in note["tables"]:
                new_evs, new_skipped = indra.add_events_from_table(
                    table, check_order=True
                )
                event_cnt += new_evs
                skipped_cnt += new_skipped
        self.log.info(
            f"Found {len(indra.events)} (added {event_cnt}) Indra events in notes, skipped {skipped_cnt}"
        )
        if do_timeline is True:
            if format != "ascii":
                format = None
            else:
                format = "ascii"
            time_par = timespec
            if time_par is not None:
                if time_par == "":
                    time_par = None
            domains_par = domains
            if domains_par is not None:
                if domains_par == "":
                    domains_par = None
                else:
                    domains_par = domains_par.split(" ")
            keywords_par = keywords
            if keywords_par is not None:
                if keywords_par == "":
                    keywords_par = None
                else:
                    keywords_par = keywords_par.split(" ")
            evts = indra.search_events(
                time=time_par,
                domains=domains_par,
                keywords=keywords_par,
                in_intervall=False,
                full_overlap=True,
                partial_overlap=False,
            )
            emph_keys = []
            if domains_par is not None:
                for dom in domains_par:
                    if dom.startswith("!"):
                        continue
                    emph_keys.append(dom)
            if keywords_par is not None:
                for k in keywords_par:
                    if k.startswith("!"):
                        continue
                    emph_keys += k.split("|")
            if time_par is not None:
                if len(evts) > 0 and time_par is not None:
                    print(" --------- < ----- > ---------")
                    indra.print_events(evts, format=format, emph_words=emph_keys)
                evts = indra.search_events(
                    time=time_par,
                    domains=domains_par,
                    keywords=keywords_par,
                    in_intervall=False,
                    full_overlap=False,
                    partial_overlap=True,
                )
                if len(evts) > 0:
                    print(" --------- <| ----- |> ---------")
                    indra.print_events(evts, format=format, emph_words=emph_keys)
                evts = indra.search_events(
                    time=time_par,
                    domains=domains_par,
                    keywords=keywords_par,
                    in_intervall=True,
                    full_overlap=False,
                    partial_overlap=False,
                )
                if len(evts) > 0:
                    print(" --------- | ----- | ---------")
                    indra.print_events(evts, format=format, emph_words=emph_keys)
            else:
                indra.print_events(evts, format=format, emph_words=emph_keys)
