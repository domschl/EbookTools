import logging
import os
import re
from rich import print as rprint
from indralib.indra_time import IndraTime  # type: ignore
from indralib.indra_event import IndraEvent  # type: ignore


class IndraTools:
    def __init__(self):
        self.log = logging.getLogger("IndraTools")
        self.events = []
        self.domains = []

    def add_events_from_table(self, table, check_order=True):
        event_cnt = 0
        events_skipped = 0
        if "columns" not in table or "rows" not in table or "metadata" not in table:
            if "columns" not in table:
                self.log.warning("Table has no 'columns', skipping")
            elif "rows" not in table:
                self.log.warning(f"Table {table['columns']} has no 'rows', skipping")
            elif "metadata" not in table:
                self.log.warning(
                    f"Table {table['columns']} has no 'metadata', skipping"
                )
            else:
                self.log.warning(f"Table {table['columns']}: Invalid Table, skipping")
            return event_cnt, 0
        if len(table["columns"]) < 2:
            self.log.debug(
                f"Table {table['columns']}, {table['metadata']} has less than 2 columns, skipping"
            )
            return event_cnt, 0
        col_nr = len(table["columns"])
        for row in table["rows"]:
            if len(row) != col_nr:
                self.log.warning(
                    f"Table {table['columns']}: Row {row} has {len(row)} columns, expected {col_nr}, invalid Table"
                )
                return event_cnt, 0
        if len(table["columns"]) == 0 or table["columns"][0] != "Date":
            self.log.debug(
                f"Table {table['columns']}: First column is not 'Date', skipping"
            )
            return event_cnt, 0
        if "domain" not in table["metadata"]:
            self.log.debug(
                f"Table {table['columns']}: Metadata has no 'domain' key, skipping"
            )
            return event_cnt, 0
        if table["metadata"]["domain"] in self.domains:
            self.log.warning(
                f"Table {table['columns']}: Domain {table['metadata']['domain']} already exists, skipping"
            )
            return event_cnt, 0
        self.domains.append(table["metadata"]["domain"])
        last_start_time = None
        last_end_time = None
        table_sorted = True
        for row in table["rows"]:
            raw_date = row[0]
            try:
                jd_date = IndraTime.string_time_to_julian(raw_date)
                # print(jd_date)
                for date_part in jd_date:
                    if date_part is None:
                        self.log.warning(
                            f"Table {table['columns']}: Row {row} has invalid date {raw_date}"
                        )
                        jd_date = None
                if jd_date is None:
                    events_skipped += 1
                    continue
                if len(jd_date) == 2:
                    if jd_date[1] < jd_date[0]:
                        self.log.error(
                            f"Table {table['columns']}: Row {row}: end-date is earlier than start-state, invalid!"
                        )
                        jd_date = None
                        events_skipped += 1
                        continue
            except ValueError:
                self.log.warning(
                    f"Table {table['columns']}: Row {row} has invalid date {raw_date}"
                )
                events_skipped += 1
                continue
            if last_start_time is not None:
                if last_start_time > jd_date[0]:
                    events_skipped += 1
                    table_sorted = False
                    if check_order is True:
                        self.log.error(
                            f"Table {table['columns']}: Row {row}: start-date is later than start-state of previous row, invalid order!"
                        )
                        self.log.warning(
                            f"{IndraTime.julian_to_string_time(last_start_time)} -> {IndraTime.julian_to_string_time(jd_date[0])}, {events_skipped}"
                        )
                    continue
                elif last_start_time == jd_date[0]:
                    if last_end_time is not None:
                        if len(jd_date) == 1:
                            events_skipped += 1
                            table_sorted = False
                            if check_order is True:
                                self.log.error(
                                    f"Table {table['columns']}: Row {row}: interval-less record is later than interval with same start-date, it should be before, invalid order!"
                                )
                                continue
                        elif last_end_time > jd_date[1]:
                            events_skipped += 1
                            table_sorted = False
                            if check_order is True:
                                self.log.error(
                                    f"Table {table['columns']}: Row {row}: intervals with same start-date, record with earlier end-date after later end-date, invalid order!"
                                )
                                continue
            last_start_time = jd_date[0]
            if len(jd_date) > 1:
                last_end_time = jd_date[1]
            else:
                last_end_time = None
            event_data = {}
            for i in range(1, col_nr):
                event_data[table["columns"][i]] = row[i]
            event = [jd_date, event_data, table["metadata"]]
            self.events.append(event)
            event_cnt += 1

        def jd_str_interval_sorter(row):
            jdi = IndraTime.string_time_to_julian(row[0])
            if len(jdi) == 1:
                jdi += jdi
            return (jdi[0], jdi[1])

        if table_sorted is False:
            sorted_table = sorted(table["rows"], key=jd_str_interval_sorter)
            print()
            print("-----------------------------------------------------------------")
            self.print_table(table["columns"], sorted_table)
            print("-----------------------------------------------------------------")
            print()

        def jd_interval_sorter(jds):
            if len(jds) == 1:
                ls = jds + jds
            else:
                ls = jds
            return (ls[0], ls[1])

        # Sort
        self.events = sorted(self.events, key=lambda x: jd_interval_sorter(x[0]))
        return event_cnt, events_skipped

    def print_table(self, columns, rows):
        print("|", end="")
        for column in columns:
            print(f" {column} |", end="")
        print()
        print("|", end="")
        for _ in range(len(columns)):
            print(" ----- |", end="")
        print()
        for row in rows:
            print("|", end="")
            for col in row:
                print(f" {col} |", end="")
            print()

    def search_keys(self, text, keys):
        result = []
        s_text = text.lower()
        found = False
        for key in keys:
            if key.startswith("!"):
                continue
            or_keys = key.split("|")
            or_found = False
            for key in or_keys:
                if key.startswith("*"):
                    key = key[1:]
                else:
                    key = r"\b" + key
                if key.endswith("*"):
                    key = key[:-1]
                else:
                    key += r"\b"
                key = key.lower().replace("*", r".*")
                if re.search(key, s_text):
                    result.append(key)
                    or_found = True
            if or_found is False:
                found = False
                return False
            else:
                found = True
        if found:
            for key in keys:
                if key.startswith("!") is False:
                    continue
                key = key[1:]
                if key.startswith("*"):
                    key = key[1:]
                else:
                    key = r"\b" + key
                if key.endswith("*"):
                    key = key[:-1]
                else:
                    key += r"\b"
                key = key.lower().replace("*", r".*")
                if re.search(key, s_text):
                    return False
        return found

    def search_events(
        self,
        time=None,
        domains=None,
        keywords=None,
        in_intervall=True,
        full_overlap=True,
        partial_overlap=True,
    ):
        if time is not None:
            if time.startswith('"') and time.endswith('"'):
                time = time[1:-1]
            time = IndraTime.string_time_to_julian(time)
            start_time = time[0]
            if len(time) > 1 and time[1] is not None:
                end_time = time[1]
            else:
                end_time = start_time
        else:
            start_time = None
            end_time = None
        if domains is not None:
            if not isinstance(domains, list):
                domains = [domains]
        if keywords is not None:
            if not isinstance(keywords, list):
                keywords = [keywords]
        result = []
        for event in self.events:
            if keywords is None:
                b_keywords = True
            else:
                b_keywords = False
                for key in event[1]:
                    if self.search_keys(key, keywords) or self.search_keys(event[1][key], keywords):
                        b_keywords = True
                        break
                if b_keywords is False:
                    for key in event[2]:
                        if self.search_keys(key, keywords) or self.search_keys(event[2][key], keywords):
                            b_keywords = True
                            break
            if not b_keywords:
                continue
            if domains is None:
                b_domains = True
            else:
                b_domains = False
                pos_dom = False
                for domain in domains:
                    if domain.startswith("!"):
                        continue
                    pos_dom = True
                    if domain.lower() in event[2]["domain"].lower() or IndraEvent.mqcmp(
                        domain.lower(), event[2]["domain"].lower()
                    ):
                        b_domains = True
                        break
                if pos_dom is False:
                    b_domains = True
                if b_domains is True:
                    for domain in domains:
                        if domain.startswith("!") is False:
                            continue
                        else:
                            domain = domain[1:]
                        if domain.lower() in event[2][
                            "domain"
                        ].lower() or IndraEvent.mqcmp(
                            domain.lower(), event[2]["domain"].lower()
                        ):
                            b_domains = False
                            break
            if not b_domains:
                continue
            if time is not None:
                b_time = False
                event_start = event[0][0]
                if len(event[0]) > 1 and event[0][1] is not None:
                    event_end = event[0][1]
                else:
                    event_end = event[0][0]
                if event_start >= start_time and event_end <= end_time:
                    if in_intervall:
                        b_time = True
                else:
                    if event_start < start_time and event_end > end_time:
                        if full_overlap:
                            b_time = True
                    else:
                        if partial_overlap:
                            if (
                                event_start >= start_time
                                and event_start < end_time
                                and event_end > end_time
                            ):
                                b_time = True
                            if (
                                event_end <= end_time
                                and event_end > start_time
                                and event_start < start_time
                            ):
                                b_time = True
                if not b_time:
                    continue
            result.append(event)
        return result

    def _get_date_string_from_event(self, ev):
        date_points = []
        for date_part in ev:
            date_points.append(IndraTime.julian_to_string_time(date_part))
            date = None
        if len(date_points) == 1:
            date = date_points[0]
        elif len(date_points) == 2:
            date = f"{date_points[0]} - {date_points[1]}"
        else:
            self.log.warning(f"Invalid date range: {date_points}")
            date = None
        return date

    def _get_event_text(self, ev1):
        event_text = ""
        for ev in ev1:
            event_text += f"{ev}: {ev1[ev]}, "
        if len(event_text) >= 2:
            event_text = event_text[:-2]
        return event_text

    def print_events(
        self,
        events,
        filename=None,
        length=None,
        header=False,
        format=None,
        emph_words=[],
    ):
        emph_words_no_esc = []
        for we in emph_words:
            if we.startswith("*"):
                we = we[1:]
            if we.endswith("*"):
                we = we[:-1]
            if "*" in we:
                we = we.split("*")
                for w in we:
                    if len(w) > 0:
                        emph_words_no_esc.append(w)
            else:
                if len(we) > 0:
                    emph_words_no_esc.append(we)
        if filename is not None:
            f = open(filename, "w")
        else:
            f = None
        if format is None:
            if header is True:
                if f is not None:
                    f.write("| Date                      | Event |\n")
                    f.write("|---------------------------|-------|\n")
                else:
                    print("| Date                      | Event |")
                    print("|---------------------------|-------|")
            for event in events:
                event_text = self._get_event_text(event[1])
                if length is not None and len(event_text) > length:
                    event_text = event_text[:length] + "..."
                date_text = self._get_date_string_from_event(event[0])
                if date_text is not None:
                    if f is not None:
                        f.write(f"| {date_text:24s} | {event_text} |\n")
                    else:
                        print(f"| {date_text:24s} | {event_text} |")
        elif format == "ascii":
            max_date = 0
            for event in events:
                date_text = self._get_date_string_from_event(event[0])
                if date_text is not None and len(date_text) > max_date:
                    max_date = len(date_text)
            if length is not None:
                width = length
            else:
                width = os.get_terminal_size()[0]
            if max_date + 7 + 20 > width:
                self.log.error("More width required for output!")
                return
            max_text = width - max_date - 7
            for event in events:
                date_text = self._get_date_string_from_event(event[0])
                event_text = self._get_event_text(event[1])
                while date_text is not None and (
                    len(date_text) > 0 or len(event_text) > 0
                ):
                    if len(event_text) > max_text:
                        br0 = max_text
                        br1 = max_text
                        for i in range(max_text, max_text // 2, -1):
                            if event_text[i] == " ":
                                br0 = i
                                br1 = i + 1
                                break
                            if event_text[i - 1] == "-":
                                br0 = i
                                br1 = i
                                break
                        evt = event_text[:br0]
                        event_text = event_text[br1:]
                    else:
                        evt = event_text
                        event_text = ""
                    if f is not None:
                        f.write(f"| {date_text:{max_date}s} | {evt:{max_text}s} |")
                    else:
                        sep = "┇"
                        ec = 4
                        max2 = max_text
                        cs0 = f"[color({ec})]"
                        cs1 = "[/]"
                        for ew in emph_words_no_esc:
                            ew = ew.lower()
                            if ew in cs0 or ew in cs1:
                                continue
                            evtl = evt.lower()
                            ind = evtl.find(ew)
                            evtn = evt
                            if ind != -1:
                                evtn = (
                                    evt[:ind]
                                    + cs0
                                    + evt[ind: ind + len(ew)]
                                    + cs1
                                    + evt[ind + len(ew):]
                                )
                            max2 += len(evtn) - len(evt)
                            evt = evtn
                        if date_text == "":
                            bg = 231
                        else:
                            bg = 230
                        esc_print = f"[black on color({bg})]{sep} {date_text:{max_date}s} {sep} {evt:{max2}s} {sep}[/]"
                        try:
                            rprint(esc_print)
                        except Exception as e:
                            print(f"Failed to print: >{esc_print}<, because of {e}")
                    date_text = ""
        if f is not None:
            f.close()
        return
