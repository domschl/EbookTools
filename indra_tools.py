import logging
from indralib.indra_time import IndraTime


class IndraTools:
    def __init__(self):
        self.log = logging.getLogger("IndraTools")
        self.events = []

    def add_events_from_table(self, table, metadata):
        event_cnt = 0
        if "columns" not in table or "rows" not in table:
            if "columns" not in table:
                self.log.warning(f"Table has no 'columns', skipping")
            else:
                self.log.warning(f"Table {table['columns']} has no 'rows', skipping")
            return event_cnt
        if len(table["columns"]) < 2:
            self.log.warning(
                f"Table {table['columns']} has less than 2 columns, skipping"
            )
            return event_cnt
        col_nr = len(table["columns"])
        for row in table["rows"]:
            if len(row) != col_nr:
                self.log.warning(
                    f"Table {table['columns']}: Row {row} has {len(row)} columns, expected {col_nr}, invalid Table"
                )
                return event_cnt
        if table["columns"][0] != "Date":
            self.log.info(
                f"Table {table['columns']}: First column is not 'Date', skipping"
            )
            return event_cnt
        if "domain" not in metadata:
            self.log.warning(f"Table {table['columns']}: Metadata has no 'domain' key")
        for row in table["rows"]:
            raw_date = row[0]
            try:
                jd_date = IndraTime.string_time_2_julian(raw_date)
            except ValueError:
                self.log.warning(
                    f"Table {table['columns']}: Row {row} has invalid date {raw_date}"
                )
                continue
            event_data = {}
            for i in range(1, col_nr):
                event_data[table["columns"][i]] = row[i]
            event = [jd_date, event_data, metadata]
            self.events.append(event)
            event_cnt += 1
        # Sort
        self.events = sorted(self.events, key=lambda x: x[0])
        return event_cnt

    def print_event(self):
        for event in self.events:
            date_points = []
            event_text = ""
            for ev in event[1]:
                event_text += f"{ev}: {event[1][ev]}, "
            event_text = event_text[:-2]
            if len(event_text) > 100:
                event_text = event_text[:100] + "..."
            for date_part in event[0]:
                date_points.append(IndraTime.julian_2_string_time(date_part))
            if len(date_points) == 1:
                self.log.info(f"{date_points[0]}: {event_text}")
            elif len(date_points) == 2:
                self.log.info(f"{date_points[0]} - {date_points[1]}: {event_text}")
            else:
                self.log.warning(f"Invalid date range: {date_points}: {event_text}")
        return
