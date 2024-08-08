import logging
from indralib.indra_time import IndraTime


class IndraTools:
    def __init__(self):
        self.log = logging.getLogger("IndraTools")
        self.events = []

    def add_events_from_table(self, table, metadata):
        event_cnt = 0
        if "columns" not in table or "rows" not in table:
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
            if "domain" not in metadata:
                self.log.warning(
                    f"Table {table['columns']}: Metadata has no 'domain' key"
                )
            self.events.append(event)
            event_cnt += 1
        # Sort
        self.events = sorted(self.events, key=lambda x: x[0])
        return event_cnt
