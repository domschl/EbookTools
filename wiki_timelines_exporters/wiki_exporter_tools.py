import re
import requests
import io
import pandas as pd  # pyright:ignore[reportMissingTypeStubs]
from indralib.indra_time import IndraTime

def remove_footnotes(text:str, numeric_only:bool=True, single_letter_alpha:bool=True) -> str:
    if numeric_only is False:
        text = re.sub(r"\[.*?\]", "", text)
    else:
        text = re.sub(r"\[\d+\]", "", text)
        if single_letter_alpha is True:
            text = re.sub(r"\[\w\]", "", text)
    return text

def extract_date_remarks(date:str) -> tuple[str,str]:
    remarks = ""
    approxies = ["ca.", "c.", "circa", "fl.", "born", "died in", "Died", "died", "buried", "early", "or later", "later", "late", "mid", "or before", "before", 
                 "after", "around", "approximately", "~", ">", "<", "≈", "≥", "≤", "(?)", "?"]
    for ap in approxies:
        if ap in date:
            date = date.replace(ap, "").strip()
            if date != "":
                remarks = f"{ap}"
    if ' or ' in date:
        idx = date.find(' or ')
        remarks = date[idx+1:] 
        date = date[:idx]
    if date.endswith('s'):
        remarks = date
        date = date[:-1]
    if "±" in date:
        dates = date.split("±")
        date = dates[0].strip()
        append = dates[1].strip().split(" ")
        if len(append) > 1:
            date = date + " " + append[1]
            remarks = "±" + append[0]
        else:
            remarks = "±" + dates[1].strip()
    date = date.replace("  ", " ")
    return date, remarks

def pre_clean(date:str) -> str:
    date = remove_footnotes(date)
    date = date.replace("–", " - ").replace("–", "-").replace("—", "-").replace("–", " - ").replace(" ", " ").replace("\u2009", " ") \
               .replace(" to ", " - ").replace(",", "").replace("\xa0", " ").replace("  ", " ") \
               .replace("AD", "").replace("BCE", "BC").replace("  ", " ").replace(" mya", " ma bp").strip()
    return date

def decenturize(date:str) -> tuple[str, str]:
    bc = False
    cent = False
    remarks = ""
    org_date = date
    postf = ['st', 'nd', 'rd', 'th']
    if ' century BC' in date:
        bc = True
        fpf = False
        for pf in postf:
            if pf in date:
                date = date.replace(pf, '')
                fpf = True
                break
        if fpf is True:
            date = date.replace(' century BC', '')
            cent = True
    elif ' century' in date:
        if 'c. ' in date:
            date = date.replace("c. ", "")
        fpf = False
        for pf in postf:
            if pf in date:
                date = date.replace(pf, '')
                fpf = True
                break
        if fpf is True:
            date = date.replace(' century', '')
            cent = True
    if cent is False:
        return org_date, remarks
    century = date.strip()
    try:
        int_cent = int(century)
    except:
        return org_date, remarks
    if bc is True:
        if int_cent == 1:
            date = f"100 BC - 1 BC"
        else:
            date = f"{int_cent}00 BC - {int_cent-1}00 BC"
    else:
        if int_cent == 1:
            date = "1 - 100"
        else:
            date = f"{int_cent-1}00 - {int_cent}00"
    return date, org_date

def check_date_parts(parts:list[str]) -> bool:
    for p in parts:
        for c in p:
            if c <'0' or c>'9':
                return False
    return True

def date_clean(date:str, default_scale:str|None = None):
    remarks = ""
    date = pre_clean(date)
    date_sub = date.split("-")
    if len(date_sub)==3 or (len(date_sub)==2 and " - " not in date and check_date_parts(date_sub)):
        return date, remarks
    if default_scale is not None:
        # date = date.split("-")
        if len(date) == 1:
            date = date_sub[0].strip()+f" {default_scale}"
        else:
            date = f"{date_sub[0].strip()} {default_scale} - {date_sub[1].strip()} {default_scale}"
    else:            
        dates = date.split(" - ")
        new_dates: list[str] = []
        for di in dates:
            dj, rem = extract_date_remarks(di)
            if len(rem)>0:
                if len(remarks) == 0:
                    remarks = rem
                else:
                    remarks += f", {rem}"
            dj, rem = decenturize(dj)
            if len(rem)>0:
                if len(remarks) == 0:
                    remarks = rem
                else:
                    remarks += f", {rem}"
            if "/" in dj:
                dj_parts = dj.split("/")
                dj = dj_parts[0].strip()
                alt_dj_stub = dj_parts[1].strip()
                alt_dj = dj[:len(dj)-len(alt_dj_stub)] + alt_dj_stub
                rem = f"Alt.: {alt_dj}"       
                if len(remarks) == 0:
                    remarks = rem
                else:
                    remarks += f", {rem}"
            new_dates.append(dj)
        date = " - ".join(new_dates)
        dates = date.split(" - ")
        if len(dates) == 2:
            sub_dates0 = dates[0].split(" ")
            sub_dates1 = dates[1].split(" ")
            if len(sub_dates0)==1 and len(sub_dates1)==2:
                date = f"{sub_dates0[0]} {sub_dates1[1]} - {dates[1]}"
    # else:
    #     if default_scale is not None:
    #         date = f"{date} {default_scale}"
    #     else:
    #         date = str(int(date))
    jd_dates = IndraTime.string_time_to_julian(date)

    if jd_dates is not None:
        if len(jd_dates) > 1 and jd_dates[1] is not None:
            date_start = IndraTime.julian_to_string_time(jd_dates[0])
            date_end = IndraTime.julian_to_string_time(jd_dates[1])
            date = f"{date_start} - {date_end}"
        else:
            date = IndraTime.julian_to_string_time(jd_dates[0])

    return date, remarks

def date_merge(year:str, rest:str) -> str:
    day = None
    month = None
    year = pre_clean(year)
    rest = rest.strip()
    year = year.strip()
    if len(rest) > 0:
        rparts = rest.split(" ")
        if len(rparts) == 1:
            month = rparts[0].strip().lower()
        else:
            try:
                month = rparts[1].strip().lower()
                day = int(rparts[0].strip())
            except ValueError:
                month = None
                day = None
            if month is None and day is None:
                try:
                    month = rparts[0].strip().lower()
                    day = int(rparts[1].strip())
                except ValueError:
                    month = None
                    day = None
    else:
        month = None
        day = None
    if month is not None:
        val_months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        try:
            month_id = val_months.index(month) + 1
        except ValueError:
            month_id = None
            return f"{year}"
        year_parts = str(year).replace("\xa0"," ").split(" ", 1)
        if len(year_parts) > 1:
            year = year_parts[0]
            appendix = year_parts[1]
        else:
            appendix = ""
        if day is not None:
            date = f"{year}-{month_id:02d}-{day:02d}"
            if appendix != "":
                date = date + f" {appendix}"
        else:
            date = f"{year}-{month_id:02d}"
            if appendix != "":
                date = date + f" {appendix}"
        return date
    else:
        return f"{year}"

def read_wiki_data_as_pandas(url:str) -> list[pd.DataFrame]|None:
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Request to {url} failed: {e}")
        print(f"Status: {e.response.status_code}")
        print(f"Text:   {e.response.text}")
        return None
        
    f: io.StringIO = io.StringIO(response.text)
    dfs = pd.read_html(f)  # pyright:ignore[reportUnknownMemberType]
    return dfs
