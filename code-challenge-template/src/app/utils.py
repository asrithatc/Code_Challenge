# src/app/utils.py

from datetime import datetime

def parse_val(v):
    v = int(v)
    return None if v == -9999 else v

def parse_wx_line(line: str):
    date_str, mx, mn, pr = line.strip().split("\t")

    date = datetime.strptime(date_str, "%Y%m%d").date()

    max_t = parse_val(mx)
    min_t = parse_val(mn)
    prec = parse_val(pr)

    # Convert tenths to real units
    if max_t is not None:
        max_t /= 10
    if min_t is not None:
        min_t /= 10
    if prec is not None:
        prec = (prec / 10) / 10   # mm → cm

    return date, max_t, min_t, prec

