"""Cheap header filter. It runs on raw text, so rejected games are never replayed."""
import re

HDR = re.compile(r'\[(\w+) "([^"]*)"\]')
MIN_RATING = 1800
MIN_BASE_SECONDS = 180  # 3+0 or slower


def headers(raw):
    return dict(HDR.findall(raw.split("\n\n", 1)[0]))


def keep(h, min_rating=MIN_RATING):
    if h.get("Event", "").find("Rated") < 0 or h.get("Termination") != "Normal":
        return False
    try:
        if min(int(h["WhiteElo"]), int(h["BlackElo"])) < min_rating:
            return False
        base = int(h["TimeControl"].split("+")[0])
    except (KeyError, ValueError):
        return False
    return base >= MIN_BASE_SECONDS
