"""下载五大联赛：Understat 场级 xG + football-data 收盘亚盘。

亚盘收盘 AHCh 从 2019-20 起才有；Understat 五大联赛可回溯到更早。
season_path 如 2526 = 2025-26，对应 Understat 赛季年 2025。
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests

from .team_map import LEAGUES, SECOND_DIVS, SEASONS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {
    "User-Agent": "Mozilla/5.0 (compatible; football-xg-research/1.0)",
    "Accept": "*/*",
}


def _get(url: str, extra: dict | None = None) -> bytes:
    headers = dict(UA)
    if extra:
        headers.update(extra)
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def fetch_football_data(season_path: str, divs: list[str] | None = None) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for div in divs or list(LEAGUES):
        url = f"https://www.football-data.co.uk/mmz4281/{season_path}/{div}.csv"
        raw = _get(url)
        path = RAW / f"fd_{div}_{season_path}.csv"
        path.write_bytes(raw)
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        print(f"football-data {div} {season_path}: {len(rows)} rows -> {path}")


def fetch_understat(season: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for meta in LEAGUES.values():
        league = meta["understat"]
        enc = quote(league)
        url = f"https://understat.com/getLeagueData/{enc}/{season}"
        raw = _get(
            url,
            {
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"https://understat.com/league/{enc}/{season}",
                "Accept-Encoding": "gzip",
            },
        )
        data = json.loads(raw)
        path = RAW / f"understat_{league.replace(' ', '_')}_{season}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        n = sum(1 for m in data["dates"] if m.get("isResult"))
        print(f"understat {league} {season}: {n}/{len(data['dates'])} played -> {path}")
        time.sleep(1.0)


def main(include_second_div: bool = True) -> None:
    for season_path, us_year in SEASONS:
        fetch_football_data(season_path)
        if include_second_div:
            try:
                fetch_football_data(season_path, list(SECOND_DIVS))
            except Exception as exc:
                print(f"二级联赛 {season_path} 跳过: {exc}")
        fetch_understat(us_year)


if __name__ == "__main__":
    main()
