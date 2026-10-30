"""皇冠主盘命中就发 Gmail。

欧洲时间 8:30 通扫当天；9:00–23:00 每个整点扫未来 1.5 小时；
0:00 扫未来 9 小时（过夜到早上）；1:00 再扫 1.5 小时；其后休息。
没命中不发信。同一场每小时提醒只发一次。

本机挂着：python -m src.alert_today --loop
本机定时唤起：python -m src.alert_today --install
盘口：InferSports 免费档，优先皇冠 crown，没有则 sbobet。
电脑关机也能跑：把仓库推到 GitHub，用 Actions 定时（见 .github/workflows/ah-ou-alert.yml）
试发：python -m src.alert_today --demo
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import subprocess
import time
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PROC = ROOT / "data" / "processed"
RULES = ROOT / "data" / "rules" / "ah_ou_alert_rules.csv"
LOG = OUT / "today_alert.txt"
STATE = OUT / "alert_state.json"
ENV_PATH = ROOT / ".env"
_LINE_LABEL = {
    0.0: "平手",
    0.25: "平半",
    0.5: "半球",
    0.75: "半一",
    1.0: "一球",
    1.25: "一球半",
    1.5: "球半",
    1.75: "球半两球",
    2.0: "两球",
    2.25: "两球/两球半",
    2.5: "两球半",
    2.75: "两球半/三球",
    3.0: "三球",
}


def snap(x: float) -> float:
    return round(float(x) * 4) / 4


def describe(line: float) -> str:
    line = snap(line)
    if abs(line) < 1e-9:
        return "平手"
    side = "主让" if line < 0 else "客让"
    key = snap(abs(line))
    return side + _LINE_LABEL.get(key, f"|h|={key}")
INFER_BASE = "https://api.infersports.dev/v1"
BOOK_PREF = ("crown", "sbobet")
PLIST_ID = "com.footballxg.ahou-alert"
EUROPE = ZoneInfo("Europe/Paris")


def load_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_rules() -> pd.DataFrame:
    src = RULES if RULES.exists() else PROC / "ah_ou_line_hits.csv"
    if not src.exists():
        raise SystemExit("没有预警规则。先跑：python -m src.analyze_ah_ou_lines")
    df = pd.read_csv(src)
    return df.sort_values("ROI", ascending=False).drop_duplicates("切片", keep="first").reset_index(drop=True)


def combo_key(ah_line: float, ou_line: float) -> str:
    return describe(ah_line) + " × " + f"{snap(ou_line):g}球"


def pick_main(rows: list[dict], a: str, b: str) -> dict | None:
    scored = []
    for row in rows:
        try:
            ha, hb = float(row[a]), float(row[b])
        except (KeyError, TypeError, ValueError):
            continue
        if ha <= 1.01 or hb <= 1.01:
            continue
        scored.append((abs(ha - hb), abs((ha + hb) / 2 - 1.9), row))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def _headers(api_key: str | None, json_body: bool = False) -> dict[str, str]:
    h = {"User-Agent": "football-xg-alert/1.0", "Accept": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _get(url: str, api_key: str | None = None) -> object:
    req = Request(url, headers=_headers(api_key))
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict, api_key: str) -> object:
    raw = json.dumps(payload).encode("utf-8")
    req = Request(url, data=raw, headers=_headers(api_key, json_body=True), method="POST")
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _team(obj) -> str:
    if isinstance(obj, dict):
        return str(obj.get("name") or "")
    return str(obj or "")


def parse_kick(raw: str) -> datetime | None:
    if not raw:
        return None
    text = str(raw).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()


def europe_now() -> datetime:
    return datetime.now(EUROPE)


def window_for(mode: str) -> tuple[datetime, datetime, str]:
    now = europe_now()
    if mode == "overnight":
        return now, now + timedelta(hours=9), "夜间未来9小时"
    if mode == "hour":
        return now, now + timedelta(minutes=90), "未来1.5小时"
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1), "今日全天"


def _utc_dates(start: datetime, end: datetime) -> list[str]:
    first = start.astimezone(timezone.utc).date()
    last = (end - timedelta(seconds=1)).astimezone(timezone.utc).date()
    out = []
    cur = first
    while cur <= last:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def list_events(api_key: str | None, start: datetime, end: datetime) -> list[dict]:
    events: list[dict] = []
    for day in _utc_dates(start, end):
        cursor = None
        while True:
            q = {"sport": "football", "status": "scheduled", "date": day, "limit": 100}
            if cursor:
                q["cursor"] = cursor
            payload = _get(f"{INFER_BASE}/events?{urlencode(q)}", api_key)
            chunk = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(chunk, list):
                break
            events.extend(chunk)
            cursor = (payload.get("page") or {}).get("next_cursor") if isinstance(payload, dict) else None
            if not cursor:
                break
    kept = []
    for ev in events:
        kick = parse_kick(ev.get("scheduled_at") or "")
        if kick is None or kick < start or kick >= end:
            continue
        kept.append(ev)
    return kept


def pick_book_line(quotes: list[dict], market: str, a: str, b: str) -> dict | None:
    grouped: dict[str, list[dict]] = {}
    for q in quotes:
        if q.get("market_type") != market:
            continue
        if q.get("period") not in (None, "full_time"):
            continue
        if str(q.get("status") or "open") != "open":
            continue
        prices = q.get("prices") or {}
        if q.get("line") is None:
            continue
        row = {"line": q.get("line"), a: prices.get(a), b: prices.get(b), "book": q.get("bookmaker")}
        grouped.setdefault(str(q.get("bookmaker") or ""), []).append(row)
    for book in BOOK_PREF:
        picked = pick_main(grouped.get(book) or [], a, b)
        if picked:
            picked["book"] = book
            return picked
    flat = [row for rows in grouped.values() for row in rows]
    return pick_main(flat, a, b)


def fetch_odds_map(api_key: str | None, event_ids: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if api_key:
        for i in range(0, len(event_ids), 20):
            chunk = event_ids[i : i + 20]
            payload = _post(
                f"{INFER_BASE}/odds/batch",
                {
                    "event_ids": chunk,
                    "bookmakers": ",".join(BOOK_PREF),
                    "markets": "asian_handicap,totals",
                    "period": "full_time",
                },
                api_key,
            )
            for row in payload.get("data") or []:
                eid = str(row.get("event_id") or "")
                block = row.get("odds") or {}
                quotes = block.get("odds") if isinstance(block, dict) else []
                out[eid] = quotes or []
        return out
    for eid in event_ids:
        q = urlencode(
            {
                "bookmakers": ",".join(BOOK_PREF),
                "markets": "asian_handicap,totals",
                "period": "full_time",
            }
        )
        try:
            payload = _get(f"{INFER_BASE}/events/{eid}/odds?{q}", api_key)
        except Exception:
            time.sleep(0.15)
            continue
        out[eid] = payload.get("odds") or []
        time.sleep(0.12)
    return out


def fetch_crown(api_key: str | None, start: datetime, end: datetime) -> list[dict]:
    events = list_events(api_key, start, end)
    ids = [str(ev.get("id")) for ev in events if ev.get("id")]
    odds_map = fetch_odds_map(api_key, ids)
    by_id = {str(ev.get("id")): ev for ev in events}
    out = []
    for eid, quotes in odds_map.items():
        ev = by_id.get(eid) or {}
        ah = pick_book_line(quotes, "asian_handicap", "home", "away")
        ou = pick_book_line(quotes, "totals", "over", "under")
        kick = parse_kick(ev.get("scheduled_at") or "")
        if not ah or not ou or ah.get("line") is None or ou.get("line") is None or kick is None:
            continue
        league = ev.get("league") if isinstance(ev.get("league"), dict) else {}
        out.append(
            {
                "id": eid,
                "league": league.get("name") or "",
                "home": _team(ev.get("home_team")),
                "away": _team(ev.get("away_team")),
                "kickoff": kick.isoformat(timespec="minutes"),
                "book": ah.get("book") or ou.get("book") or "crown",
                "ah_line": snap(float(ah["line"])),
                "ah_home": float(ah["home"]),
                "ah_away": float(ah["away"]),
                "ou_line": snap(float(ou["line"])),
                "over": float(ou["over"]),
                "under": float(ou["under"]),
            }
        )
    return out


def match_alerts(games: list[dict], rules: pd.DataFrame) -> list[dict]:
    by = {str(r["切片"]): r for r in rules.to_dict("records")}
    hits = []
    for g in games:
        key = combo_key(g["ah_line"], g["ou_line"])
        rule = by.get(key)
        if not rule:
            continue
        hits.append({**g, "combo": key, "边": rule["边"], "n": rule["n"], "胜率": rule["胜率"], "ROI": rule["ROI"]})
    return hits


def format_body(hits: list[dict], when: str, label: str) -> str:
    lines = [
        f"皇冠主盘命中预警  {when}  [{label}]",
        f"共 {len(hits)} 场。当前皇冠主盘对照回测过门槛组合。不是稳赚建议。",
        "",
    ]
    for h in hits:
        kick = str(h.get("kickoff") or "")[:16].replace("T", " ")
        lines.append(
            f"{kick}  {h.get('league','')}\n"
            f"  {h['home']} vs {h['away']}\n"
            f"  盘口 {h['combo']}  → 买【{h['边']}】\n"
            f"  亚盘 {h['ah_line']:+g}  主 {h['ah_home']:.2f} / 客 {h['ah_away']:.2f}"
            f"  大小 {h['ou_line']:g}  大 {h['over']:.2f} / 小 {h['under']:.2f}\n"
            f"  回测 n={int(h['n'])}  胜率 {float(h['胜率']):.1%}  ROI {float(h['ROI']):+.1%}\n"
        )
    return "\n".join(lines)


def gmail_defaults(to: str, user: str, host: str, port: int) -> tuple[str, int]:
    gmail = "gmail.com" in (to or "").lower() or "gmail.com" in (user or "").lower()
    if gmail and not host:
        return "smtp.gmail.com", 587 if port in (0, 465) else port
    return host, port


def send_smtp(subject: str, body: str, to: str) -> None:
    user = os.environ.get("SMTP_USER", "").strip() or to
    password = os.environ.get("SMTP_PASS", "").replace(" ", "").strip()
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "0") or 0)
    host, port = gmail_defaults(to, user, host, port)
    if not host or not user or not password:
        raise RuntimeError("smtp-not-configured")
    if port == 0:
        port = 587 if "gmail.com" in host else 465
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=30) as s:
            s.login(user, password)
            s.sendmail(user, [to], msg.as_string())
        return
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        s.login(user, password)
        s.sendmail(user, [to], msg.as_string())


def send_macos_mail(subject: str, body: str, to: str) -> None:
    script = f"""
tell application "Mail"
  set msg to make new outgoing message with properties {{subject:{json.dumps(subject)}, content:{json.dumps(body)}, visible:false}}
  tell msg
    make new to recipient at end of to recipients with properties {{address:{json.dumps(to)}}}
    send
  end tell
end tell
"""
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "macos-mail-failed")


def send_mail(subject: str, body: str) -> str:
    to = os.environ.get("ALERT_TO", "").strip() or os.environ.get("SMTP_USER", "").strip()
    if not to:
        raise SystemExit("请在 .env 里设置 ALERT_TO=你的Gmail")
    try:
        send_smtp(subject, body, to)
        return f"gmail:{to}"
    except RuntimeError as exc:
        if str(exc) != "smtp-not-configured":
            raise
    if os.environ.get("GITHUB_ACTIONS"):
        raise SystemExit("GitHub Actions 必须配置 SMTP_USER / SMTP_PASS（Gmail 应用专用密码）")
    send_macos_mail(subject, body, to)
    return f"mail.app:{to}"


def load_state() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def today_key() -> str:
    return europe_now().strftime("%Y-%m-%d")


def detect_mode() -> str:
    """8:30 早盘；0:00 夜间 9 小时；9:00–23:00 与 1:00 小时盘；其余 rest。"""
    now = europe_now()
    h, m = now.hour, now.minute
    state = load_state()
    day = today_key()
    morning_done = state.get("date") == day and state.get("morning_sent")
    overnight_done = state.get("date") == day and state.get("overnight_sent")
    if h == 8 and m >= 30:
        return "rest" if morning_done else "morning"
    if h == 0 and m <= 10:
        return "rest" if overnight_done else "overnight"
    if h == 1 and m <= 10:
        return "hour"
    if 1 < h < 8 or (h == 1 and m > 10) or (h == 8 and m < 30):
        return "rest"
    if m <= 10 and h >= 9:
        return "hour"
    return "rest"


def next_slot() -> datetime:
    now = europe_now()
    start = now.replace(second=0, microsecond=0)
    cand: list[datetime] = []
    for day in (0, 1):
        base = (start.replace(hour=0, minute=0) + timedelta(days=day))
        cand.append(base.replace(hour=8, minute=30))
        for hour in list(range(9, 24)) + [0, 1]:
            cand.append(base.replace(hour=hour, minute=0))
    future = [t for t in cand if t > now + timedelta(seconds=2)]
    return min(future)


def filter_hourly_dupes(hits: list[dict], state: dict) -> list[dict]:
    day = today_key()
    if state.get("date") != day:
        state.clear()
        state["date"] = day
        state["hourly_ids"] = []
        state["morning_sent"] = False
        state["overnight_sent"] = False
    seen = set(state.get("hourly_ids") or [])
    fresh = [h for h in hits if str(h.get("id")) not in seen]
    state["hourly_ids"] = list(seen | {str(h.get("id")) for h in fresh})
    return fresh


def demo_games() -> list[dict]:
    now = datetime.now().astimezone()
    return [
        {
            "id": 0,
            "league": "DEMO",
            "home": "示例主队",
            "away": "示例客队",
            "kickoff": (now + timedelta(minutes=40)).isoformat(timespec="minutes"),
            "book": "crown",
            "ah_line": -0.75,
            "ah_home": 1.88,
            "ah_away": 1.98,
            "ou_line": 2.0,
            "over": 1.90,
            "under": 1.96,
        }
    ]


def run(mode: str, demo: bool = False) -> None:
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    start, end, label = window_for(mode)
    state = load_state()
    if demo:
        games, book = demo_games(), "DEMO"
    else:
        api_key = (
            os.environ.get("INFERSPORTS_API_KEY", "").strip()
            or os.environ.get("ODDS_API_KEY", "").strip()
        )
        try:
            games = fetch_crown(api_key or None, start, end)
        except Exception as exc:
            raise SystemExit(f"拉盘失败：{exc}") from exc
        books = sorted({str(g.get("book") or "") for g in games if g.get("book")})
        book = ",".join(books) if books else "crown"
    hits = match_alerts(games, rules)
    if mode in ("hour", "overnight") and not demo:
        hits = filter_hourly_dupes(hits, state)
        if mode == "overnight":
            state["overnight_sent"] = True
    elif mode == "morning":
        if state.get("date") != today_key():
            state = {
                "date": today_key(),
                "hourly_ids": [],
                "morning_sent": False,
                "overnight_sent": False,
            }
        state["morning_sent"] = True
        state["date"] = today_key()
        state.setdefault("hourly_ids", [])
        state.setdefault("overnight_sent", False)
    save_state(state)
    when = europe_now().strftime("%Y-%m-%d %H:%M")
    text = (
        format_body(hits, when, label)
        if hits
        else f"{when} [{label}] 扫描 {len(games)} 场（{book or '无'}），没有命中。\n"
    )
    LOG.write_text(text, encoding="utf-8")
    print(text)
    if not hits:
        return
    if mode == "morning":
        tag = "早盘通扫"
    elif mode == "overnight":
        tag = "夜间9小时"
    else:
        tag = "开球前1.5小时"
    sent = send_mail(f"皇冠预警{tag} {len(hits)}场 {when[:10]}", text)
    print("已发送", sent)


def sleep_until_next_slot() -> None:
    nxt = next_slot()
    sec = max(5.0, (nxt - europe_now()).total_seconds())
    print(f"挂起到欧洲时间 {nxt.strftime('%Y-%m-%d %H:%M')}（约 {sec / 60:.0f} 分钟）")
    time.sleep(sec)


def _calendar_intervals() -> str:
    slots = [(8, 30), (0, 0), (1, 0)] + [(h, 0) for h in range(9, 24)]
    blocks = []
    for hour, minute in slots:
        blocks.append(
            "    <dict>\n"
            f"      <key>Hour</key><integer>{hour}</integer>\n"
            f"      <key>Minute</key><integer>{minute}</integer>\n"
            "    </dict>"
        )
    return "\n".join(blocks)


def install_launchd() -> Path:
    dest = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_ID}.plist"
    py = ROOT / ".venv" / "bin" / "python"
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{PLIST_ID}</string>
  <key>WorkingDirectory</key><string>{ROOT}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{py}</string>
    <string>-m</string>
    <string>src.alert_today</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
{_calendar_intervals()}
  </array>
  <key>StandardOutPath</key><string>{ROOT / "output" / "alert_launchd.log"}</string>
  <key>StandardErrorPath</key><string>{ROOT / "output" / "alert_launchd.err"}</string>
</dict>
</plist>
"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(plist, encoding="utf-8")
    subprocess.run(["launchctl", "unload", str(dest)], check=False, capture_output=True)
    subprocess.run(["launchctl", "load", str(dest)], check=True)
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--install", action="store_true", help="本机按欧洲作息整点唤起（电脑要开着）")
    p.add_argument("--loop", action="store_true", help="进程挂着按欧洲时间作息跑")
    p.add_argument("--print-mode", action="store_true", help="只打印 morning/overnight/hour/rest")
    p.add_argument("--mode", choices=("auto", "morning", "overnight", "hour", "rest"), default="auto")
    args = p.parse_args()
    load_dotenv()
    if args.print_mode:
        mode = args.mode if args.mode != "auto" else detect_mode()
        print(mode)
        return
    if args.install:
        dest = install_launchd()
        print(f"已安装本机定时：{dest}")
        print("欧洲 8:30 通扫，0:00 扫未来9小时，9:00–23:00 与 1:00 整点扫未来1.5小时。休息时段不再唤起。电脑睡眠/关机不会跑。")
        return
    if args.loop:
        print("挂起运行（欧洲时间）。Ctrl+C 结束。关机即停。")
        while True:
            mode = detect_mode()
            if mode == "rest":
                sleep_until_next_slot()
                continue
            run(mode, demo=args.demo)
            args.demo = False
            sleep_until_next_slot()
    mode = args.mode if args.mode != "auto" else detect_mode()
    if mode == "rest":
        return
    run(mode, demo=args.demo)


if __name__ == "__main__":
    main()
