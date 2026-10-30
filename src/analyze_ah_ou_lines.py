"""收盘亚盘主盘 × 收盘大小球主盘：哪组球头组合胜率 / ROI 更高。

数据：data/kaggle_odds/match_closing.parquet（真实 2 / 2.25 / 2.75 / 3 … 档，不是固定 2.5）。
只用皇冠（数据集里的 Singbet）收盘主盘，不用 Pinnacle / Bet365。
不依赖 Understat / football-data 的 2.5 大小球。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_ah_form_move import describe, snap
from .analyze_ah_l3_slices import pnl
from .analyze_ah_xg import OUT, PROC, ROOT, settle_ah_home, wilson_ci

KAGGLE_LOCAL = PROC.parent / "kaggle_odds"
KAGGLE_HUB = "archangel0x01/football-odds-movements-2018-2026"
BOOK = "Singbet"  # 皇冠
MIN_N = 120
HIT_MIN = 0.55
ROI_MIN = 0.03


def _col(df: pd.DataFrame, *cands: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in cands:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _need(df: pd.DataFrame, *cands: str) -> str:
    col = _col(df, *cands)
    if col is None:
        raise KeyError(f"找不到列 {cands}；现有列: {list(df.columns)}")
    return col


def dataset_dir() -> Path:
    for cand in (KAGGLE_LOCAL,):
        if (cand / "match_closing.parquet").exists() or (cand / "match_closing.csv").exists():
            return cand
    try:
        import kagglehub

        cached = Path(kagglehub.dataset_download(KAGGLE_HUB))
        if (cached / "match_closing.parquet").exists() or (cached / "match_closing.csv").exists():
            return cached
    except Exception:
        pass
    raise SystemExit(
        "找不到 Kaggle 数据。先运行：python -c \"import kagglehub; print(kagglehub.dataset_download("
        "'archangel0x01/football-odds-movements-2018-2026'))\""
    )


def _read_table(folder: Path, stem: str) -> pd.DataFrame:
    pq = folder / f"{stem}.parquet"
    csv = folder / f"{stem}.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except ImportError:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(stem)


def load_closing() -> pd.DataFrame:
    folder = dataset_dir()
    raw = _read_table(folder, "match_closing")
    match_path = folder / "matches.parquet"
    if not match_path.exists():
        match_path = folder / "matches.csv"

    book_col = _need(raw, "bookmaker", "book", "Book")
    mid = _need(raw, "match_id", "Match_ID")
    ah_line = _need(raw, "close_ah_line", "ah_close_line", "close_handicap")
    ah_home = _need(raw, "close_ah_home", "close_ah_home_odds", "ah_close_home")
    ah_away = _need(raw, "close_ah_away", "close_ah_away_odds", "ah_close_away")
    ou_line = _need(raw, "close_ou_line", "close_total", "ou_close_line", "close_ou")
    ou_over = _need(raw, "close_ou_over", "close_over", "ou_close_over")
    ou_under = _need(raw, "close_ou_under", "close_under", "ou_close_under")

    d = raw.rename(
        columns={
            mid: "match_id",
            book_col: "bookmaker",
            ah_line: "close_line",
            ah_home: "ah_home_odds",
            ah_away: "ah_away_odds",
            ou_line: "ou_line",
            ou_over: "over_odds",
            ou_under: "under_odds",
        }
    ).copy()
    for c in ("close_line", "ah_home_odds", "ah_away_odds", "ou_line", "over_odds", "under_odds"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close_line", "ou_line", "ah_home_odds", "ah_away_odds", "over_odds", "under_odds"])
    d = d[
        (d["ah_home_odds"] > 1.01)
        & (d["ah_away_odds"] > 1.01)
        & (d["over_odds"] > 1.01)
        & (d["under_odds"] > 1.01)
    ]

    d = d[d["bookmaker"] == BOOK].drop_duplicates("match_id", keep="first")

    if match_path.exists():
        m = _read_table(folder, "matches")
        m_id = _need(m, "match_id", "Match_ID")
        hg = _need(m, "home_goals", "ft_home", "home_score", "FTHG", "score_home")
        ag = _need(m, "away_goals", "ft_away", "away_score", "FTAG", "score_away")
        keep = [m_id, hg, ag]
        league = _col(m, "league_name", "competition", "league", "tournament")
        kick = _col(m, "kickoff_utc", "kickoff", "date", "match_date")
        season = _col(m, "season")
        for extra in (league, kick, season):
            if extra:
                keep.append(extra)
        m = m[keep].rename(
            columns={
                m_id: "match_id",
                hg: "FTHG",
                ag: "FTAG",
                **({league: "league"} if league else {}),
                **({kick: "date"} if kick else {}),
                **({season: "season"} if season else {}),
            }
        )
        d = d.merge(m, on="match_id", how="inner")
    else:
        hg = _col(d, "home_goals", "ft_home", "FTHG")
        ag = _col(d, "away_goals", "ft_away", "FTAG")
        if not hg or not ag:
            raise SystemExit("match_closing 里没有比分，请同时放入 matches.parquet")
        d["FTHG"] = pd.to_numeric(d[hg], errors="coerce")
        d["FTAG"] = pd.to_numeric(d[ag], errors="coerce")

    d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
    d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
    d = d.dropna(subset=["FTHG", "FTAG"]).copy()
    d["gd"] = d["FTHG"] - d["FTAG"]
    d["goals"] = d["FTHG"] + d["FTAG"]
    d["close_line"] = d["close_line"].map(snap)
    d["ou_line"] = d["ou_line"].map(snap)
    d["home_cover"] = [settle_ah_home(g, ln) for g, ln in zip(d["gd"], d["close_line"])]
    d["over_cover"] = [settle_ah_home(g, -ln) for g, ln in zip(d["goals"], d["ou_line"])]
    d["home_hit"] = (d["home_cover"] > 0).astype(float)
    d["over_hit"] = (d["over_cover"] > 0).astype(float)
    d["pnl_home"] = pnl(d["home_cover"], d["ah_home_odds"])
    d["pnl_away"] = pnl(-d["home_cover"], d["ah_away_odds"])
    d["pnl_over"] = pnl(d["over_cover"], d["over_odds"])
    d["pnl_under"] = pnl(-d["over_cover"], d["under_odds"])

    fav = np.where(d["close_line"] < 0, "home", np.where(d["close_line"] > 0, "away", "pickem"))
    pickem = fav == "pickem"
    fav = np.where(pickem & (d["ah_home_odds"] < d["ah_away_odds"]), "home", fav)
    fav = np.where(pickem & (d["ah_away_odds"] < d["ah_home_odds"]), "away", fav)
    d["favorite"] = fav
    d = d[d["favorite"] != "pickem"].copy()
    d["favorite_cover"] = np.where(d["favorite"] == "home", d["home_cover"], -d["home_cover"])
    d["up_hit"] = (d["favorite_cover"] > 0).astype(float)
    d["fav_odds"] = np.where(d["favorite"] == "home", d["ah_home_odds"], d["ah_away_odds"])
    d["dog_odds"] = np.where(d["favorite"] == "home", d["ah_away_odds"], d["ah_home_odds"])
    d["pnl_up"] = pnl(d["favorite_cover"], d["fav_odds"])
    d["pnl_dn"] = pnl(-d["favorite_cover"], d["dog_odds"])
    d["ah_lab"] = d["close_line"].map(describe)
    d["ou_lab"] = d["ou_line"].map(lambda x: f"{x:g}球")
    d["combo"] = d["ah_lab"] + " × " + d["ou_lab"]
    d["abs_ah"] = d["close_line"].abs()
    return d.reset_index(drop=True)


def eval_side(sub: pd.DataFrame, side: str) -> dict | None:
    n = len(sub)
    if n < MIN_N:
        return None
    mapping = {
        "主队盘": ("home_hit", "pnl_home"),
        "客队盘": (None, "pnl_away"),
        "上盘": ("up_hit", "pnl_up"),
        "下盘": (None, "pnl_dn"),
        "大球": ("over_hit", "pnl_over"),
        "小球": (None, "pnl_under"),
        "上盘+大球": (None, None),
        "上盘+小球": (None, None),
        "下盘+大球": (None, None),
        "下盘+小球": (None, None),
        "主队盘+大球": (None, None),
        "主队盘+小球": (None, None),
        "客队盘+大球": (None, None),
        "客队盘+小球": (None, None),
    }
    if side == "客队盘":
        hit = (sub["home_cover"] < 0).astype(float)
        roi = sub["pnl_away"]
    elif side == "下盘":
        hit = (sub["favorite_cover"] < 0).astype(float)
        roi = sub["pnl_dn"]
    elif side == "小球":
        hit = (sub["over_cover"] < 0).astype(float)
        roi = sub["pnl_under"]
    elif side == "上盘+大球":
        hit = ((sub["favorite_cover"] > 0) & (sub["over_cover"] > 0)).astype(float)
        roi = (sub["pnl_up"] + sub["pnl_over"]) / 2
    elif side == "上盘+小球":
        hit = ((sub["favorite_cover"] > 0) & (sub["over_cover"] < 0)).astype(float)
        roi = (sub["pnl_up"] + sub["pnl_under"]) / 2
    elif side == "下盘+大球":
        hit = ((sub["favorite_cover"] < 0) & (sub["over_cover"] > 0)).astype(float)
        roi = (sub["pnl_dn"] + sub["pnl_over"]) / 2
    elif side == "下盘+小球":
        hit = ((sub["favorite_cover"] < 0) & (sub["over_cover"] < 0)).astype(float)
        roi = (sub["pnl_dn"] + sub["pnl_under"]) / 2
    elif side == "主队盘+大球":
        hit = ((sub["home_cover"] > 0) & (sub["over_cover"] > 0)).astype(float)
        roi = (sub["pnl_home"] + sub["pnl_over"]) / 2
    elif side == "主队盘+小球":
        hit = ((sub["home_cover"] > 0) & (sub["over_cover"] < 0)).astype(float)
        roi = (sub["pnl_home"] + sub["pnl_under"]) / 2
    elif side == "客队盘+大球":
        hit = ((sub["home_cover"] < 0) & (sub["over_cover"] > 0)).astype(float)
        roi = (sub["pnl_away"] + sub["pnl_over"]) / 2
    elif side == "客队盘+小球":
        hit = ((sub["home_cover"] < 0) & (sub["over_cover"] < 0)).astype(float)
        roi = (sub["pnl_away"] + sub["pnl_under"]) / 2
    else:
        hit_col, roi_col = mapping[side]
        hit = sub[hit_col]
        roi = sub[roi_col]

    k = int(hit.sum())
    p = k / n
    lo, hi = wilson_ci(k, n)
    roi_m = float(roi.mean())
    se = float(roi.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {
        "n": n,
        "边": side,
        "胜率": p,
        "CI低": lo,
        "CI高": hi,
        "ROI": roi_m,
        "ROI se": se,
    }


SIDES = (
    "主队盘",
    "客队盘",
    "上盘",
    "下盘",
    "大球",
    "小球",
    "上盘+大球",
    "上盘+小球",
    "下盘+大球",
    "下盘+小球",
    "主队盘+大球",
    "主队盘+小球",
    "客队盘+大球",
    "客队盘+小球",
)


def scan(df: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(key, observed=True):
        for side in SIDES:
            rec = eval_side(sub, side)
            if rec:
                rec["切片"] = str(name)
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def show(df: pd.DataFrame, n: int = 20) -> str:
    if df is None or df.empty:
        return "(没有)"
    cols = [c for c in ["切片", "边", "n", "胜率", "CI低", "ROI", "ROI se"] if c in df.columns]
    return df[cols].head(n).round(4).to_string(index=False)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_closing()
    grid = scan(d, "combo")
    by_ou = scan(d, "ou_lab")
    by_ah = scan(d, "ah_lab")
    grid.to_csv(PROC / "ah_ou_line_combo.csv", index=False)

    hits = grid[(grid["胜率"] >= HIT_MIN) & (grid["ROI"] >= ROI_MIN) & (grid["n"] >= MIN_N)]
    hits = hits.sort_values(["ROI", "胜率", "n"], ascending=[False, False, False])
    hits.to_csv(PROC / "ah_ou_line_hits.csv", index=False)
    rules_dir = ROOT / "data" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    unique = hits.sort_values("ROI", ascending=False).drop_duplicates("切片")
    unique.to_csv(rules_dir / "ah_ou_alert_rules.csv", index=False)

    # 同一切片只留 ROI 最高的一边，避免四边一起刷屏
    best = (
        grid.sort_values("ROI", ascending=False).drop_duplicates("切片")
        if not grid.empty
        else grid
    )
    best_hits = (
        hits.sort_values("ROI", ascending=False).drop_duplicates("切片") if not hits.empty else hits
    )

    dist = (
        d.groupby(["ah_lab", "ou_lab"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    dist.to_csv(PROC / "ah_ou_line_counts.csv", index=False)

    lines = [
        "收盘亚盘主盘 × 收盘大小球主盘（只用皇冠 Singbet）",
        f"样本 n={len(d)}  比赛数={d.match_id.nunique()}  公司分布={d.bookmaker.value_counts().to_dict()}",
        f"大小球盘口档: {sorted(d.ou_line.unique().tolist())}",
        f"门槛：n≥{MIN_N} 且 胜率≥{HIT_MIN:.0%} 且 ROI≥{ROI_MIN:.0%}",
        "组合边里「上盘+大球」是各下 1 单位、ROI 取平均，不是串关。胜率是两边同时打出。",
        "",
        "=== 大小球主盘分布（前 15）===",
        d["ou_lab"].value_counts().head(15).to_string(),
        "",
        "=== 亚盘主盘分布（前 15）===",
        d["ah_lab"].value_counts().head(15).to_string(),
        "",
        "=== 按大小球盘口，各边 ROI 最高 ===",
        show(by_ou.sort_values("ROI", ascending=False).drop_duplicates("切片"), 20),
        "",
        "=== 按亚盘盘口，各边 ROI 最高 ===",
        show(by_ah.sort_values("ROI", ascending=False).drop_duplicates("切片"), 20),
        "",
        f"=== 球头组合过门槛 {len(best_hits)} 条 ===",
        show(best_hits, 30),
        "",
        "=== 全部组合里 ROI 最高（不论门槛，前 20）===",
        show(best, 20),
        "",
        f"全场主队盘胜率 {d.home_hit.mean():.1%} ROI {d.pnl_home.mean():+.1%}；"
        f"大球胜率 {d.over_hit.mean():.1%} ROI {d.pnl_over.mean():+.1%}；"
        f"上盘胜率 {d.up_hit.mean():.1%} ROI {d.pnl_up.mean():+.1%}",
    ]
    text = "\n".join(lines)
    (OUT / "ah_ou_lines.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
