"""收盘亚盘结果 vs 同赛事同主客近10场、以及初盘→收盘的变盘。

比较基准是「同一条收盘盘口自己的打出率和 ROI」，避免把盘口底子当成信号。
样本：football-data 有初盘 AHh、收盘 AHCh 的联赛，2019-20～2025-26（不含未完赛的 2026-27）。
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .analyze_ah_l3_slices import line_label, pnl
from .analyze_ah_xg import OUT, PROC, RAW, settle_ah_home

COMPLETE = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]

# 五大 + 二级 + 其他有亚盘的联赛。文件不存在就跳过。
DIVS: list[tuple[str, str, str]] = [
    ("E0", "英超", "五大"),
    ("SP1", "西甲", "五大"),
    ("D1", "德甲", "五大"),
    ("I1", "意甲", "五大"),
    ("F1", "法甲", "五大"),
    ("E1", "英冠", "二级"),
    ("SP2", "西乙", "二级"),
    ("D2", "德乙", "二级"),
    ("I2", "意乙", "二级"),
    ("F2", "法乙", "二级"),
    ("N1", "荷甲", "其他"),
    ("B1", "比甲", "其他"),
    ("P1", "葡超", "其他"),
    ("T1", "土超", "其他"),
    ("G1", "希超", "其他"),
    ("SC0", "苏超", "其他"),
    ("E2", "英甲", "其他"),
]


def snap(x: float) -> float:
    return round(float(x) * 4) / 4


def describe(line: float) -> str:
    line = snap(line)
    if abs(line) < 1e-9:
        return "平手"
    side = "主让" if line < 0 else "客让"
    return side + line_label(abs(line))


def load_matches() -> pd.DataFrame:
    frames = []
    for season in COMPLETE:
        for div, name, tier in DIVS:
            path = RAW / f"fd_{div}_{season}.csv"
            if not path.exists():
                continue
            fd = pd.read_csv(path)
            if "AHh" not in fd.columns or "AHCh" not in fd.columns:
                continue
            fd["div"] = div
            fd["league"] = name
            fd["tier"] = tier
            fd["season_path"] = season
            frames.append(fd)
    fd = pd.concat(frames, ignore_index=True)
    fd["date"] = pd.to_datetime(fd["Date"], dayfirst=True, errors="coerce")
    for col in ("AHh", "AHCh", "FTHG", "FTAG", "AvgCAHH", "AvgCAHA", "B365CAHH", "B365CAHA"):
        if col not in fd.columns:
            fd[col] = np.nan
        fd[col] = pd.to_numeric(fd[col], errors="coerce")
    fd = fd.dropna(subset=["date", "AHh", "AHCh", "FTHG", "FTAG", "HomeTeam", "AwayTeam"])
    fd["ah_home_odds"] = fd["AvgCAHH"].fillna(fd["B365CAHH"])
    fd["ah_away_odds"] = fd["AvgCAHA"].fillna(fd["B365CAHA"])
    fd = fd.dropna(subset=["ah_home_odds", "ah_away_odds"])
    fd = fd[(fd["ah_home_odds"] > 1.01) & (fd["ah_away_odds"] > 1.01)].copy()
    fd["open_line"] = fd["AHh"].map(snap)
    fd["close_line"] = fd["AHCh"].map(snap)
    fd["gd"] = fd["FTHG"] - fd["FTAG"]
    fd["home_cover"] = [settle_ah_home(g, ln) for g, ln in zip(fd["gd"], fd["close_line"])]
    fd["home_hit"] = (fd["home_cover"] > 0).astype(float)
    fd["away_cover"] = -fd["home_cover"]
    fd["pnl_home"] = pnl(fd["home_cover"], fd["ah_home_odds"])
    fd["pnl_away"] = pnl(fd["away_cover"], fd["ah_away_odds"])
    fd["abs_line"] = fd["close_line"].abs()
    fav = np.where(fd["close_line"] < 0, "home", np.where(fd["close_line"] > 0, "away", "pickem"))
    pickem = fav == "pickem"
    fav = np.where(pickem & (fd["ah_home_odds"] < fd["ah_away_odds"]), "home", fav)
    fav = np.where(pickem & (fd["ah_away_odds"] < fd["ah_home_odds"]), "away", fav)
    fd["favorite"] = fav
    fd = fd[fd["favorite"] != "pickem"].copy()
    fd["favorite_cover"] = np.where(fd["favorite"] == "home", fd["home_cover"], fd["away_cover"])
    fd["up_hit"] = (fd["favorite_cover"] > 0).astype(float)
    fd["fav_odds"] = np.where(fd["favorite"] == "home", fd["ah_home_odds"], fd["ah_away_odds"])
    fd["dog_odds"] = np.where(fd["favorite"] == "home", fd["ah_away_odds"], fd["ah_home_odds"])
    fd["pnl_up"] = pnl(fd["favorite_cover"], fd["fav_odds"])
    fd["pnl_dn"] = pnl(-fd["favorite_cover"], fd["dog_odds"])
    # 正数 = 收盘比初盘更看好主队（主队多让，或客队少让）
    fd["home_steps"] = ((fd["open_line"] - fd["close_line"]) / 0.25).round().astype(int)
    fd["open_lab"] = fd["open_line"].map(describe)
    fd["close_lab"] = fd["close_line"].map(describe)
    fd["transition"] = fd["open_lab"] + "→" + fd["close_lab"]
    return fd.sort_values(["div", "date", "HomeTeam"]).reset_index(drop=True)


def attach_venue10(df: pd.DataFrame) -> pd.DataFrame:
    """同分区、赛前、主队近10个主场 vs 客队近10个客场。跨赛季，不跨联赛。"""
    home_hist: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    away_hist: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    h_gd = np.full(len(df), np.nan)
    a_gd = np.full(len(df), np.nan)
    h_ppg = np.full(len(df), np.nan)
    a_ppg = np.full(len(df), np.nan)
    h_cov = np.full(len(df), np.nan)
    a_cov = np.full(len(df), np.nan)

    for i, r in enumerate(df.itertuples(index=False)):
        hk = (r.div, r.HomeTeam)
        ak = (r.div, r.AwayTeam)
        hh = home_hist[hk]
        aa = away_hist[ak]
        if len(hh) >= 10 and len(aa) >= 10:
            h10, a10 = hh[-10:], aa[-10:]
            h_gd[i] = float(np.mean([x[0] for x in h10]))
            a_gd[i] = float(np.mean([x[0] for x in a10]))
            h_ppg[i] = float(np.mean([x[1] for x in h10]))
            a_ppg[i] = float(np.mean([x[1] for x in a10]))
            h_cov[i] = float(np.mean([x[2] for x in h10]))
            a_cov[i] = float(np.mean([x[2] for x in a10]))
        hpts = 3.0 if r.FTHG > r.FTAG else (1.0 if r.FTHG == r.FTAG else 0.0)
        apts = 3.0 if r.FTAG > r.FTHG else (1.0 if r.FTHG == r.FTAG else 0.0)
        hh.append((float(r.gd), hpts, float(r.home_cover)))
        aa.append((float(-r.gd), apts, float(r.away_cover)))

    out = df.copy()
    out["h10_gd"] = h_gd
    out["a10_gd"] = a_gd
    out["h10_ppg"] = h_ppg
    out["a10_ppg"] = a_ppg
    out["h10_cover"] = h_cov
    out["a10_cover"] = a_cov
    # 主队近10主场场均净胜 − 客队近10客场场均净胜 − 收盘主队让球
    # 正数：同主客赛果比这条收盘盘更看好主队
    out["form_edge"] = out["h10_gd"] - out["a10_gd"] - (-out["close_line"])
    out["ppg_gap"] = out["h10_ppg"] - out["a10_ppg"]
    out["cover_gap"] = out["h10_cover"] - out["a10_cover"]
    return out


def add_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """每场减去「同一条收盘盘口」的均值。超额>0 表示比这条盘本身更好。"""
    g = df.groupby("close_line", sort=False)
    out = df.copy()
    for src, dst in (
        ("home_hit", "base_home_hit"),
        ("home_cover", "base_home_cover"),
        ("pnl_home", "base_pnl_home"),
        ("pnl_away", "base_pnl_away"),
        ("up_hit", "base_up_hit"),
        ("pnl_up", "base_pnl_up"),
        ("pnl_dn", "base_pnl_dn"),
    ):
        out[dst] = g[src].transform("mean")
    out["ex_home_hit"] = out["home_hit"] - out["base_home_hit"]
    out["ex_home_cover"] = out["home_cover"] - out["base_home_cover"]
    out["ex_pnl_home"] = out["pnl_home"] - out["base_pnl_home"]
    out["ex_pnl_away"] = out["pnl_away"] - out["base_pnl_away"]
    out["ex_up_hit"] = out["up_hit"] - out["base_up_hit"]
    out["ex_pnl_up"] = out["pnl_up"] - out["base_pnl_up"]
    out["ex_pnl_dn"] = out["pnl_dn"] - out["base_pnl_dn"]
    return out


def _pack(name: str, sub: pd.DataFrame, min_n: int) -> dict | None:
    n = len(sub)
    if n < min_n:
        return None
    def mean_se(col: str) -> tuple[float, float]:
        x = sub[col].to_numpy(dtype=float)
        m = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        return m, se

    rows = {"切片": name, "n": n}
    for col, key in (
        ("home_hit", "主队打出"),
        ("ex_home_hit", "主队打出超额"),
        ("home_cover", "主队cover"),
        ("ex_home_cover", "主队cover超额"),
        ("pnl_home", "主队ROI"),
        ("ex_pnl_home", "主队ROI超额"),
        ("pnl_away", "客队ROI"),
        ("ex_pnl_away", "客队ROI超额"),
        ("up_hit", "上盘打出"),
        ("ex_up_hit", "上盘打出超额"),
        ("pnl_up", "上盘ROI"),
        ("ex_pnl_up", "上盘ROI超额"),
        ("pnl_dn", "下盘ROI"),
        ("ex_pnl_dn", "下盘ROI超额"),
    ):
        m, se = mean_se(col)
        rows[key] = m
        if col.startswith("ex_"):
            rows[key + "se"] = se
    # 赛季符号是否一致（主队 ROI 超额）
    by = sub.groupby("season_path")["ex_pnl_home"].mean()
    rows["赛季数"] = int(by.notna().sum())
    rows["主队ROI超额为正的赛季"] = int((by > 0).sum())
    return rows


def move_bucket(steps: int) -> str:
    if steps >= 2:
        return "主升≥2档"
    if steps == 1:
        return "主升1档"
    if steps == 0:
        return "盘口不变"
    if steps == -1:
        return "主降1档"
    return "主降≥2档"


def form_bucket(edge: float) -> str:
    if edge <= -0.75:
        return "同主客比盘口弱≥0.75球"
    if edge <= -0.25:
        return "同主客比盘口弱0.25~0.75"
    if edge < 0.25:
        return "同主客接近盘口"
    if edge < 0.75:
        return "同主客比盘口强0.25~0.75"
    return "同主客比盘口强≥0.75球"


def corr_row(name: str, x: pd.Series, y: pd.Series) -> dict:
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(m)
    if n < 50 or m["x"].std() == 0 or m["y"].std() == 0:
        return {"关系": name, "n": n, "相关": np.nan, "p": np.nan}
    r = float(m["x"].corr(m["y"]))
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # 双侧，正态近似足够（n 很大）
    from math import erfc

    p = float(erfc(abs(t) / np.sqrt(2)))
    return {"关系": name, "n": n, "相关": r, "p": p}


def side_vs_line(df: pd.DataFrame) -> pd.DataFrame:
    """旧结论核对：某盘口×主/客上盘，相对这条盘口（不区分主客）的超额。"""
    rows = []
    for line, g in df.groupby(df["abs_line"]):
        if len(g) < 200:
            continue
        base_up = float((g["favorite_cover"] > 0).mean())
        base_dn = float((g["favorite_cover"] < 0).mean())
        base_roi_up = float(g["pnl_up"].mean())
        base_roi_dn = float(g["pnl_dn"].mean())
        lab = line_label(line)
        for side, name in (("home", "主队上盘"), ("away", "客队上盘")):
            sub = g[g["favorite"] == side]
            if len(sub) < 80:
                continue
            up = float((sub["favorite_cover"] > 0).mean())
            dn = float((sub["favorite_cover"] < 0).mean())
            roi_up = float(sub["pnl_up"].mean())
            roi_dn = float(sub["pnl_dn"].mean())
            rows.append(
                {
                    "盘口": lab,
                    "球头": line,
                    "上盘方": name,
                    "n": len(sub),
                    "上盘打出": up,
                    "下盘打出": dn,
                    "该盘口上盘打出": base_up,
                    "上盘打出超额": up - base_up,
                    "下盘打出超额": dn - base_dn,
                    "上盘ROI": roi_up,
                    "下盘ROI": roi_dn,
                    "该盘口上盘ROI": base_roi_up,
                    "该盘口下盘ROI": base_roi_dn,
                    "上盘ROI超额": roi_up - base_roi_up,
                    "下盘ROI超额": roi_dn - base_roi_dn,
                }
            )
    return pd.DataFrame(rows).sort_values(["球头", "上盘方"])


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_matches()
    featured = add_baseline(attach_venue10(raw))

    featured["变盘"] = featured["home_steps"].map(move_bucket)
    has_form = featured["form_edge"].notna()
    featured.loc[has_form, "同主客档"] = featured.loc[has_form, "form_edge"].map(form_bucket)

    side = side_vs_line(featured)
    side.to_csv(PROC / "ah_side_vs_line.csv", index=False)

    move_rows = []
    for name, sub in featured.groupby("变盘"):
        rec = _pack(f"变盘={name}", sub, min_n=200)
        if rec:
            move_rows.append(rec)
    move_df = pd.DataFrame(move_rows)

    # 变盘 × 档位，只留样本够的；主队打出超额相对「同一条收盘」
    trans_rows = []
    for name, sub in featured.groupby("transition"):
        rec = _pack(name, sub, min_n=250)
        if rec:
            trans_rows.append(rec)
    trans_df = pd.DataFrame(trans_rows)
    if len(trans_df):
        trans_df = trans_df.reindex(
            trans_df["主队ROI超额"].abs().sort_values(ascending=False).index
        )

    form_rows = []
    form = featured[has_form]
    for name, sub in form.groupby("同主客档"):
        rec = _pack(f"赛果={name}", sub, min_n=200)
        if rec:
            form_rows.append(rec)
    # 近10场同主客「自己的亚盘」谁在赢盘
    form = form.copy()
    form["盘路档"] = pd.qcut(form["cover_gap"], 5, duplicates="drop")
    for name, sub in form.groupby("盘路档", observed=True):
        rec = _pack(f"近10场同主客盘路差 {name}", sub, min_n=200)
        if rec:
            form_rows.append(rec)
    form_df = pd.DataFrame(form_rows)

    # 一级/二级/其他 是否同号
    split_rows = []
    for tier, g in featured.groupby("tier"):
        rec = _pack(f"{tier}·全部", g, min_n=200)
        if rec:
            split_rows.append(rec)
        for name, sub in g.groupby("变盘"):
            rec = _pack(f"{tier}·变盘={name}", sub, min_n=150)
            if rec:
                split_rows.append(rec)
    for tier, g in form.groupby("tier"):
        for name, sub in g.groupby("同主客档"):
            rec = _pack(f"{tier}·赛果={name}", sub, min_n=120)
            if rec:
                split_rows.append(rec)
    split_df = pd.DataFrame(split_rows)

    corrs = [
        corr_row("变盘档数 → 主队cover超额", featured["home_steps"], featured["ex_home_cover"]),
        corr_row("变盘档数 → 主队ROI超额", featured["home_steps"], featured["ex_pnl_home"]),
        corr_row("同主客净胜差相对盘口 → 主队cover超额", form["form_edge"], form["ex_home_cover"]),
        corr_row("同主客净胜差相对盘口 → 主队ROI超额", form["form_edge"], form["ex_pnl_home"]),
        corr_row("同主客积分差 → 主队cover超额", form["ppg_gap"], form["ex_home_cover"]),
        corr_row("近10场同主客盘路差 → 主队cover超额", form["cover_gap"], form["ex_home_cover"]),
        corr_row("近10场同主客盘路差 → 主队ROI超额", form["cover_gap"], form["ex_pnl_home"]),
    ]
    for tier, g in featured.groupby("tier"):
        corrs.append(corr_row(f"{tier} 变盘档数 → 主队cover超额", g["home_steps"], g["ex_home_cover"]))
    for tier, g in form.groupby("tier"):
        corrs.append(
            corr_row(f"{tier} 同主客相对盘口 → 主队cover超额", g["form_edge"], g["ex_home_cover"])
        )
    corr_df = pd.DataFrame(corrs)

    move_df.to_csv(PROC / "ah_line_move.csv", index=False)
    trans_df.to_csv(PROC / "ah_line_transitions.csv", index=False)
    form_df.to_csv(PROC / "ah_venue10.csv", index=False)
    split_df.to_csv(PROC / "ah_form_move_by_tier.csv", index=False)
    corr_df.to_csv(PROC / "ah_form_move_corr.csv", index=False)

    leagues = featured.groupby(["tier", "league"]).size().reset_index(name="n")
    summary = {
        "n": int(len(featured)),
        "n_form": int(has_form.sum()),
        "seasons": COMPLETE,
        "leagues": leagues.to_dict(orient="records"),
        "home_hit": float(featured["home_hit"].mean()),
        "pnl_home": float(featured["pnl_home"].mean()),
        "pnl_away": float(featured["pnl_away"].mean()),
        "unchanged": float((featured["home_steps"] == 0).mean()),
    }
    (PROC / "ah_form_move_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def show(df: pd.DataFrame, cols: list[str]) -> str:
        if df is None or df.empty:
            return "(空)"
        return df[cols].round(4).to_string(index=False)

    lines = [
        f"样本 n={summary['n']}（有初盘+收盘+收盘水位，2019-20～2025-26）",
        f"其中双方都满近10场同赛事同主客 n={summary['n_form']}",
        f"主队盘打出 {summary['home_hit']:.1%}  主队ROI {summary['pnl_home']:+.2%}  客队ROI {summary['pnl_away']:+.2%}",
        f"收盘与初盘相同占 {summary['unchanged']:.1%}",
        "超额 = 该切片 − 同一条收盘盘口（带方向，如「主让平半」）的均值。",
        "正的主队打出超额 = 买收盘主队，比买这条盘口的平均更容易打出。",
        "",
        "=== 联赛样本量 ===",
        leagues.to_string(index=False),
        "",
        "=== 相关（已经扣掉收盘盘口）===",
        corr_df.round(4).to_string(index=False),
        "",
        "=== 旧切片：主/客上盘相对「这条盘口不区分主客」===",
        show(
            side,
            [
                "盘口",
                "上盘方",
                "n",
                "上盘打出",
                "该盘口上盘打出",
                "上盘打出超额",
                "下盘打出",
                "下盘打出超额",
                "上盘ROI",
                "下盘ROI",
                "下盘ROI超额",
            ],
        ),
        "",
        "=== 变盘档（相对同一条收盘）===",
        show(
            move_df,
            ["切片", "n", "主队打出", "主队打出超额", "主队ROI", "主队ROI超额", "客队ROI超额", "主队ROI超额为正的赛季", "赛季数"],
        ),
        "",
        "=== 同主客近10场（相对同一条收盘）===",
        show(
            form_df,
            ["切片", "n", "主队打出", "主队打出超额", "主队ROI", "主队ROI超额", "主队ROI超额为正的赛季", "赛季数"],
        ),
        "",
        "=== 初盘→收盘，|主队ROI超额|最大的变盘（n≥250）===",
        show(
            trans_df.head(25),
            ["切片", "n", "主队打出", "主队打出超额", "主队ROI", "主队ROI超额", "客队ROI超额", "主队ROI超额为正的赛季", "赛季数"],
        ),
    ]
    text = "\n".join(lines)
    (OUT / "ah_form_move.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
