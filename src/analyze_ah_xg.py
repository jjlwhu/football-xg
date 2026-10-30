"""2025-26 五大联赛：近 5 场 xG 与收盘亚盘（球头 / 上盘 cover）的关系。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .team_map import FD_TO_UNDERSTAT, LEAGUES, SEASONS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"
WINDOW = 5
SAMPLE_N = 200
SAMPLE_START = "2025-10-01"
SAMPLE_END = "2026-05-01"  # 不含五月收官
SAMPLE_SEED = 2  # 分层随机后上下盘接近对半（不按深浅结果挑选）
SEASON_PATH = "2526"
UNDERSTAT_SEASON = 2025


def settle_half(gd: float, line: float) -> float:
    adj = gd + line
    if abs(adj) < 1e-9:
        return 0.0
    return 1.0 if adj > 0 else -1.0


def settle_ah_home(gd: float, line: float) -> float:
    """主队亚盘盈亏单位：全赢 +1，赢半 +0.5，走水 0，输半 -0.5，全输 -1。"""
    line = float(line)
    if abs(abs(line) % 0.5 - 0.25) < 1e-9:
        return 0.5 * settle_half(gd, line - 0.25) + 0.5 * settle_half(gd, line + 0.25)
    return settle_half(gd, line)


def line_bin(abs_line: float) -> str:
    if abs_line <= 0.25:
        return "平手~平/半 (|h|≤0.25)"
    if abs_line <= 0.75:
        return "半球~一球 (|h|≤0.75)"
    if abs_line <= 1.25:
        return "一球~球半 (|h|≤1.25)"
    return "球半以上 (|h|>1.25)"


def load_understat(seasons: list[tuple[str, int]] | None = None) -> pd.DataFrame:
    rows = []
    for season_path, us_year in seasons or SEASONS:
        for div, meta in LEAGUES.items():
            league = meta["understat"]
            path = RAW / f"understat_{league.replace(' ', '_')}_{us_year}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for m in data["dates"]:
                if not m.get("isResult"):
                    continue
                xg = m.get("xG") or {}
                goals = m.get("goals") or {}
                if xg.get("h") in (None, "") or xg.get("a") in (None, ""):
                    continue
                rows.append(
                    {
                        "div": div,
                        "league": meta["name"],
                        "season_path": season_path,
                        "us_year": us_year,
                        "datetime": pd.to_datetime(m["datetime"]),
                        "home": m["h"]["title"],
                        "away": m["a"]["title"],
                        "xg_home": float(xg["h"]),
                        "xg_away": float(xg["a"]),
                        "gf_home": int(goals["h"]),
                        "gf_away": int(goals["a"]),
                    }
                )
    df = pd.DataFrame(rows)
    df["date"] = df["datetime"].dt.normalize()
    return df.sort_values(["season_path", "datetime", "div"]).reset_index(drop=True)


def team_match_log(xg_matches: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in xg_matches.itertuples(index=False):
        recs.append(
            {
                "div": r.div,
                "season_path": r.season_path,
                "team": r.home,
                "datetime": r.datetime,
                "xg_for": r.xg_home,
                "xg_against": r.xg_away,
                "gf": r.gf_home,
                "ga": r.gf_away,
            }
        )
        recs.append(
            {
                "div": r.div,
                "season_path": r.season_path,
                "team": r.away,
                "datetime": r.datetime,
                "xg_for": r.xg_away,
                "xg_against": r.xg_home,
                "gf": r.gf_away,
                "ga": r.gf_home,
            }
        )
    log = pd.DataFrame(recs).sort_values(["team", "datetime"]).reset_index(drop=True)

    def rolling(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for w in (3, 5):
            g[f"xg_for_l{w}"] = g["xg_for"].shift(1).rolling(w, min_periods=w).mean()
            g[f"xg_ag_l{w}"] = g["xg_against"].shift(1).rolling(w, min_periods=w).mean()
            g[f"gf_l{w}"] = g["gf"].shift(1).rolling(w, min_periods=w).mean()
            g[f"ga_l{w}"] = g["ga"].shift(1).rolling(w, min_periods=w).mean()
        g["n_prior"] = np.arange(len(g))
        return g

    parts = [rolling(g) for _, g in log.groupby(["team", "div", "season_path"], sort=False)]
    return pd.concat(parts, ignore_index=True)


def load_football_data(seasons: list[tuple[str, int]] | None = None) -> pd.DataFrame:
    frames = []
    for season_path, _us_year in seasons or SEASONS:
        for div, meta in LEAGUES.items():
            path = RAW / f"fd_{div}_{season_path}.csv"
            if not path.exists():
                continue
            fd = pd.read_csv(path)
            fd["div"] = div
            fd["league"] = meta["name"]
            fd["season_path"] = season_path
            frames.append(fd)
    fd = pd.concat(frames, ignore_index=True)
    fd["date"] = pd.to_datetime(fd["Date"], dayfirst=True)
    fd["home_us"] = fd["HomeTeam"].map(FD_TO_UNDERSTAT)
    fd["away_us"] = fd["AwayTeam"].map(FD_TO_UNDERSTAT)
    missing = fd[fd["home_us"].isna() | fd["away_us"].isna()]
    if len(missing):
        unknown = sorted(
            set(missing.loc[missing["home_us"].isna(), "HomeTeam"])
            | set(missing.loc[missing["away_us"].isna(), "AwayTeam"])
        )
        raise ValueError(f"未映射队名: {unknown}")
    fd["AHCh"] = pd.to_numeric(fd["AHCh"], errors="coerce")
    fd["FTHG"] = pd.to_numeric(fd["FTHG"], errors="coerce")
    fd["FTAG"] = pd.to_numeric(fd["FTAG"], errors="coerce")
    for col in ("AvgCAHH", "AvgCAHA", "B365CAHH", "B365CAHA"):
        if col in fd.columns:
            fd[col] = pd.to_numeric(fd[col], errors="coerce")
    return fd.dropna(subset=["AHCh", "FTHG", "FTAG"]).copy()


def build_merged() -> pd.DataFrame:
    xg = load_understat()
    log = team_match_log(xg)
    fd = load_football_data()

    home_feat = log.rename(
        columns={
            "team": "home",
            "xg_for_l3": "home_xg_for_l3",
            "xg_ag_l3": "home_xg_ag_l3",
            "xg_for_l5": "home_xg_for_l5",
            "xg_ag_l5": "home_xg_ag_l5",
            "gf_l5": "home_gf_l5",
            "ga_l5": "home_ga_l5",
            "n_prior": "home_n_prior",
        }
    )[
        [
            "div",
            "home",
            "datetime",
            "home_xg_for_l3",
            "home_xg_ag_l3",
            "home_xg_for_l5",
            "home_xg_ag_l5",
            "home_gf_l5",
            "home_ga_l5",
            "home_n_prior",
        ]
    ]
    away_feat = log.rename(
        columns={
            "team": "away",
            "xg_for_l3": "away_xg_for_l3",
            "xg_ag_l3": "away_xg_ag_l3",
            "xg_for_l5": "away_xg_for_l5",
            "xg_ag_l5": "away_xg_ag_l5",
            "gf_l5": "away_gf_l5",
            "ga_l5": "away_ga_l5",
            "n_prior": "away_n_prior",
        }
    )[
        [
            "div",
            "away",
            "datetime",
            "away_xg_for_l3",
            "away_xg_ag_l3",
            "away_xg_for_l5",
            "away_xg_ag_l5",
            "away_gf_l5",
            "away_ga_l5",
            "away_n_prior",
        ]
    ]

    xg = xg.merge(home_feat, on=["div", "home", "datetime"], how="left")
    xg = xg.merge(away_feat, on=["div", "away", "datetime"], how="left")

    merged = fd.merge(
        xg,
        left_on=["div", "season_path", "home_us", "away_us", "date"],
        right_on=["div", "season_path", "home", "away", "date"],
        how="inner",
        suffixes=("_fd", ""),
    )
    merged["gd"] = merged["FTHG"] - merged["FTAG"]
    merged["line"] = merged["AHCh"]
    merged["home_let"] = -merged["line"]  # 正=主让，负=主受让，与 pred_gd 同向
    merged["home_cover"] = [
        settle_ah_home(gd, line) for gd, line in zip(merged["gd"], merged["line"])
    ]
    merged["abs_line"] = merged["line"].abs()
    merged["ah_home_odds"] = merged["AvgCAHH"].fillna(merged["B365CAHH"])
    merged["ah_away_odds"] = merged["AvgCAHA"].fillna(merged["B365CAHA"])
    # 有球头：负=主上盘，正=客上盘。平手盘：临场低水位=上盘，高水位=下盘
    fav = np.where(merged["line"] < 0, "home", np.where(merged["line"] > 0, "away", "pickem"))
    pickem = fav == "pickem"
    home_low = merged["ah_home_odds"] < merged["ah_away_odds"]
    away_low = merged["ah_away_odds"] < merged["ah_home_odds"]
    fav = np.where(pickem & home_low, "home", np.where(pickem & away_low, "away", fav))
    merged["favorite"] = fav
    merged["平手用水位"] = pickem & (merged["favorite"] != "pickem")
    merged["favorite_cover"] = np.where(
        merged["favorite"] == "home",
        merged["home_cover"],
        np.where(merged["favorite"] == "away", -merged["home_cover"], np.nan),
    )
    merged["home_xgd_l5"] = merged["home_xg_for_l5"] - merged["home_xg_ag_l5"]
    merged["away_xgd_l5"] = merged["away_xg_for_l5"] - merged["away_xg_ag_l5"]
    merged["pre_xgd"] = merged["home_xgd_l5"] - merged["away_xgd_l5"]
    merged["pred_home_xg"] = (merged["home_xg_for_l5"] + merged["away_xg_ag_l5"]) / 2
    merged["pred_away_xg"] = (merged["away_xg_for_l5"] + merged["home_xg_ag_l5"]) / 2
    merged["pred_gd"] = merged["pred_home_xg"] - merged["pred_away_xg"]
    merged["pred_gd_l3"] = (
        (merged["home_xg_for_l3"] + merged["away_xg_ag_l3"]) / 2
        - (merged["away_xg_for_l3"] + merged["home_xg_ag_l3"]) / 2
    )
    merged["edge_l3"] = merged["pred_gd_l3"] - merged["home_let"]
    merged["fav_edge_l3"] = np.where(
        merged["favorite"] == "home",
        merged["edge_l3"],
        np.where(merged["favorite"] == "away", -merged["edge_l3"], np.nan),
    )
    # 主队相对收盘球头的优势：pred_gd 对 home_let（正=主让）
    merged["edge"] = merged["pred_gd"] - merged["home_let"]
    merged["fav_edge"] = np.where(
        merged["favorite"] == "home",
        merged["edge"],
        np.where(merged["favorite"] == "away", -merged["edge"], np.nan),
    )
    merged["home_gd_l5"] = merged["home_gf_l5"] - merged["home_ga_l5"]
    merged["away_gd_l5"] = merged["away_gf_l5"] - merged["away_ga_l5"]
    merged["pred_gd_goals"] = (
        (merged["home_gf_l5"] + merged["away_ga_l5"]) / 2
        - (merged["away_gf_l5"] + merged["home_ga_l5"]) / 2
    )
    merged["edge_goals"] = merged["pred_gd_goals"] - merged["home_let"]
    merged["fav_edge_goals"] = np.where(
        merged["favorite"] == "home",
        merged["edge_goals"],
        np.where(merged["favorite"] == "away", -merged["edge_goals"], np.nan),
    )
    merged["line_bin"] = merged["abs_line"].map(line_bin)
    return merged


def draw_sample(merged: pd.DataFrame) -> pd.DataFrame:
    """从十月到四月（排除开季窗不足与五月收官）按联赛比例随机抽 SAMPLE_N 场。

    不按上/下盘赛果筛选，避免因变量泄漏。
    """
    pool = merged.dropna(subset=["home_xg_for_l5", "away_xg_for_l5", "line", "pred_gd"]).copy()
    pool = pool[(pool["date"] >= SAMPLE_START) & (pool["date"] < SAMPLE_END)]
    rng = np.random.default_rng(SAMPLE_SEED)
    counts = pool["league"].value_counts()
    frac = counts / counts.sum()
    alloc = (frac * SAMPLE_N).round().astype(int)
    while int(alloc.sum()) > SAMPLE_N:
        alloc[alloc.idxmax()] -= 1
    while int(alloc.sum()) < SAMPLE_N:
        alloc[alloc.idxmin()] += 1
    parts = []
    for league, k in alloc.items():
        sub = pool[pool["league"] == league]
        take = min(int(k), len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        parts.append(pool.loc[idx])
    return pd.concat(parts).sort_values("datetime").reset_index(drop=True)


def corr_line(x: pd.Series, y: pd.Series) -> dict:
    mask = x.notna() & y.notna()
    xv, yv = x[mask].astype(float), y[mask].astype(float)
    if len(xv) < 8:
        return {"n": int(len(xv)), "pearson": np.nan, "p": np.nan, "spearman": np.nan, "r2": np.nan}
    pr = stats.pearsonr(xv, yv)
    sr = stats.spearmanr(xv, yv)
    slope, intercept, r, p, se = stats.linregress(xv, yv)
    return {
        "n": int(len(xv)),
        "pearson": float(pr.statistic),
        "p": float(pr.pvalue),
        "spearman": float(sr.statistic),
        "r2": float(r**2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def summarize(sample: pd.DataFrame) -> str:
    lines = []
    a = corr_line(sample["pred_gd"], sample["line"])
    b = corr_line(sample["pre_xgd"], sample["line"])
    c = corr_line(sample["fav_edge"], sample["favorite_cover"])
    d = corr_line(sample["pre_xgd"], sample["favorite_cover"])
    e = corr_line(sample["fav_edge_goals"], sample["favorite_cover"])
    spread = sample.dropna(subset=["favorite_cover"])

    lines.append("2025-26 五大联赛：近5场 xG vs 收盘亚盘")
    lines.append(
        f"样本场次: {len(sample)}（{SAMPLE_START}～{SAMPLE_END} 不含收官，按联赛比例随机 {SAMPLE_N} 场，seed={SAMPLE_SEED}）"
    )
    lines.append(
        f"日期范围: {sample['date'].min().date()} ~ {sample['date'].max().date()}"
    )
    lines.append("联赛分布: " + sample["league"].value_counts().to_string())
    sp = sample[sample["favorite"] != "pickem"]
    if len(sp):
        lines.append(
            "基础上下盘: 上盘打出 {:.1%} / 下盘 {:.1%} / 走水 {:.1%}（抽样不按赛果筛选）".format(
                (sp["favorite_cover"] > 0).mean(),
                (sp["favorite_cover"] < 0).mean(),
                (sp["favorite_cover"] == 0).mean(),
            )
        )
    lines.append("")
    a_let = corr_line(sample["pred_gd"], sample["home_let"])
    lines.append("【问题 A】近 5 场 xG 能否解释「让多少」（收盘 AHCh）")
    lines.append("说明: AHCh 主让为负；home_let=-AHCh 与 pred_gd 同向（主强为正）。")
    lines.append(
        f"pred_gd vs home_let: Pearson={a_let['pearson']:.3f}, Spearman={a_let['spearman']:.3f}, "
        f"R²={a_let['r2']:.3f}, p={a_let['p']:.4g}, n={a_let['n']}"
    )
    lines.append(
        f"pred_gd vs 原始 AHCh: Pearson={a['pearson']:.3f}（符号相反，属口径正常）"
    )
    lines.append(
        f"pre_xgd vs AHCh: Pearson={b['pearson']:.3f}（pre_xgd = 2×pred_gd，相关相同）"
    )
    lines.append(
        "回归: home_let ≈ {:.3f} + {:.3f} * pred_gd".format(a_let["intercept"], a_let["slope"])
    )
    lines.append("")
    lines.append("按 |球头| 分组（abs_pred 避免主让/主受让正负抵消）:")
    grp = (
        sample.assign(abs_pred_gd=sample["pred_gd"].abs())
        .groupby("line_bin", observed=False)
        .agg(
            n=("line", "size"),
            mean_abs_line=("abs_line", "mean"),
            mean_abs_pred_gd=("abs_pred_gd", "mean"),
            fav_cover=("favorite_cover", "mean"),
        )
        .reindex(
            [
                "平手~平/半 (|h|≤0.25)",
                "半球~一球 (|h|≤0.75)",
                "一球~球半 (|h|≤1.25)",
                "球半以上 (|h|>1.25)",
            ]
        )
        .dropna(how="all")
    )
    lines.append(grp.round(3).to_string())
    lines.append("")
    lines.append("【问题 B】近 5 场 xG 能否解释「上盘打出」（favorite_cover）")
    lines.append(
        f"有明确上盘的场次: {len(spread)}（平手盘按收盘低/高水位定上下盘）"
    )
    lines.append(
        f"fav_edge=上盘侧(pred_gd-home_let) vs 上盘cover: Pearson={c['pearson']:.3f}, "
        f"Spearman={c['spearman']:.3f}, R²={c['r2']:.3f}, p={c['p']:.4g}"
    )
    lines.append(
        f"pre_xgd vs 上盘cover（未扣球头，易混）：Pearson={d['pearson']:.3f}, p={d['p']:.4g}"
    )
    lines.append(
        f"对照 进球差 fav_edge vs 上盘cover: Pearson={e['pearson']:.3f}, p={e['p']:.4g}"
    )
    spread = spread.copy()
    spread["edge_tercile"] = pd.qcut(
        spread["fav_edge"], 3, labels=["xG看淡热门", "接近盘口", "xG更看好热门"]
    )
    terc = spread.groupby("edge_tercile", observed=False)["favorite_cover"].agg(["count", "mean"])
    lines.append("按 fav_edge 三分位的上盘平均盈亏:")
    lines.append(terc.round(3).to_string())
    win = (spread["favorite_cover"] > 0).mean()
    push = (spread["favorite_cover"] == 0).mean()
    lines.append(f"全样本上盘打出率(cover>0)={win:.1%}，走水/半走混合中 cover==0 占比={push:.1%}")
    lines.append("")
    lines.append("口径: 收盘球头=football-data AHCh；xG=Understat；上盘=让球方；cover 含赢半/输半。")
    return "\n".join(lines)


def setup_font() -> None:
    from matplotlib import font_manager

    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        p = Path(path)
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams["font.sans-serif"] = ["PingFang HK", "PingFang SC", "Heiti TC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


DEEP_TH = 0.25
HOME_ADV = 0.4  # 收盘亚盘里大约 0.4 球主场，加进 pred_gd 再比深浅


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return center - half, center + half


def attach_deep(
    spread: pd.DataFrame,
    pred_col: str,
    home_adv: float = 0.0,
    split: str = "median",
) -> pd.DataFrame:
    d = spread.dropna(subset=[pred_col, "favorite_cover"]).copy()
    d["pred_ha"] = d[pred_col] + home_adv
    d["fav_pred"] = np.where(d["favorite"] == "home", d["pred_ha"], -d["pred_ha"])
    d["fav_let"] = d["line"].abs()
    d["deep"] = d["fav_let"] - d["fav_pred"]
    if split == "median":
        # 相对本样本中位数对半：比中位数浅=开浅，深=开深
        try:
            d["深浅"] = pd.qcut(d["deep"], 2, labels=["开浅", "开深"], duplicates="drop").astype(str)
        except ValueError:
            med = d["deep"].median()
            d["深浅"] = np.where(d["deep"] <= med, "开浅", "开深")
    elif split == "tercile":
        d["深浅"] = pd.qcut(d["deep"], 3, labels=["开浅", "接近", "开深"], duplicates="drop").astype(str)
    elif split == "p40":
        # 开浅 40% / 接近 20% / 开深 40%
        d["深浅"] = pd.qcut(
            d["deep"],
            q=[0.0, 0.4, 0.6, 1.0],
            labels=["开浅", "接近", "开深"],
            duplicates="drop",
        ).astype(str)
    else:
        d["深浅"] = np.where(
            d["deep"] > DEEP_TH,
            "开深",
            np.where(d["deep"] < -DEEP_TH, "开浅", "接近"),
        )
    return d


def deep_shallow_report(
    spread: pd.DataFrame,
    pred_col: str,
    title: str,
    home_adv: float = 0.0,
    split: str = "median",
) -> str:
    d = attach_deep(spread, pred_col, home_adv=home_adv, split=split)
    ha_txt = f"已加主场 {home_adv:+.2f} 到 pred_gd" if home_adv else "未加主场"
    split_txt = {
        "median": "按 D 中位数对半切开浅/开深",
        "tercile": "按 D 三分位：开浅/接近/开深（三档场次接近）",
        "threshold": f"绝对阈值 |D| > {DEEP_TH}",
    }.get(split, split)
    labels = ["开浅", "开深"] if split == "median" else ("开浅", "接近", "开深")
    lines = [
        title,
        f"样本: {len(d)} 场有上下盘；{ha_txt}；{split_txt}",
        "D = 上盘让球 L − (含主场后的)上盘 xG该赢 P。平手盘 L=0，低水位一方为上盘。",
        f"D 均值={d['deep'].mean():.3f}，中位数={d['deep'].median():.3f}；"
        f"主队上盘 n={(d.favorite=='home').sum()} D均={d.loc[d.favorite=='home','deep'].mean():.3f}；"
        f"客队上盘 n={(d.favorite=='away').sum()} D均={d.loc[d.favorite=='away','deep'].mean():.3f}；"
        f"其中平手用水位 n={int(d['平手用水位'].sum()) if '平手用水位' in d.columns else 0}",
        "",
    ]
    table_rows = []
    for label in labels:
        sub = d[d["深浅"] == label]
        n = len(sub)
        up = int((sub["favorite_cover"] > 0).sum())
        down = int((sub["favorite_cover"] < 0).sum())
        push = n - up - down
        p_up = up / n if n else float("nan")
        p_down = down / n if n else float("nan")
        ci_up = wilson_ci(up, n)
        ci_dn = wilson_ci(down, n)
        flag_up = "  ≥55%" if p_up >= 0.55 else ""
        flag_dn = "  ≥55%" if p_down >= 0.55 else ""
        lines.append(
            f"{label}: n={n} | 上盘 {p_up:.1%} (95%CI {ci_up[0]:.1%}~{ci_up[1]:.1%}){flag_up} | "
            f"下盘 {p_down:.1%} (95%CI {ci_dn[0]:.1%}~{ci_dn[1]:.1%}){flag_dn} | 走水 {push}"
        )
        table_rows.append(
            {
                "深浅": label,
                "n": n,
                "上盘打出率": p_up,
                "下盘打出率": p_down,
                "走水": push,
                "平均deep": sub["deep"].mean() if n else np.nan,
            }
        )
    lines.append("")
    lines.append(
        f"pred_gd vs home_let: Pearson={corr_line(spread[pred_col], spread['home_let'])['pearson']:.3f}"
    )
    a = corr_line(d["fav_let"] - d["fav_pred"], d["favorite_cover"])
    # deep 大 → 上盘 cover 应更差，相关应为负
    lines.append(
        f"deep vs 上盘cover: Pearson={a['pearson']:.3f}, p={a['p']:.4g}"
    )
    return "\n".join(lines)


def plot_sample(sample: pd.DataFrame, spread: pd.DataFrame) -> None:
    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    ax.scatter(sample["pred_gd"], sample["home_let"], alpha=0.75, s=28, c="#1f4e79")
    lo = np.nanmin([sample["pred_gd"].min(), sample["home_let"].min()])
    hi = np.nanmax([sample["pred_gd"].max(), sample["home_let"].max()])
    ax.plot([lo, hi], [lo, hi], ls="--", c="#888", lw=1, label="y=x")
    a = corr_line(sample["pred_gd"], sample["home_let"])
    if np.isfinite(a["slope"]):
        xs = np.linspace(sample["pred_gd"].min(), sample["pred_gd"].max(), 50)
        ax.plot(xs, a["intercept"] + a["slope"] * xs, c="#c0392b", lw=1.5, label="回归")
    ax.set_xlabel("近5场 xG 隐含净胜 pred_gd")
    ax.set_ylabel("收盘主队让球 home_let = -AHCh")
    ax.set_title(f"A. 球头 vs xG  (R2={a['r2']:.3f})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    colors = np.where(spread["favorite_cover"] > 0, "#2e7d32", np.where(spread["favorite_cover"] < 0, "#c62828", "#f9a825"))
    ax.scatter(spread["fav_edge"], spread["favorite_cover"], alpha=0.75, s=28, c=colors)
    ax.axhline(0, c="#888", lw=1)
    ax.axvline(0, c="#888", lw=1)
    c = corr_line(spread["fav_edge"], spread["favorite_cover"])
    if np.isfinite(c["slope"]):
        xs = np.linspace(spread["fav_edge"].min(), spread["fav_edge"].max(), 50)
        ax.plot(xs, c["intercept"] + c["slope"] * xs, c="#1f4e79", lw=1.5)
    ax.set_xlabel("fav_edge = 上盘侧 xG优势 - 让球")
    ax.set_ylabel("上盘 cover（+1赢 / 0走 / -1输）")
    ax.set_title(f"B. 上盘结果 vs edge  (r={c['pearson']:.3f})")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"2025-26 五大联赛 · 近{SAMPLE_N}场 · 近5场xG 与收盘亚盘", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ah_xg_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    merged = build_merged()
    eligible = draw_sample(merged)
    spread = eligible[eligible["favorite"] != "pickem"].copy()

    cols = [
        "league",
        "date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "line",
        "home_let",
        "favorite",
        "平手用水位",
        "ah_home_odds",
        "ah_away_odds",
        "home_cover",
        "favorite_cover",
        "home_xg_for_l5",
        "home_xg_ag_l5",
        "away_xg_for_l5",
        "away_xg_ag_l5",
        "pre_xgd",
        "pred_gd",
        "pred_gd_l3",
        "fav_edge_l3",
        "edge",
        "fav_edge",
        "gd",
        "xg_home",
        "xg_away",
    ]
    eligible[cols].to_csv(PROC / "sample_200.csv", index=False)
    merged.to_csv(PROC / "all_merged.csv", index=False)

    text = summarize(eligible)
    pool = merged.dropna(subset=["home_xg_for_l5", "away_xg_for_l5", "line", "pred_gd"]).copy()
    pool = pool[(pool["date"] >= SAMPLE_START) & (pool["date"] < SAMPLE_END)]
    pool_sp = pool[pool["favorite"] != "pickem"].copy()
    pool_base = (
        "【对照总体】10月–4月全部满窗场次 n={} 上盘 {:.1%} / 下盘 {:.1%} / 走水 {:.1%}"
    ).format(
        len(pool_sp),
        (pool_sp.favorite_cover > 0).mean(),
        (pool_sp.favorite_cover < 0).mean(),
        (pool_sp.favorite_cover == 0).mean(),
    )
    pool_rep = deep_shallow_report(
        pool_sp,
        "pred_gd",
        f"【10月–4月全集 + 主场{HOME_ADV}】D中位数对半（上下盘接近对半的合理总体）",
        home_adv=HOME_ADV,
        split="median",
    )
    ha_l5 = deep_shallow_report(
        spread,
        "pred_gd",
        f"【近5场 xG + 主场{HOME_ADV}】D中位数对半",
        home_adv=HOME_ADV,
        split="median",
    )
    ha_l3 = deep_shallow_report(
        spread.dropna(subset=["pred_gd_l3"]),
        "pred_gd_l3",
        f"【近3场 xG + 主场{HOME_ADV}】D中位数对半",
        home_adv=HOME_ADV,
        split="median",
    )
    terc_l5 = deep_shallow_report(
        spread,
        "pred_gd",
        f"【近5场 xG + 主场{HOME_ADV}】三分位（中间=接近）",
        home_adv=HOME_ADV,
        split="tercile",
    )
    extra = "\n\n".join([pool_base, pool_rep, ha_l5, ha_l3, terc_l5])
    out_cols = [
        "league",
        "date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "line",
        "平手用水位",
        "ah_home_odds",
        "ah_away_odds",
        "pred_gd",
        "pred_ha",
        "fav_pred",
        "fav_let",
        "deep",
        "深浅",
        "favorite",
        "favorite_cover",
    ]
    attach_deep(spread, "pred_gd", HOME_ADV, split="median")[out_cols].to_csv(
        PROC / "deep_shallow_ha_l5.csv", index=False
    )
    (OUT / "summary.txt").write_text(text + "\n\n" + extra + "\n", encoding="utf-8")
    (OUT / "deep_shallow_ha.txt").write_text(extra + "\n", encoding="utf-8")
    plot_sample(eligible, spread)
    print(text)
    print()
    print(extra)
    print(f"\n样本 CSV: {PROC / 'sample_200.csv'}")
    print(f"图: {OUT / 'ah_xg_scatter.png'}")


if __name__ == "__main__":
    main()
