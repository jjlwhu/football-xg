"""近10场进球失球 × 变盘 × 开深开浅：找胜率≥55% 且 ROI 高的边。

进球失球：同联赛近10场（不分主客）场均进球/失球。
开深开浅：用这10场估出来的净胜，对比收盘让球（口径与 xG 开深相同：球头 − 预期净胜，大为开深）。
变盘：初盘 AHh → 收盘 AHCh。
超额：减去同一条收盘盘口的均值，避免把盘口底子当信号。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .analyze_ah_form_move import (
    OUT,
    PROC,
    add_baseline,
    load_matches,
    move_bucket,
)
from .analyze_ah_xg import HOME_ADV, wilson_ci

MIN_N = 150
HIT_MIN = 0.55
ROI_MIN = 0.03
EX_MIN = 0.02


def attach_l10_goals(df: pd.DataFrame) -> pd.DataFrame:
    hist: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    h_gf = np.full(len(df), np.nan)
    h_ga = np.full(len(df), np.nan)
    a_gf = np.full(len(df), np.nan)
    a_ga = np.full(len(df), np.nan)
    for i, r in enumerate(df.itertuples(index=False)):
        hk = (r.div, r.HomeTeam)
        ak = (r.div, r.AwayTeam)
        hh, aa = hist[hk], hist[ak]
        if len(hh) >= 10 and len(aa) >= 10:
            h10, a10 = hh[-10:], aa[-10:]
            h_gf[i] = float(np.mean([x[0] for x in h10]))
            h_ga[i] = float(np.mean([x[1] for x in h10]))
            a_gf[i] = float(np.mean([x[0] for x in a10]))
            a_ga[i] = float(np.mean([x[1] for x in a10]))
        hist[hk].append((float(r.FTHG), float(r.FTAG)))
        hist[ak].append((float(r.FTAG), float(r.FTHG)))
    out = df.copy()
    out["h10_gf"] = h_gf
    out["h10_ga"] = h_ga
    out["a10_gf"] = a_gf
    out["a10_ga"] = a_ga
    out["pred_home"] = (out["h10_gf"] + out["a10_ga"]) / 2
    out["pred_away"] = (out["a10_gf"] + out["h10_ga"]) / 2
    out["pred_gd"] = out["pred_home"] - out["pred_away"]
    out["pred_tot"] = out["pred_home"] + out["pred_away"]
    out["home_let"] = -out["close_line"]
    # 与 analyze_ah_xg.attach_deep 相同：deep = 上盘球头 − 上盘预期净胜；大为开深
    fav_pred = np.where(out["favorite"] == "home", out["pred_gd"] + HOME_ADV, -(out["pred_gd"] + HOME_ADV))
    out["deep"] = out["abs_line"] - fav_pred
    return out


def bins(s: pd.Series, labels: list[str]) -> pd.Series:
    return pd.qcut(s, len(labels), labels=labels, duplicates="drop")


def eval_side(sub: pd.DataFrame, side: str) -> dict | None:
    n = len(sub)
    if n < MIN_N:
        return None
    if side == "主队盘":
        hit = sub["home_hit"]
        roi = sub["pnl_home"]
        ex = sub["ex_pnl_home"]
    elif side == "客队盘":
        hit = (sub["home_cover"] < 0).astype(float)
        roi = sub["pnl_away"]
        ex = sub["ex_pnl_away"]
    elif side == "上盘":
        hit = sub["up_hit"]
        roi = sub["pnl_up"]
        ex = sub["ex_pnl_up"]
    else:
        hit = (sub["favorite_cover"] < 0).astype(float)
        roi = sub["pnl_dn"]
        ex = sub["ex_pnl_dn"]
    k = int(hit.sum())
    p = k / n
    lo, hi = wilson_ci(k, n)
    roi_m = float(roi.mean())
    ex_m = float(ex.mean())
    se = float(roi.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    by = roi.groupby(sub["season_path"]).mean()
    n_sz = int(by.notna().sum())
    pos = int((by > 0).sum())
    return {
        "n": n,
        "边": side,
        "胜率": p,
        "CI低": lo,
        "CI高": hi,
        "ROI": roi_m,
        "ROI超额": ex_m,
        "ROI se": se,
        "赛季数": n_sz,
        "ROI为正的赛季": pos,
    }


def scan(df: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(key, observed=True):
        for side in ("主队盘", "客队盘", "上盘", "下盘"):
            rec = eval_side(sub, side)
            if rec:
                rec["切片"] = str(name)
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def keep_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        (df["胜率"] >= HIT_MIN)
        & (df["ROI"] >= ROI_MIN)
        & (df["ROI超额"] >= EX_MIN)
        & (df["n"] >= MIN_N)
    ].sort_values(["ROI", "胜率", "n"], ascending=[False, False, False])


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = add_baseline(attach_l10_goals(load_matches()))
    d["变盘"] = d["home_steps"].map(move_bucket)
    d["变盘3"] = np.where(d["home_steps"] > 0, "主升", np.where(d["home_steps"] < 0, "主降", "盘不动"))
    d = d.dropna(subset=["h10_gf", "deep", "pred_tot"]).copy()
    d["深浅"] = bins(d["deep"], ["开浅", "接近", "开深"])
    d["总进球"] = bins(d["pred_tot"], ["小球", "中球", "大球"])
    d["主攻"] = np.where(d["h10_gf"] >= d["h10_gf"].median(), "主攻强", "主攻弱")
    d["主防"] = np.where(d["h10_ga"] <= d["h10_ga"].median(), "主防强", "主防弱")
    d["客攻"] = np.where(d["a10_gf"] >= d["a10_gf"].median(), "客攻强", "客攻弱")
    d["客防"] = np.where(d["a10_ga"] <= d["a10_ga"].median(), "客防强", "客防弱")

    d["三轴"] = d["变盘3"].astype(str) + " · " + d["深浅"].astype(str) + " · " + d["总进球"].astype(str)
    d["主队攻防"] = (
        d["变盘3"].astype(str)
        + " · "
        + d["深浅"].astype(str)
        + " · "
        + d["主攻"]
        + " · "
        + d["主防"]
    )
    d["客队攻防"] = (
        d["变盘3"].astype(str)
        + " · "
        + d["深浅"].astype(str)
        + " · "
        + d["客攻"]
        + " · "
        + d["客防"]
    )
    d["变盘深浅"] = d["变盘3"].astype(str) + " · " + d["深浅"].astype(str)
    d["攻防"] = d["主攻"] + " · " + d["主防"] + " · " + d["客攻"] + " · " + d["客防"]
    d["变盘攻防"] = d["变盘3"].astype(str) + " · " + d["攻防"]

    blocks = {
        "变盘×深浅": scan(d, "变盘深浅"),
        "变盘×深浅×总进球": scan(d, "三轴"),
        "变盘×深浅×主攻防": scan(d, "主队攻防"),
        "变盘×深浅×客攻防": scan(d, "客队攻防"),
        "变盘×双方攻防": scan(d, "变盘攻防"),
    }
    all_rows = []
    for src, df in blocks.items():
        df = df.copy()
        df["来源"] = src
        df.to_csv(PROC / f"ah_combo_{src}.csv", index=False)
        all_rows.append(df)
    cat = pd.concat(all_rows, ignore_index=True)
    cat.to_csv(PROC / "ah_combo_all.csv", index=False)

    hits = keep_hits(cat)
    # 同一切片四条边只留 ROI 最高
    if not hits.empty:
        hits = hits.sort_values("ROI", ascending=False).drop_duplicates(subset=["切片"], keep="first")
    hits.to_csv(PROC / "ah_combo_hits.csv", index=False)

    # 放宽：只看 55%+ROI>0，不要求超额和赛季
    loose = cat[(cat["胜率"] >= HIT_MIN) & (cat["ROI"] > 0) & (cat["n"] >= MIN_N)]
    loose = loose.sort_values(["ROI", "胜率"], ascending=False).drop_duplicates("切片")

    # 稳的：再加赛季
    stable = hits[hits["ROI为正的赛季"] >= 5] if not hits.empty else hits

    def show(df: pd.DataFrame, n: int = 20) -> str:
        if df is None or df.empty:
            return "(没有)"
        cols = [c for c in ["来源", "切片", "边", "n", "胜率", "CI低", "ROI", "ROI超额", "ROI为正的赛季", "赛季数"] if c in df.columns]
        return df[cols].head(n).round(4).to_string(index=False)

    # 两轴全表：变盘×深浅 四边
    two = blocks["变盘×深浅"].copy()
    two_best = two.sort_values("ROI", ascending=False).drop_duplicates("切片")

    lines = [
        f"样本（满近10场进球失球）n={len(d)}  联赛={d.league.nunique()}  2019-20～2025-26",
        f"开深=近10场估的上盘净胜小于收盘球头（盘开得比进球失球深）",
        f"门槛：n≥{MIN_N} 且 该边打出≥{HIT_MIN:.0%} 且 ROI≥{ROI_MIN:.0%} 且 相对同盘口超额≥{EX_MIN:.0%}",
        "",
        "=== 变盘 × 开深开浅（每格四边里 ROI 最高）===",
        show(two_best, 30),
        "",
        f"=== 过门槛 {len(hits)} 条（同切片只留最赚边）===",
        show(hits, 25),
        "",
        f"=== 过门槛且 ≥5 个赛季 ROI 为正：{len(stable)} 条 ===",
        show(stable, 15),
        "",
        f"=== 放宽到 胜率≥55% 且 ROI>0：{len(loose)} 条，前15 ===",
        show(loose, 15),
        "",
        f"全样本主队盘胜率 {(d.home_hit.mean()):.1%} ROI {d.pnl_home.mean():+.1%}；"
        f"客队盘胜率 {(d.home_cover<0).mean():.1%} ROI {d.pnl_away.mean():+.1%}；"
        f"上盘 {(d.up_hit.mean()):.1%} ROI {d.pnl_up.mean():+.1%}",
    ]
    text = "\n".join(lines)
    (OUT / "ah_combo.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\n--- 过门槛明细赛季检查预留 ---")
    print("hits n=", len(hits), "stable n=", len(stable))


if __name__ == "__main__":
    main()
