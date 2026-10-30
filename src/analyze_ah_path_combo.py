"""具体变盘路径（半球→半一这类）× 近10场得失球 × 开深开浅。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analyze_ah_combo import attach_l10_goals, bins, eval_side
from .analyze_ah_form_move import OUT, PROC, add_baseline, load_matches
from .analyze_ah_xg import wilson_ci

PATH_MIN = 80  # 这条变盘路径本身
CELL_MIN = 50  # 路径再切深浅/攻防


def eval_side_n(sub: pd.DataFrame, side: str, min_n: int) -> dict | None:
    old = eval_side.__defaults__
    # 临时放宽：直接复制逻辑
    n = len(sub)
    if n < min_n:
        return None
    if side == "主队盘":
        hit, roi, ex = sub["home_hit"], sub["pnl_home"], sub["ex_pnl_home"]
    elif side == "客队盘":
        hit = (sub["home_cover"] < 0).astype(float)
        roi, ex = sub["pnl_away"], sub["ex_pnl_away"]
    elif side == "上盘":
        hit, roi, ex = sub["up_hit"], sub["pnl_up"], sub["ex_pnl_up"]
    else:
        hit = (sub["favorite_cover"] < 0).astype(float)
        roi, ex = sub["pnl_dn"], sub["ex_pnl_dn"]
    k = int(hit.sum())
    p = k / n
    lo, hi = wilson_ci(k, n)
    by = roi.groupby(sub["season_path"]).mean()
    return {
        "n": n,
        "边": side,
        "胜率": p,
        "CI低": lo,
        "ROI": float(roi.mean()),
        "ROI超额": float(ex.mean()),
        "ROI se": float(roi.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "赛季数": int(by.notna().sum()),
        "ROI为正的赛季": int((by > 0).sum()),
    }


def best_of(sub: pd.DataFrame, min_n: int) -> dict | None:
    recs = [eval_side_n(sub, s, min_n) for s in ("主队盘", "客队盘", "上盘", "下盘")]
    recs = [r for r in recs if r]
    if not recs:
        return None
    return max(recs, key=lambda r: r["ROI"])


def scan_key(df: pd.DataFrame, key: str, min_n: int) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(key, observed=True):
        for side in ("主队盘", "客队盘", "上盘", "下盘"):
            rec = eval_side_n(sub, side, min_n)
            if rec:
                rec["切片"] = str(name)
                rows.append(rec)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = add_baseline(attach_l10_goals(load_matches()))
    d = d.dropna(subset=["h10_gf", "deep", "pred_tot", "transition"]).copy()
    d["moved"] = d["open_line"] != d["close_line"]
    def tercile_deep(s: pd.Series) -> pd.Series:
        r = s.rank(pct=True, method="average")
        return pd.Series(np.where(r <= 0.4, "开浅", np.where(r <= 0.6, "接近", "开深")), index=s.index)

    d["深浅"] = d.groupby("close_line", group_keys=False)["deep"].apply(tercile_deep)
    d["总进球"] = bins(d["pred_tot"], ["小球", "中球", "大球"])
    d["主攻"] = np.where(d["h10_gf"] >= d["h10_gf"].median(), "主攻强", "主攻弱")
    d["主防"] = np.where(d["h10_ga"] <= d["h10_ga"].median(), "主防强", "主防弱")
    d["攻防"] = d["主攻"] + d["主防"].str.replace("主", "")

    moved = d[d["moved"]].copy()
    vc = moved["transition"].value_counts()
    keep = set(vc[vc >= PATH_MIN].index)
    paths = moved[moved["transition"].isin(keep)].copy()

    path_rows = []
    for name, sub in paths.groupby("transition"):
        rec = best_of(sub, PATH_MIN)
        if rec:
            rec["切片"] = name
            rec["初盘"] = sub["open_lab"].iloc[0]
            rec["收盘"] = sub["close_lab"].iloc[0]
            rec["档数"] = int(sub["home_steps"].iloc[0])
            path_rows.append(rec)
    path_df = pd.DataFrame(path_rows).sort_values(["ROI", "胜率"], ascending=False)

    paths["路径深浅"] = paths["transition"] + " · " + paths["深浅"].astype(str)
    paths["路径进球"] = paths["transition"] + " · " + paths["总进球"].astype(str)
    paths["路径攻防"] = paths["transition"] + " · " + paths["攻防"].astype(str)
    paths["三轴"] = paths["transition"] + " · " + paths["深浅"].astype(str) + " · " + paths["攻防"].astype(str)
    paths["路径深浅进球"] = (
        paths["transition"] + " · " + paths["深浅"].astype(str) + " · " + paths["总进球"].astype(str)
    )

    deep_df = scan_key(paths, "路径深浅", CELL_MIN)
    tot_df = scan_key(paths, "路径进球", CELL_MIN)
    atk_df = scan_key(paths, "路径攻防", CELL_MIN)
    tri_df = scan_key(paths, "三轴", CELL_MIN)
    dtot_df = scan_key(paths, "路径深浅进球", CELL_MIN)

    def collapse(df: pd.DataFrame, src: str) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.sort_values("ROI", ascending=False).drop_duplicates("切片")
        out["来源"] = src
        return out

    all_best = pd.concat(
        [
            collapse(path_df.assign(来源="仅路径"), "仅路径") if False else path_df.assign(来源="仅路径"),
            collapse(deep_df, "路径×深浅"),
            collapse(tot_df, "路径×总进球"),
            collapse(atk_df, "路径×攻防"),
            collapse(dtot_df, "路径×深浅×总进球"),
            collapse(tri_df, "路径×深浅×攻防"),
        ],
        ignore_index=True,
    )
    path_df.to_csv(PROC / "ah_path_only.csv", index=False)
    deep_df.to_csv(PROC / "ah_path_x_deep.csv", index=False)
    all_best.to_csv(PROC / "ah_path_combo_best.csv", index=False)

    def gate(df: pd.DataFrame, hit=0.55, roi=0.03, ex=0.02, n=CELL_MIN) -> pd.DataFrame:
        if df.empty:
            return df
        return df[(df["胜率"] >= hit) & (df["ROI"] >= roi) & (df["ROI超额"] >= ex) & (df["n"] >= n)]

    hits55 = gate(all_best)
    hits52 = gate(all_best, hit=0.52, roi=0.03, ex=0.0)
    loose = all_best[(all_best["胜率"] >= 0.55) & (all_best["n"] >= CELL_MIN)]
    roi_pos = all_best[(all_best["ROI"] >= 0.03) & (all_best["n"] >= CELL_MIN)].sort_values("ROI", ascending=False)

    def show(df: pd.DataFrame, n=20) -> str:
        if df is None or df.empty:
            return "(没有)"
        cols = [c for c in ["来源", "切片", "边", "n", "胜率", "CI低", "ROI", "ROI超额", "ROI为正的赛季", "赛季数"] if c in df.columns]
        return df[cols].head(n).round(4).to_string(index=False)

    lines = [
        f"具体变盘路径（初盘≠收盘）n={len(moved)}，其中路径本身≥{PATH_MIN}场的有 {len(keep)} 条、{len(paths)} 场",
        f"开深开浅按同一条收盘盘口内三分位。得失球=近10场场均进/失。",
        "",
        f"=== 胜率≥55% 且 ROI≥3% 且超额≥2%：{len(hits55)} 条 ===",
        show(hits55.sort_values(["ROI", "n"], ascending=False), 40),
        "",
        f"=== 胜率≥55%（不问 ROI）：{len(loose)} 条 ===",
        show(loose.sort_values(["胜率", "ROI"], ascending=False), 20),
        "",
        f"=== 放宽到胜率≥52% 且 ROI≥3%：{len(hits52)} 条 ===",
        show(hits52.sort_values("ROI", ascending=False), 20),
        "",
        "=== 只看具体路径（不切深浅/得失球），ROI 最高 15 条 ===",
        show(path_df.head(15), 15),
        "",
        "=== 只看具体路径，胜率最高 10 条 ===",
        show(path_df.sort_values("胜率", ascending=False).head(10), 10),
        "",
        "=== 路径×深浅，ROI 最高 15 条 ===",
        show(collapse(deep_df, "路径×深浅").head(15), 15),
        "",
        "=== 路径×深浅×得失球，ROI≥3% 且 n≥50 前 15 ===",
        show(roi_pos.head(15), 15),
    ]
    text = "\n".join(lines)
    (OUT / "ah_path_combo.txt").write_text(text, encoding="utf-8")
    print(text)
    print("keep paths:", len(keep))


if __name__ == "__main__":
    main()
