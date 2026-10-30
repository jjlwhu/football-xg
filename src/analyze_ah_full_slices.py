"""2019-20～2025-26 五大联赛：近3/5场 xG × 盘口类型，同时看打出率与 ROI。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .analyze_ah_l3_slices import LINE_ORDER, decorate, line_label, metrics, pnl, prepare
from .analyze_ah_xg import HOME_ADV, OUT, PROC, attach_deep, wilson_ci

MIN_N = 100


def with_roi_side(r: dict) -> dict:
    r = dict(r)
    r["方向ROI"] = r["上盘ROI"] if r["方向"] == "上盘" else r["下盘ROI"]
    r["更赚边"] = "上盘" if r["上盘ROI"] >= r["下盘ROI"] else "下盘"
    r["更赚ROI"] = max(r["上盘ROI"], r["下盘ROI"])
    r["双过"] = bool(r["CI低>50%"] and r["方向ROI"] > 0)
    return r


def add_l5(d: pd.DataFrame) -> pd.DataFrame:
    d5 = attach_deep(d, "pred_gd", home_adv=HOME_ADV, split="p40")
    d = d.copy()
    d["深浅_l5"] = d5.reindex(d.index)["深浅"]
    d["deep_l5"] = d5.reindex(d.index)["deep"]
    return d


def scan_full(d: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add(name: str, sub: pd.DataFrame, min_n: int = MIN_N) -> None:
        r = metrics(name, sub, min_n=min_n)
        if r:
            rows.append(with_roi_side(r))

    add("全样本", d)
    add("主队上盘", d[d["favorite"] == "home"])
    add("客队上盘", d[d["favorite"] == "away"])
    add("上盘低水", d[d["上盘低水"]])
    add("上盘高水", d[~d["上盘低水"]])
    for lab in ("开浅", "接近", "开深"):
        add(f"近3场xG {lab}", d[d["深浅"] == lab])
        add(f"近5场xG {lab}", d[d["深浅_l5"] == lab])
    for lab in ("开浅", "开深"):
        add(f"近3场 D中位数{lab}", d[d["深浅对半"] == lab])

    for q, tag in [(0.2, "最20%"), (0.15, "最15%"), (0.1, "最10%")]:
        lo, hi = d["deep"].quantile(q), d["deep"].quantile(1 - q)
        add(f"近3场D最浅{tag}", d[d["deep"] <= lo])
        add(f"近3场D最深{tag}", d[d["deep"] >= hi])

    for thr in (0.25, 0.5, 0.75):
        add(f"近3场开浅 D<{-thr}", d[d["deep"] < -thr])
        add(f"近3场开深 D>{thr}", d[d["deep"] > thr])

    add("开浅·主上盘", d[(d["深浅"] == "开浅") & (d["favorite"] == "home")])
    add("开浅·客上盘", d[(d["深浅"] == "开浅") & (d["favorite"] == "away")])
    add("开深·主上盘", d[(d["深浅"] == "开深") & (d["favorite"] == "home")])
    add("开深·客上盘", d[(d["深浅"] == "开深") & (d["favorite"] == "away")])
    add("开浅·上盘低水", d[(d["深浅"] == "开浅") & d["上盘低水"]])
    add("开深·上盘低水", d[(d["深浅"] == "开深") & d["上盘低水"]])
    add("开浅·上盘高水", d[(d["深浅"] == "开浅") & ~d["上盘低水"]])
    add("开深·上盘高水", d[(d["深浅"] == "开深") & ~d["上盘低水"]])

    for lg, g in d.groupby("league"):
        add(f"联赛={lg}", g)

    for line in LINE_ORDER:
        sub = d[np.isclose(d["abs_line"], line)]
        lab = line_label(line)
        add(f"盘口={lab}", sub, min_n=80)
        add(f"{lab}·主上盘", sub[sub["favorite"] == "home"], min_n=80)
        add(f"{lab}·客上盘", sub[sub["favorite"] == "away"], min_n=80)
        add(f"{lab}·上盘低水", sub[sub["上盘低水"]], min_n=80)
        add(f"{lab}·上盘高水", sub[~sub["上盘低水"]], min_n=80)
        for deep in ("开浅", "接近", "开深"):
            ss = sub[sub["深浅"] == deep]
            add(f"{lab}·近3场{deep}", ss, min_n=80)
            add(f"{lab}·近5场{deep}", sub[sub["深浅_l5"] == deep], min_n=80)
            add(f"{lab}·近3场{deep}·主上盘", ss[ss["favorite"] == "home"], min_n=80)
            add(f"{lab}·近3场{deep}·客上盘", ss[ss["favorite"] == "away"], min_n=80)
        for lg, g in sub.groupby("league"):
            add(f"{lab}·{lg}", g, min_n=80)

    df = pd.DataFrame(rows).drop_duplicates(subset=["切片"])
    return df.sort_values(["双过", "方向ROI", "CI低>50%", "点估计", "n"], ascending=[False, False, False, False, False]).reset_index(drop=True)


def loso(d: pd.DataFrame, mask_fn, name: str) -> list[dict]:
    recs = []
    seasons = sorted(d["season_path"].unique())
    for hold in seasons:
        train = d[d["season_path"] != hold]
        test = d[d["season_path"] == hold]
        tr = metrics(f"{name}|train {hold}", mask_fn(train), min_n=40)
        te = metrics(f"{name}|test {hold}", mask_fn(test), min_n=20)
        if not tr or not te:
            continue
        tr, te = with_roi_side(tr), with_roi_side(te)
        recs.append(
            {
                "切片": name,
                "留出赛季": hold,
                "train_n": tr["n"],
                "test_n": te["n"],
                "train方向": tr["方向"],
                "train点估计": tr["点估计"],
                "train方向ROI": tr["方向ROI"],
                "test方向": te["方向"],
                "test点估计": te["点估计"],
                "test上盘ROI": te["上盘ROI"],
                "test下盘ROI": te["下盘ROI"],
                "test方向ROI": te["方向ROI"],
            }
        )
    return recs


def by_line_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in LINE_ORDER:
        r = metrics(line_label(line), d[np.isclose(d["abs_line"], line)], min_n=50)
        if r:
            r["球头"] = line
            rows.append(with_roi_side(r))
    return pd.DataFrame(rows)


def cross_line_deep(d: pd.DataFrame, col: str = "深浅") -> pd.DataFrame:
    rows = []
    for line in LINE_ORDER:
        sub = d[np.isclose(d["abs_line"], line)]
        for lab in ("开浅", "接近", "开深"):
            r = metrics(f"{line_label(line)}·{lab}", sub[sub[col] == lab], min_n=80)
            if r:
                r["盘口"] = line_label(line)
                r["条件"] = lab
                rows.append(with_roi_side(r))
    return pd.DataFrame(rows)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = add_l5(prepare("complete"))
    slices = scan_full(d)
    by_line = by_line_table(d)
    cross = cross_line_deep(d)
    cross5 = cross_line_deep(d, "深浅_l5")
    slices.to_csv(PROC / "ah_full_slices.csv", index=False)
    by_line.to_csv(PROC / "ah_full_by_line.csv", index=False)
    cross.to_csv(PROC / "ah_full_line_x_l3.csv", index=False)

    both = slices[(slices["n"] >= 100) & (slices["CI低>50%"]) & (slices["方向ROI"] > 0)]
    hit_only = slices[(slices["n"] >= 100) & (slices["CI低>50%"])]
    roi_only = slices[(slices["n"] >= 100) & (slices["方向ROI"] > 0.03) & (slices["点估计"] >= 0.52)]

    masks = {
        "盘口=半一": lambda x: x[np.isclose(x.abs_line, 0.75)],
        "半一·主上盘": lambda x: x[np.isclose(x.abs_line, 0.75) & (x.favorite == "home")],
        "半一·近3场开深": lambda x: x[np.isclose(x.abs_line, 0.75) & (x["深浅"] == "开深")],
        "半一·近3场开深·客上盘": lambda x: x[
            np.isclose(x.abs_line, 0.75) & (x["深浅"] == "开深") & (x.favorite == "away")
        ],
        "半一·近3场开浅": lambda x: x[np.isclose(x.abs_line, 0.75) & (x["深浅"] == "开浅")],
        "平半·近3场开深": lambda x: x[np.isclose(x.abs_line, 0.25) & (x["深浅"] == "开深")],
        "半球·近3场开深": lambda x: x[np.isclose(x.abs_line, 0.5) & (x["深浅"] == "开深")],
        "球半 无条件": lambda x: x[np.isclose(x.abs_line, 1.5)],
        "一球半·近3场开深": lambda x: x[np.isclose(x.abs_line, 1.25) & (x["深浅"] == "开深")],
    }
    # pick extra from both.head if named
    loso_rows = []
    for name, fn in masks.items():
        loso_rows.extend(loso(d, fn, name))
    loso_df = pd.DataFrame(loso_rows)

    q40, q60 = d["deep"].quantile(0.4), d["deep"].quantile(0.6)
    vc = d["深浅"].value_counts()
    lines = [
        f"2019-20～2025-26 五大联赛满窗 n={len(d)}（近3场xG + 收盘亚盘，不含 2026-27）",
        f"深浅切法: 开浅40% / 接近20% / 开深40%  （D≤{q40:.3f} 开浅；{q40:.3f}<D≤{q60:.3f} 接近；D>{q60:.3f} 开深）",
        f"实际档内 n 开浅={int(vc.get('开浅',0))} 接近={int(vc.get('接近',0))} 开深={int(vc.get('开深',0))}",
        f"全样本上盘 {(d.favorite_cover>0).mean():.1%} 下盘 {(d.favorite_cover<0).mean():.1%} "
        f"cover={d.favorite_cover.mean():+.3f} 上ROI={d.pnl_up.mean():+.1%} 下ROI={d.pnl_dn.mean():+.1%}",
        f"近3场D vs cover Pearson={d['deep'].corr(d['favorite_cover']):.3f}；"
        f"近5场D vs cover Pearson={d['deep_l5'].corr(d['favorite_cover']):.3f}",
        f"扫描 {len(slices)} 条，n≥{MIN_N}",
        "",
        "=== 按盘口（无条件）===",
        by_line[["切片", "n", "上盘打出", "下盘打出", "均cover", "上盘ROI", "下盘ROI", "方向", "方向ROI", "赢半占比"]]
        .round(3)
        .to_string(index=False),
        "",
        "=== 盘口 × 近3场xG深浅 ===",
        cross[["盘口", "条件", "n", "方向", "点估计", "均cover", "上盘ROI", "下盘ROI", "方向ROI", "CI低"]]
        .round(3)
        .to_string(index=False),
        "",
        "=== 同时：CI下限>50% 且 该方向ROI>0 且 n≥100 ===",
    ]
    if both.empty:
        lines.append("没有切片同时满足。")
    else:
        for _, r in both.iterrows():
            lines.append(
                f"{r['切片']}: n={r['n']} {r['方向']} {r['点估计']:.1%} "
                f"({r['CI低']:.1%}~{r['CI高']:.1%}) ROI={r['方向ROI']:+.1%} "
                f"上{r['上盘ROI']:+.1%} 下{r['下盘ROI']:+.1%} cover={r['均cover']:+.3f}"
            )
    lines += ["", "=== CI下限>50%（只看概率，n≥100）==="]
    for _, r in hit_only.head(20).iterrows():
        flag = "  且ROI>0" if r["方向ROI"] > 0 else f"  该边ROI={r['方向ROI']:+.1%}"
        lines.append(
            f"{r['切片']}: n={r['n']} {r['方向']} {r['点估计']:.1%} ({r['CI低']:.1%}~{r['CI高']:.1%}){flag}"
        )
    lines += ["", "=== 该方向ROI>3% 且打出≥52%（n≥100，概率未必过CI）==="]
    for _, r in roi_only.head(15).iterrows():
        lines.append(
            f"{r['切片']}: n={r['n']} {r['方向']} {r['点估计']:.1%} ROI={r['方向ROI']:+.1%} "
            f"CI低={r['CI低']:.1%}"
        )
    if len(loso_df):
        lines += ["", "=== 留一赛季：test 上盘ROI ==="]
        g = (
            loso_df.groupby("切片")
            .agg(
                年数=("留出赛季", "size"),
                test均点估计=("test点估计", "mean"),
                test均上ROI=("test上盘ROI", "mean"),
                test均下ROI=("test下盘ROI", "mean"),
                上盘ROI为正的年=("test上盘ROI", lambda s: int((s > 0).sum())),
            )
            .round(3)
        )
        lines.append(g.to_string())

    text = "\n".join(lines)
    (OUT / "ah_full_slices.txt").write_text(text + "\n", encoding="utf-8")
    payload = {
        "n": int(len(d)),
        "split": "p40",
        "q40": float(q40),
        "q60": float(q60),
        "n_shallow": int(vc.get("开浅", 0)),
        "n_mid": int(vc.get("接近", 0)),
        "n_deep": int(vc.get("开深", 0)),
        "base_up": float((d.favorite_cover > 0).mean()),
        "base_dn": float((d.favorite_cover < 0).mean()),
        "cover": float(d.favorite_cover.mean()),
        "roi_up": float(d.pnl_up.mean()),
        "roi_dn": float(d.pnl_dn.mean()),
        "corr_l3": float(d["deep"].corr(d["favorite_cover"])),
        "corr_l5": float(d["deep_l5"].corr(d["favorite_cover"])),
        "n_slices": int(len(slices)),
        "by_line": by_line.to_dict(orient="records"),
        "cross_l3": cross.to_dict(orient="records"),
        "cross_l5": cross5.to_dict(orient="records") if len(cross5) else [],
        "both": both.head(25).to_dict(orient="records"),
        "hit_only": hit_only.head(20).to_dict(orient="records"),
        "roi_only": roi_only.head(20).to_dict(orient="records"),
        "loso": loso_df.to_dict(orient="records") if len(loso_df) else [],
    }
    (PROC / "ah_full_slices.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(text)
    print(f"\n写出 {OUT / 'ah_full_slices.txt'}")


if __name__ == "__main__":
    main()
