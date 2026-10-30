"""近 3 场 xG 与收盘大小球 2.5：扫描点估计 ≥55% 的切片。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analyze_ah_xg import (
    HOME_ADV,
    OUT,
    PROC,
    SAMPLE_END,
    SAMPLE_START,
    build_merged,
    wilson_ci,
)

OU_LINE = 2.5
MIN_N = 50


def prepare() -> pd.DataFrame:
    m = build_merged()
    d = m.dropna(subset=["pred_gd_l3", "FTHG", "FTAG"]).copy()
    d = d[(d["date"] >= SAMPLE_START) & (d["date"] < SAMPLE_END)]
    d["goals"] = d["FTHG"] + d["FTAG"]
    d["pred_home_l3"] = (d["home_xg_for_l3"] + d["away_xg_ag_l3"]) / 2
    d["pred_away_l3"] = (d["away_xg_for_l3"] + d["home_xg_ag_l3"]) / 2
    d["pred_total"] = d["pred_home_l3"] + d["pred_away_l3"]
    d["gap"] = d["pred_total"] - OU_LINE  # >0 表示 xG 比 2.5 高，更该出大
    d["over"] = d["goals"] > OU_LINE
    d["under"] = d["goals"] < OU_LINE
    over_col = "AvgC>2.5" if "AvgC>2.5" in d.columns else "B365C>2.5"
    under_col = "AvgC<2.5" if "AvgC<2.5" in d.columns else "B365C<2.5"
    d["over_odds"] = pd.to_numeric(d[over_col], errors="coerce")
    d["under_odds"] = pd.to_numeric(d[under_col], errors="coerce")
    d["mkt_over"] = d["over_odds"] < d["under_odds"]  # 大球低水
    d["mkt_under"] = d["under_odds"] < d["over_odds"]
    d = d.dropna(subset=["pred_total"])
    return d


def slice_rate(name: str, sub: pd.DataFrame, min_n: int = MIN_N) -> dict | None:
    n = len(sub)
    if n < min_n:
        return None
    rec = {"切片": name, "n": n}
    best = None
    for side, mask in (("大球", sub["over"]), ("小球", sub["under"])):
        k = int(mask.sum())
        p = k / n
        lo, hi = wilson_ci(k, n)
        rec[f"{side}率"] = p
        rec[f"{side}CI低"] = lo
        if best is None or p > best[1]:
            best = (side, p, lo, hi)
    rec["方向"] = best[0]
    rec["点估计"] = best[1]
    rec["CI低"] = best[2]
    rec["CI高"] = best[3]
    rec["≥55%"] = best[1] >= 0.55
    rec["CI低>50%"] = best[2] > 0.5
    return rec


def scan(d: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add(name: str, sub: pd.DataFrame, min_n: int = MIN_N) -> None:
        r = slice_rate(name, sub, min_n=min_n)
        if r:
            rows.append(r)

    add("中段全样本 2.5", d)
    add("收盘大球低水", d[d["mkt_over"]])
    add("收盘小球低水", d[d["mkt_under"]])

    med = d["gap"].median()
    add("xG总进球相对2.5 偏低(gap≤中位数)", d[d["gap"] <= med])
    add("xG总进球相对2.5 偏高(gap>中位数)", d[d["gap"] > med])

    for q, lab in [(0.33, "最低1/3"), (0.25, "最低25%"), (0.2, "最低20%"), (0.15, "最低15%"), (0.1, "最低10%")]:
        lo, hi = d["gap"].quantile(q), d["gap"].quantile(1 - q)
        add(f"xG相对2.5 {lab}（更该出小）", d[d["gap"] <= lo], min_n=40)
        add(f"xG相对2.5 最高{int((1-q)*100) if q!=0.33 else 67}%对称尾（更该出大）", d[d["gap"] >= hi], min_n=40)

    for thr in (2.0, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2):
        add(f"近3场xG总进球 ≥ {thr}", d[d["pred_total"] >= thr], min_n=40)
        add(f"近3场xG总进球 ≤ {thr}", d[d["pred_total"] <= thr], min_n=40)

    add("xG看大(gap>0.25) 且 市场低水也大", d[(d["gap"] > 0.25) & d["mkt_over"]], min_n=40)
    add("xG看大(gap>0.25) 且 市场低水却小", d[(d["gap"] > 0.25) & d["mkt_under"]], min_n=40)
    add("xG看小(gap<-0.25) 且 市场低水也小", d[(d["gap"] < -0.25) & d["mkt_under"]], min_n=40)
    add("xG看小(gap<-0.25) 且 市场低水却大", d[(d["gap"] < -0.25) & d["mkt_over"]], min_n=40)
    add("xG看大(gap>0.4)", d[d["gap"] > 0.4], min_n=40)
    add("xG看小(gap<-0.4)", d[d["gap"] < -0.4], min_n=40)

    # 近3场双方场均总xG（不错配，只用各自创造）
    d = d.copy()
    d["raw_for"] = d["home_xg_for_l3"] + d["away_xg_for_l3"]
    for thr in (2.2, 2.5, 2.8, 3.0):
        add(f"双方近3场创造xG之和 ≥ {thr}", d[d["raw_for"] >= thr], min_n=40)
        add(f"双方近3场创造xG之和 ≤ {thr}", d[d["raw_for"] <= thr], min_n=40)

    df = pd.DataFrame(rows).drop_duplicates(subset=["切片"])
    return df.sort_values(["≥55%", "点估计", "n"], ascending=[False, False, False]).reset_index(drop=True)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = prepare()
    base_over = d["over"].mean()
    corr = d["pred_total"].corr(d["goals"])
    table = scan(d)
    table.to_csv(PROC / "ou_l3_slices.csv", index=False)

    lines = [
        "近3场 xG vs 收盘大小球 2.5（football-data 只有 2.5 这一档）",
        f"样本: {SAMPLE_START}～{SAMPLE_END} n={len(d)} 实际大球 {base_over:.1%} 小球 {d['under'].mean():.1%}",
        f"近3场xG隐含总进球 vs 实际总进球 Pearson={corr:.3f}",
        f"xG隐含总进球均值={d['pred_total'].mean():.2f} 实际场均进球={d['goals'].mean():.2f}",
        f"主场常数 {HOME_ADV} 未加进总进球（大小球用攻防错配总和）",
        "",
        "=== 点估计 ≥55% ===",
    ]
    hit = table[table["≥55%"]]
    if hit.empty:
        lines.append("无")
    else:
        for r in hit.itertuples(index=False):
            flag = "  CI低>50%" if bool(r[-1] if False else table.loc[table["切片"]==r.切片, "CI低>50%"].iloc[0]) else ""
            lines.append(
                f"{r.切片}: n={r.n} → {r.方向} {r.点估计:.1%} ({r.CI低:.1%}~{r.CI高:.1%}){flag}"
            )
    lines.append("")
    lines.append("=== CI下限>50%（更站得住）===")
    ok = table[table["CI低>50%"]]
    if ok.empty:
        lines.append("无")
    else:
        for r in ok.itertuples(index=False):
            lines.append(
                f"{r.切片}: n={r.n} → {r.方向} {r.点估计:.1%} ({r.CI低:.1%}~{r.CI高:.1%})"
            )
    lines.append("")
    lines.append("=== 点估计最高的 12 条 ===")
    for r in table.head(12).itertuples(index=False):
        lines.append(
            f"{r.切片}: n={r.n} → {r.方向} {r.点估计:.1%} ({r.CI低:.1%}~{r.CI高:.1%})"
        )
    lines.append("")
    lines.append(f"共扫描 {len(table)} 条切片，≥55% 需警惕多重检验。")
    text = "\n".join(lines)
    (OUT / "ou_xg_l3.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n明细: {PROC / 'ou_l3_slices.csv'}")


if __name__ == "__main__":
    main()
