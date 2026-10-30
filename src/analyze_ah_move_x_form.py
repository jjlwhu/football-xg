"""全变盘扫描 + 变盘×同主客近10场交叉。找「绝对 ROI 为正」且相对同收盘超额为正的边。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .analyze_ah_form_move import (
    OUT,
    PROC,
    _pack,
    add_baseline,
    attach_venue10,
    form_bucket,
    load_matches,
    move_bucket,
)
from .analyze_ah_l3_slices import line_label


def best_side(sub: pd.DataFrame) -> dict:
    n = len(sub)
    home = float(sub["pnl_home"].mean())
    away = float(sub["pnl_away"].mean())
    up = float(sub["pnl_up"].mean())
    dn = float(sub["pnl_dn"].mean())
    ex_h = float(sub["ex_pnl_home"].mean())
    ex_a = float(sub["ex_pnl_away"].mean())
    sides = [
        ("主队盘", home, ex_h, sub["ex_pnl_home"]),
        ("客队盘", away, ex_a, sub["ex_pnl_away"]),
        ("上盘", up, float(sub["ex_pnl_up"].mean()), sub["ex_pnl_up"]),
        ("下盘", dn, float(sub["ex_pnl_dn"].mean()), sub["ex_pnl_dn"]),
    ]
    name, roi, ex, series = max(sides, key=lambda x: x[1])
    by = series.groupby(sub["season_path"]).mean()
    n_sz = int(by.notna().sum())
    pos = int((by > 0).sum())
    se = float(series.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    hit = {
        "主队盘": float(sub["home_hit"].mean()),
        "客队盘": 1.0 - float(sub["home_hit"].mean()) - float((sub["home_cover"] == 0).mean()),
        "上盘": float(sub["up_hit"].mean()),
        "下盘": float((sub["favorite_cover"] < 0).mean()),
    }[name]
    return {
        "n": n,
        "更赚边": name,
        "ROI": roi,
        "ROI超额": ex,
        "ROI se": se,
        "打出": hit,
        "赛季数": n_sz,
        "ROI为正的赛季": pos,
        "主队ROI": home,
        "客队ROI": away,
        "上盘ROI": up,
        "下盘ROI": dn,
        "主队ROI超额": ex_h,
        "客队ROI超额": ex_a,
    }


def scan_groups(df: pd.DataFrame, key: str, min_n: int) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(key, observed=True):
        if len(sub) < min_n:
            continue
        rec = best_side(sub)
        rec["切片"] = str(name)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["ROI", "ROI超额", "n"], ascending=[False, False, False]).reset_index(drop=True)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = add_baseline(attach_venue10(load_matches()))
    d["变盘"] = d["home_steps"].map(move_bucket)
    d["abs_lab"] = d["abs_line"].map(line_label)
    has = d["form_edge"].notna()
    d.loc[has, "同主客档"] = d.loc[has, "form_edge"].map(form_bucket)

    def agree(r) -> str:
        if pd.isna(r.form_edge):
            return "无近10场"
        if r.home_steps == 0:
            if abs(r.form_edge) < 0.25:
                return "盘不动·赛果接近"
            return "盘不动·赛果偏离"
        # 变盘与同主客是否同向：主升且赛果强主 = 同向
        same = (r.home_steps > 0 and r.form_edge > 0) or (r.home_steps < 0 and r.form_edge < 0)
        if same:
            return "变盘与赛果同向"
        return "变盘与赛果反向"

    d["同向"] = [agree(r) for r in d.itertuples(index=False)]
    form = d[has].copy()
    form["交叉"] = form["变盘"].astype(str) + " × " + form["同主客档"].astype(str)

    trans = scan_groups(d, "transition", min_n=200)
    trans.to_csv(PROC / "ah_all_transitions_roi.csv", index=False)

    # 只看真正变了的（去掉 平手→平手 这类）
    moved = d[d["home_steps"] != 0]
    trans_moved = scan_groups(moved, "transition", min_n=150)
    trans_moved.to_csv(PROC / "ah_moved_transitions_roi.csv", index=False)

    cross = scan_groups(form, "交叉", min_n=150)
    cross.to_csv(PROC / "ah_move_x_form.csv", index=False)

    agree_df = scan_groups(d, "同向", min_n=200)
    agree_df.to_csv(PROC / "ah_move_form_agree.csv", index=False)

    # 变盘档 × 同主客档（5×5）
    grid_rows = []
    for mv, g1 in form.groupby("变盘"):
        for fm, g2 in g1.groupby("同主客档"):
            if len(g2) < 80:
                continue
            rec = best_side(g2)
            rec["变盘"] = mv
            rec["同主客档"] = fm
            rec["切片"] = f"{mv} × {fm}"
            grid_rows.append(rec)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(PROC / "ah_move_x_form_grid.csv", index=False)

    def keep_alpha(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df[
            (df["n"] >= 150)
            & (df["ROI"] > 0.03)
            & (df["ROI超额"] > 0.02)
            & (df["ROI为正的赛季"] >= 5)
            & (df["赛季数"] >= 6)
        ]

    hits = pd.concat(
        [
            trans.assign(来源="全部变盘路径"),
            trans_moved.assign(来源="只含真正变盘"),
            cross.assign(来源="变盘×近10场"),
            agree_df.assign(来源="同向/反向"),
        ],
        ignore_index=True,
    )
    alpha = keep_alpha(hits)
    alpha.to_csv(PROC / "ah_move_form_alpha.csv", index=False)

    # 交叉表：变盘档 × 同主客，主队ROI（不是更赚边）方便读方向
    pivot_home = form.pivot_table(
        index="变盘",
        columns="同主客档",
        values="pnl_home",
        aggfunc="mean",
        observed=True,
    )
    pivot_n = form.pivot_table(
        index="变盘",
        columns="同主客档",
        values="pnl_home",
        aggfunc="size",
        observed=True,
    )
    order_m = ["主升≥2档", "主升1档", "盘口不变", "主降1档", "主降≥2档"]
    order_f = [
        "同主客比盘口弱≥0.75球",
        "同主客比盘口弱0.25~0.75",
        "同主客接近盘口",
        "同主客比盘口强0.25~0.75",
        "同主客比盘口强≥0.75球",
    ]
    pivot_home = pivot_home.reindex(index=order_m, columns=order_f)
    pivot_n = pivot_n.reindex(index=order_m, columns=order_f)
    pivot_away = form.pivot_table(
        index="变盘", columns="同主客档", values="pnl_away", aggfunc="mean", observed=True
    ).reindex(index=order_m, columns=order_f)
    pivot_exh = form.pivot_table(
        index="变盘", columns="同主客档", values="ex_pnl_home", aggfunc="mean", observed=True
    ).reindex(index=order_m, columns=order_f)

    def fmt_df(df: pd.DataFrame, pct: bool) -> str:
        show = df.copy()
        if pct:
            show = show.map(lambda x: "" if pd.isna(x) else f"{x:+.1%}")
        else:
            show = show.map(lambda x: "" if pd.isna(x) else f"{int(x)}")
        return show.to_string()

    top_move = trans_moved.head(15)
    top_cross = cross.head(15)
    worst = trans.sort_values("ROI").head(8)

    lines = [
        "口径：ROI 是按收盘水位买这一边的平均盈亏。超额 = 减去同一条收盘盘口的均值。",
        "「更赚边」是主队盘/客队盘/上盘/下盘里绝对 ROI 最高的那一边。",
        f"全样本 n={len(d)}  有近10场 n={len(form)}  真正变盘 n={len(moved)}",
        f"全样本主队ROI {d.pnl_home.mean():+.1%} 客队ROI {d.pnl_away.mean():+.1%}",
        "",
        "=== 一句话 ===",
        "没有通过「n≥150、绝对ROI>3%、相对同盘口超额>2%、至少6个赛季里5个为正」的切片。"
        if alpha.empty
        else f"通过门槛的切片 {len(alpha)} 条。",
        "",
        "=== 变盘 vs 近10场：同向还是反向 ===",
        agree_df.round(4).to_string(index=False),
        "",
        "=== 变盘档 × 同主客：主队ROI ===",
        fmt_df(pivot_home, True),
        "",
        "=== 同上，客队ROI ===",
        fmt_df(pivot_away, True),
        "",
        "=== 同上，样本量 ===",
        fmt_df(pivot_n, False),
        "",
        "=== 同上，主队ROI超额（扣掉盘口底子）===",
        fmt_df(pivot_exh, True),
        "",
        "=== 真正变盘路径：绝对ROI最高的15条（n≥150）===",
        top_move.round(4).to_string(index=False),
        "",
        "=== 变盘路径 × 近10场：绝对ROI最高的15条（n≥150）===",
        top_cross.round(4).to_string(index=False),
        "",
        "=== 所有路径里绝对ROI最差的（含不变盘）===",
        worst.round(4).to_string(index=False),
    ]
    if not alpha.empty:
        lines += ["", "=== 通过门槛 ===", alpha.round(4).to_string(index=False)]

    text = "\n".join(lines)
    (OUT / "ah_move_x_form.txt").write_text(text, encoding="utf-8")
    meta = {
        "n": int(len(d)),
        "n_form": int(len(form)),
        "n_moved": int(len(moved)),
        "n_alpha": int(len(alpha)),
        "home_roi": float(d.pnl_home.mean()),
        "away_roi": float(d.pnl_away.mean()),
        "pivot_home": pivot_home.where(pivot_home.notna(), None).to_dict(),
        "pivot_away": pivot_away.where(pivot_away.notna(), None).to_dict(),
        "pivot_n": pivot_n.fillna(0).to_dict(),
        "pivot_exh": pivot_exh.where(pivot_exh.notna(), None).to_dict(),
    }
    (PROC / "ah_move_x_form_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()
