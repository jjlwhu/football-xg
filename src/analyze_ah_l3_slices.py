"""近 3 场 xG × 收盘亚盘类型：扫描何种条件下上/下盘打出率更高。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .analyze_ah_xg import (
    HOME_ADV,
    OUT,
    PROC,
    RAW,
    SAMPLE_END,
    SAMPLE_START,
    attach_deep,
    build_merged,
    settle_ah_home,
    wilson_ci,
)
from .team_map import SECOND_DIVS, SEASONS

MIN_N = 40
LINE_ORDER = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def line_label(abs_line: float) -> str:
    m = {
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
    key = round(float(abs_line) * 4) / 4
    return m.get(key, f"|h|={key}")


def pnl(cover, odds) -> np.ndarray:
    cover = np.asarray(cover, dtype=float)
    odds = np.asarray(odds, dtype=float)
    return np.where(cover > 0, cover * (odds - 1.0), cover)


def decorate(d: pd.DataFrame) -> pd.DataFrame:
    d = attach_deep(d, "pred_gd_l3", home_adv=HOME_ADV, split="p40")
    d["line_lab"] = d["abs_line"].map(line_label)
    d["fav_odds"] = np.where(d["favorite"] == "home", d["ah_home_odds"], d["ah_away_odds"])
    d["dog_odds"] = np.where(d["favorite"] == "home", d["ah_away_odds"], d["ah_home_odds"])
    d["pnl_up"] = pnl(d["favorite_cover"], d["fav_odds"])
    d["pnl_dn"] = pnl(-d["favorite_cover"], d["dog_odds"])
    d["上盘低水"] = d["fav_odds"] < d["dog_odds"]
    med = d["deep"].median()
    d["深浅对半"] = np.where(d["deep"] <= med, "开浅", "开深")
    return d


def prepare(window: str = "2526_mid") -> pd.DataFrame:
    m = build_merged()
    d = m.dropna(subset=["pred_gd_l3", "favorite_cover", "line", "ah_home_odds", "ah_away_odds"]).copy()
    d = d[d["favorite"] != "pickem"]
    if window == "2526_mid":
        d = d[(d["date"] >= SAMPLE_START) & (d["date"] < SAMPLE_END)]
    elif window == "complete":
        d = d[d["season_path"] != "2627"]
    elif window == "current":
        d = d[d["season_path"] == "2627"]
    return decorate(d)


def metrics(name: str, sub: pd.DataFrame, min_n: int = MIN_N) -> dict | None:
    n = len(sub)
    if n < min_n:
        return None
    up = int((sub["favorite_cover"] > 0).sum())
    dn = int((sub["favorite_cover"] < 0).sum())
    push = n - up - dn
    p_up, p_dn = up / n, dn / n
    ci_up, ci_dn = wilson_ci(up, n), wilson_ci(dn, n)
    if p_up >= p_dn:
        side, p, lo, hi = "上盘", p_up, ci_up[0], ci_up[1]
    else:
        side, p, lo, hi = "下盘", p_dn, ci_dn[0], ci_dn[1]
    return {
        "切片": name,
        "n": n,
        "上盘打出": p_up,
        "下盘打出": p_dn,
        "走水": push / n,
        "上盘CI低": ci_up[0],
        "上盘CI高": ci_up[1],
        "下盘CI低": ci_dn[0],
        "下盘CI高": ci_dn[1],
        "方向": side,
        "点估计": p,
        "CI低": lo,
        "CI高": hi,
        "均cover": float(sub["favorite_cover"].mean()),
        "上盘ROI": float(sub["pnl_up"].mean()),
        "下盘ROI": float(sub["pnl_dn"].mean()),
        "≥55%": bool(p >= 0.55),
        "CI低>50%": bool(lo > 0.5),
        "赢半占比": float((sub["favorite_cover"] == 0.5).mean()),
        "全赢占比": float((sub["favorite_cover"] == 1).mean()),
        "全输占比": float((sub["favorite_cover"] == -1).mean()),
    }


def scan(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []

    def add(name: str, sub: pd.DataFrame, min_n: int = MIN_N) -> None:
        r = metrics(name, sub, min_n=min_n)
        if r:
            rows.append(r)

    add("中段全样本", d)
    add("主队上盘", d[d["favorite"] == "home"])
    add("客队上盘", d[d["favorite"] == "away"])
    add("上盘收盘低水", d[d["上盘低水"]])
    add("上盘收盘高水", d[~d["上盘低水"]])

    for lab in ("开浅", "接近", "开深"):
        add(f"近3场xG相对盘口 {lab}", d[d["深浅"] == lab])
    for lab in ("开浅", "开深"):
        add(f"近3场xG D中位数{lab}", d[d["深浅对半"] == lab])

    for q, lab in [(0.33, "最低1/3"), (0.25, "最低25%"), (0.2, "最低20%")]:
        lo, hi = d["deep"].quantile(q), d["deep"].quantile(1 - q)
        add(f"D最浅{lab}（xG看盘开浅）", d[d["deep"] <= lo], min_n=40)
        add(f"D最深对称尾{lab.replace('最低', '')}（xG看盘开深）", d[d["deep"] >= hi], min_n=40)

    for thr in (0.25, 0.4, 0.6, 0.8):
        add(f"开浅 D<{ -thr}", d[d["deep"] < -thr], min_n=40)
        add(f"开深 D>{thr}", d[d["deep"] > thr], min_n=40)

    add("开浅且主队上盘", d[(d["深浅"] == "开浅") & (d["favorite"] == "home")], min_n=40)
    add("开浅且客队上盘", d[(d["深浅"] == "开浅") & (d["favorite"] == "away")], min_n=40)
    add("开深且主队上盘", d[(d["深浅"] == "开深") & (d["favorite"] == "home")], min_n=40)
    add("开深且客队上盘", d[(d["深浅"] == "开深") & (d["favorite"] == "away")], min_n=40)
    add("开浅且上盘低水", d[(d["深浅"] == "开浅") & d["上盘低水"]], min_n=40)
    add("开浅且上盘高水", d[(d["深浅"] == "开浅") & ~d["上盘低水"]], min_n=40)
    add("开深且上盘低水", d[(d["深浅"] == "开深") & d["上盘低水"]], min_n=40)
    add("开深且上盘高水", d[(d["深浅"] == "开深") & ~d["上盘低水"]], min_n=40)

    for lg, g in d.groupby("league"):
        add(f"联赛={lg}", g, min_n=40)

    line_rows = []
    cross_rows = []
    for line in LINE_ORDER:
        sub = d[np.isclose(d["abs_line"], line)]
        r = metrics(f"盘口={line_label(line)}", sub, min_n=30)
        if r:
            r["盘口"] = line_label(line)
            r["球头"] = line
            line_rows.append(r)
            add(f"盘口={line_label(line)}", sub, min_n=30)
            for side, mask in (("主上盘", sub["favorite"] == "home"), ("客上盘", sub["favorite"] == "away")):
                add(f"{line_label(line)} · {side}", sub[mask], min_n=30)
            for lab in ("开浅", "接近", "开深"):
                ss = sub[sub["深浅"] == lab]
                cr = metrics(f"{line_label(line)} · {lab}", ss, min_n=25)
                if cr:
                    cr["盘口"] = line_label(line)
                    cr["条件"] = lab
                    cross_rows.append(cr)
                    add(f"{line_label(line)} · 近3场xG{lab}", ss, min_n=25)
            for lab in ("开浅", "开深"):
                ss = sub[sub["深浅对半"] == lab]
                add(f"{line_label(line)} · D中位数{lab}", ss, min_n=25)

    slices = pd.DataFrame(rows).drop_duplicates(subset=["切片"])
    slices = slices.sort_values(["≥55%", "CI低>50%", "点估计", "n"], ascending=[False, False, False, False]).reset_index(drop=True)
    by_line = pd.DataFrame(line_rows).sort_values("球头") if line_rows else pd.DataFrame()
    cross = pd.DataFrame(cross_rows) if cross_rows else pd.DataFrame()
    return slices, by_line, cross


def focus_cross(d: pd.DataFrame, min_n: int = 40) -> pd.DataFrame:
    rows = []
    for line in (0.5, 0.75, 1.0, 1.5):
        sub = d[np.isclose(d["abs_line"], line)]
        for lab in ("开浅", "接近", "开深"):
            r = metrics(f"{line_label(line)} · {lab}", sub[sub["深浅"] == lab], min_n=min_n)
            if r:
                r["盘口"] = line_label(line)
                r["条件"] = lab
                rows.append(r)
        r = metrics(f"{line_label(line)} 无条件", sub, min_n=min_n)
        if r:
            r["盘口"] = line_label(line)
            r["条件"] = "全部"
            rows.append(r)
    return pd.DataFrame(rows)


def per_season_slice(d: pd.DataFrame, line: float, lab: str, min_n: int = 20) -> pd.DataFrame:
    rows = []
    sub0 = d[np.isclose(d["abs_line"], line)]
    if lab != "全部":
        sub0 = sub0[sub0["深浅"] == lab]
    for sp, g in sub0.groupby("season_path"):
        r = metrics(f"{sp}", g, min_n=min_n)
        if r:
            r["season_path"] = sp
            rows.append(r)
    return pd.DataFrame(rows)


def load_second_div() -> pd.DataFrame:
    frames = []
    for season_path, _y in SEASONS:
        for div, meta in SECOND_DIVS.items():
            path = RAW / f"fd_{div}_{season_path}.csv"
            if not path.exists():
                continue
            fd = pd.read_csv(path)
            if "AHCh" not in fd.columns:
                continue
            fd["div"] = div
            fd["league"] = meta["name"]
            fd["season_path"] = season_path
            frames.append(fd)
    fd = pd.concat(frames, ignore_index=True)
    fd["date"] = pd.to_datetime(fd["Date"], dayfirst=True)
    fd["AHCh"] = pd.to_numeric(fd["AHCh"], errors="coerce")
    fd["FTHG"] = pd.to_numeric(fd["FTHG"], errors="coerce")
    fd["FTAG"] = pd.to_numeric(fd["FTAG"], errors="coerce")
    for col in ("AvgCAHH", "AvgCAHA", "B365CAHH", "B365CAHA"):
        if col in fd.columns:
            fd[col] = pd.to_numeric(fd[col], errors="coerce")
        else:
            fd[col] = np.nan
    fd = fd.dropna(subset=["AHCh", "FTHG", "FTAG"]).copy()
    fd = fd[fd["season_path"] != "2627"]
    fd["gd"] = fd["FTHG"] - fd["FTAG"]
    fd["line"] = fd["AHCh"]
    fd["abs_line"] = fd["line"].abs()
    fd["home_cover"] = [settle_ah_home(gd, line) for gd, line in zip(fd["gd"], fd["line"])]
    fd["ah_home_odds"] = fd["AvgCAHH"].fillna(fd["B365CAHH"])
    fd["ah_away_odds"] = fd["AvgCAHA"].fillna(fd["B365CAHA"])
    fav = np.where(fd["line"] < 0, "home", np.where(fd["line"] > 0, "away", "pickem"))
    pickem = fav == "pickem"
    home_low = fd["ah_home_odds"] < fd["ah_away_odds"]
    away_low = fd["ah_away_odds"] < fd["ah_home_odds"]
    fav = np.where(pickem & home_low, "home", np.where(pickem & away_low, "away", fav))
    fd["favorite"] = fav
    fd = fd[fd["favorite"] != "pickem"].copy()
    fd["favorite_cover"] = np.where(
        fd["favorite"] == "home",
        fd["home_cover"],
        -fd["home_cover"],
    )
    fd["fav_odds"] = np.where(fd["favorite"] == "home", fd["ah_home_odds"], fd["ah_away_odds"])
    fd["dog_odds"] = np.where(fd["favorite"] == "home", fd["ah_away_odds"], fd["ah_home_odds"])
    fd = fd.dropna(subset=["fav_odds", "dog_odds", "favorite_cover"])
    fd["pnl_up"] = pnl(fd["favorite_cover"], fd["fav_odds"])
    fd["pnl_dn"] = pnl(-fd["favorite_cover"], fd["dog_odds"])
    fd["line_lab"] = fd["abs_line"].map(line_label)
    return fd


def fmt_row(r: dict | pd.Series) -> str:
    return (
        f"{r['切片']}: n={r['n']} → {r['方向']} {r['点估计']:.1%} "
        f"({r['CI低']:.1%}~{r['CI高']:.1%}) cover={r['均cover']:+.3f} "
        f"上ROI={r['上盘ROI']:+.1%} 下ROI={r['下盘ROI']:+.1%}"
        + ("  CI低>50%" if r["CI低>50%"] else "")
    )


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = prepare()
    slices, by_line, cross = scan(d)
    slices.to_csv(PROC / "ah_l3_slices.csv", index=False)
    by_line.to_csv(PROC / "ah_l3_by_line.csv", index=False)
    cross.to_csv(PROC / "ah_l3_line_x_deep.csv", index=False)

    payload = {
        "n": int(len(d)),
        "start": SAMPLE_START,
        "end": SAMPLE_END,
        "home_adv": HOME_ADV,
        "base_up": float((d.favorite_cover > 0).mean()),
        "base_dn": float((d.favorite_cover < 0).mean()),
        "base_push": float((d.favorite_cover == 0).mean()),
        "mean_cover": float(d.favorite_cover.mean()),
        "roi_up": float(d.pnl_up.mean()),
        "roi_dn": float(d.pnl_dn.mean()),
        "corr_deep_cover": float(d["deep"].corr(d["favorite_cover"])),
        "by_line": by_line.to_dict(orient="records"),
        "cross": cross.to_dict(orient="records"),
        "hot": slices[slices["≥55%"]].head(25).to_dict(orient="records"),
        "solid": slices[slices["CI低>50%"]].head(20).to_dict(orient="records"),
        "tercile": slices[slices["切片"].str.startswith("近3场xG相对盘口")].to_dict(orient="records"),
        "deep_half": slices[slices["切片"].str.startswith("近3场xG D中位数")].to_dict(orient="records"),
        "home_away": slices[slices["切片"].isin(["主队上盘", "客队上盘", "中段全样本"])].to_dict(orient="records"),
        "league": slices[slices["切片"].str.startswith("联赛=")].to_dict(orient="records"),
    }
    (PROC / "ah_l3_slices.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"近3场 xG vs 收盘亚盘类型（+主场{HOME_ADV}）",
        f"样本: {SAMPLE_START}～{SAMPLE_END} n={len(d)} 上盘打出 {payload['base_up']:.1%} / 下盘 {payload['base_dn']:.1%} / 走水 {payload['base_push']:.1%}",
        f"全样本均cover={payload['mean_cover']:+.3f} 上盘ROI={payload['roi_up']:+.1%} 下盘ROI={payload['roi_dn']:+.1%}",
        f"D_l3 vs 上盘cover Pearson={payload['corr_deep_cover']:.3f}",
        "",
        "=== 按盘口（打出率含赢半；均cover/ROI才是钱）===",
    ]
    if len(by_line):
        show = by_line[
            ["盘口", "n", "上盘打出", "下盘打出", "均cover", "上盘ROI", "下盘ROI", "赢半占比"]
        ].copy()
        lines.append(show.round(3).to_string(index=False))
    lines += ["", "=== 盘口 × 近3场xG深浅三分位 ==="]
    if len(cross):
        show = cross[["盘口", "条件", "n", "方向", "点估计", "上盘打出", "下盘打出", "均cover", "上盘ROI", "下盘ROI"]]
        lines.append(show.round(3).to_string(index=False))
    lines += ["", "=== 点估计 ≥55% ==="]
    hot = slices[slices["≥55%"]]
    for _, r in hot.iterrows():
        flag = "  CI低>50%" if r["CI低>50%"] else ""
        lines.append(
            f"{r['切片']}: n={r['n']} → {r['方向']} {r['点估计']:.1%} "
            f"({r['CI低']:.1%}~{r['CI高']:.1%}) cover={r['均cover']:+.3f} "
            f"上ROI={r['上盘ROI']:+.1%} 下ROI={r['下盘ROI']:+.1%}{flag}"
        )
    lines += ["", "=== CI下限>50% ==="]
    solid = slices[slices["CI低>50%"]]
    if solid.empty:
        lines.append("没有切片的 Wilson 下限超过 50%。")
    else:
        for _, r in solid.iterrows():
            lines.append(
                f"{r['切片']}: n={r['n']} → {r['方向']} {r['点估计']:.1%} ({r['CI低']:.1%}~{r['CI高']:.1%})"
            )
    lines.append(f"\n共扫描 {len(slices)} 条切片，≥55% 需警惕多重检验。")
    text = "\n".join(lines)
    (OUT / "ah_l3_slices.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nCSV: {PROC / 'ah_l3_slices.csv'}")

    complete = prepare("complete")
    current = prepare("current")
    focus = focus_cross(complete, min_n=80)
    focus.to_csv(PROC / "ah_l3_multi_focus.csv", index=False)
    by_season = per_season_slice(complete, 0.75, "开深", min_n=15)
    by_season_all = per_season_slice(complete, 0.75, "全部", min_n=20)

    extra = [
        "",
        "=" * 88,
        f"【多赛季复现】2019-20～2025-26 完整赛季 近3场xG满窗 n={len(complete)}",
        f"上盘 {(complete.favorite_cover>0).mean():.1%} 下盘 {(complete.favorite_cover<0).mean():.1%} "
        f"均cover={complete.favorite_cover.mean():+.3f} 上ROI={complete.pnl_up.mean():+.1%} 下ROI={complete.pnl_dn.mean():+.1%}",
        f"D vs cover Pearson={complete['deep'].corr(complete['favorite_cover']):.3f}",
        f"当前 2026-27 进行中满窗 n={len(current)}",
        "",
        "=== 半球 / 半一 / 一球 / 球半 × 深浅（完整赛季）===",
    ]
    if len(focus):
        extra.append(
            focus[["盘口", "条件", "n", "上盘打出", "下盘打出", "均cover", "上盘ROI", "下盘ROI", "CI低", "CI高"]]
            .round(3)
            .to_string(index=False)
        )
    extra += ["", "=== 半一 × 开深 分赛季 ==="]
    if len(by_season):
        extra.append(
            by_season[["season_path", "n", "上盘打出", "下盘打出", "均cover", "上盘ROI", "点估计", "CI低"]]
            .round(3)
            .to_string(index=False)
        )
    extra += ["", "=== 半一无条件 分赛季 ==="]
    if len(by_season_all):
        extra.append(
            by_season_all[["season_path", "n", "上盘打出", "下盘打出", "均cover", "上盘ROI", "CI低"]]
            .round(3)
            .to_string(index=False)
        )

    second = load_second_div()
    extra += [
        "",
        f"=== 二级联赛（英冠/西乙/德乙/意乙/法乙，无 xG，仅盘口结果）n={len(second)} ===",
    ]
    sec_rows = []
    for line in (0.5, 0.75, 1.0, 1.5):
        r = metrics(f"二级 {line_label(line)}", second[np.isclose(second.abs_line, line)], min_n=80)
        if r:
            sec_rows.append(r)
            extra.append(fmt_row(r))
    for lg, g in second.groupby("league"):
        r = metrics(f"二级 {lg} 半一", g[np.isclose(g.abs_line, 0.75)], min_n=40)
        if r:
            extra.append(fmt_row(r))
    if current is not None and len(current):
        extra += ["", "=== 2026-27 进行中：半一 ==="]
        r = metrics("2627 半一", current[np.isclose(current.abs_line, 0.75)], min_n=10)
        if r:
            extra.append(fmt_row(r))
        r = metrics("2627 半一开深", current[(np.isclose(current.abs_line, 0.75)) & (current["深浅"]=="开深")], min_n=8)
        if r:
            extra.append(fmt_row(r))

    payload_m = {
        "complete_n": int(len(complete)),
        "current_n": int(len(current)),
        "second_n": int(len(second)),
        "focus": focus.to_dict(orient="records") if len(focus) else [],
        "bani_deep_by_season": by_season.to_dict(orient="records") if len(by_season) else [],
        "bani_all_by_season": by_season_all.to_dict(orient="records") if len(by_season_all) else [],
        "second_lines": sec_rows,
        "complete_up": float((complete.favorite_cover > 0).mean()),
        "complete_cover": float(complete.favorite_cover.mean()),
        "complete_roi_up": float(complete.pnl_up.mean()),
        "corr": float(complete["deep"].corr(complete["favorite_cover"])),
    }
    (PROC / "ah_l3_multi.json").write_text(
        json.dumps(payload_m, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    extra_txt = "\n".join(extra)
    (OUT / "ah_l3_multi.txt").write_text(extra_txt + "\n", encoding="utf-8")
    print(extra_txt)


if __name__ == "__main__":
    main()
