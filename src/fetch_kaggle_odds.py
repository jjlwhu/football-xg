"""下载带真实亚盘球头、真实大小球盘口（2 / 2.25 / 2.75 / 3 …）的收盘主盘。

来源：Kaggle「Football Odds Movements 2018-2026, 4 Bookmakers」
https://www.kaggle.com/datasets/archangel0x01/football-odds-movements-2018-2026

只要 match_closing + matches（约几十 MB）。不要下 odds_ah / odds_ou 那 3GB 逐笔流水。

需要本机有 Kaggle API token：
  1. 打开 https://www.kaggle.com/settings
  2. Create New Token，把 kaggle.json 放到 ~/.kaggle/kaggle.json
  3. 本机跑：python -m src.fetch_kaggle_odds
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "kaggle_odds"
FILES = ("match_closing.parquet", "matches.parquet")
DATASET = "archangel0x01/football-odds-movements-2018-2026"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "未安装 kagglehub。先：.venv/bin/pip install kagglehub\n"
            "再确认 ~/.kaggle/kaggle.json 存在。"
        ) from exc

    for name in FILES:
        target = DEST / name
        if target.exists():
            print(f"已有 {target}")
            continue
        print(f"下载 {DATASET} / {name} …")
        cached = Path(kagglehub.dataset_download(DATASET, path=name))
        if cached.is_file():
            target.write_bytes(cached.read_bytes())
        else:
            found = next(cached.rglob(name), None)
            if found is None:
                raise SystemExit(f"下载后找不到 {name}，实际路径: {cached}")
            target.write_bytes(found.read_bytes())
        print(f"-> {target} ({target.stat().st_size / 1e6:.1f} MB)")
    print("完成。下一步：python -m src.analyze_ah_ou_lines")


if __name__ == "__main__":
    main()
