#!/usr/bin/env python3
"""先下载 2025-26 数据，再分析近 5 场 xG 与收盘亚盘。"""

from src.fetch_data import main as fetch
from src.analyze_ah_xg import main as analyze


if __name__ == "__main__":
    fetch()
    analyze()
