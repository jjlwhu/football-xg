# 近 5 场 xG 与收盘亚盘（2025-26 五大联赛）

口径：

- **xG**：Understat 场级 xG  
- **盘口**：football-data.co.uk 收盘亚盘 `AHCh`（主队让球；主让为负）  
- **样本**：双方赛前都满 5 场联赛后，取赛季最晚 **100** 场  

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

只重跑分析（数据已在 `data/raw/`）：

```bash
python -c "from src.analyze_ah_xg import main; main()"
```

结果：

- `data/processed/sample_100.csv` 样本明细  
- `output/summary.txt` 相关与分组  
- `output/ah_xg_scatter.png` 散点图  

2025-26 赛季末约 100 场（2026-05-10 ~ 05-24）结论摘要：

- **让多少**：近 5 场 xG 隐含净胜 `pred_gd` 与收盘 `home_let` Pearson **0.61**，R² **0.37**（p≈2.6e-11）。回归截距约 **0.41**，接近常见主场让球。  
- **上盘打出**：扣掉球头后的 `fav_edge` 与上盘 cover **无显著相关**（r≈-0.04，p≈0.69）。这 100 场收盘盘已基本消化近 5 场 xG。
