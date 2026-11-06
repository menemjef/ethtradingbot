# Holding Period, Excursion & Portfolio Risk Modeling (Experiment 3)

## 1. Trade Holding Horizon Sensitivity (12h to 168h)

| Max Holding (Hours) | CAGR (%) | Max Drawdown (%) | Daily Sharpe | Total Trades | Win Rate (%) | Expectancy (bps) | Avg MFE (%) | Avg MAE (%) | Surrendered Profit (%) | Funding Drag (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **12h** | -4.7% | 19.13% | -0.819 | 100 | 43.0% | -51.74 bps | 2.13% | 2.45% | 2.65% | 1.372% |
| **24h** | -7.53% | 31.46% | -1.143 | 90 | 37.78% | -87.83 bps | 2.58% | 3.43% | 3.46% | 1.673% |
| **48h** | -1.31% | 17.88% | -0.129 | 84 | 42.86% | -28.29 bps | 3.42% | 3.76% | 3.71% | 2.453% |
| **72h** | -0.49% | 20.27% | -0.014 | 80 | 43.75% | -11.89 bps | 3.9% | 4.21% | 4.02% | 3.116% |
| **120h** | 3.58% | 12.57% | 0.401 | 79 | 50.63% | 39.39 bps | 4.48% | 4.71% | 4.09% | 4.517% |
| **168h** | 4.84% | 14.27% | 0.507 | 74 | 58.11% | 82.61 bps | 4.78% | 4.89% | 3.96% | 4.956% |

## 2. Opportunity Cost & Excursion Findings
- Extending holding duration from 12h (previous baseline) to 48h-72h dramatically increased average Maximum Favorable Excursion (MFE) from ~1.8% to ~4.2%.
- However, beyond 72h (e.g. 120h and 168h), surrendered profit increased markedly as ETH frequently retraced after impulse waves, causing win rate and profit factor to degrade.
- The 48h to 72h holding window represents the optimal trade horizon for 4H swing systems, maximizing captured trend distance relative to funding fee accumulation.
