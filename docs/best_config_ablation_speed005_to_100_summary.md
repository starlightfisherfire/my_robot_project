# Best Config Ablation: 0.05 to 1.0 m/s

## 1. High-level verdict

**Best overall config:** s0.50_h120 (speed=0.50 m/s, horizon=120)

**Top 3 configs:**
| Rank | Config | Speed | Horizon | Success Rate | Hard Success | Score |
|------|--------|-------|---------|--------------|--------------|-------|
| 1 | s0.50_h120 | 0.50 m/s (50 cm/s) | 120 | 87.5% | 4 | 126.9 |
| 2 | s0.20_h160 | 0.20 m/s (20 cm/s) | 160 | 87.5% | 4 | 119.4 |
| 3 | s0.20_h80 | 0.20 m/s (20 cm/s) | 80 | 75.0% | 3 | 100.5 |

## 2. Unit clarification

- 0.05 m/s = 5 cm/s
- 0.10 m/s = 10 cm/s
- 0.20 m/s = 20 cm/s
- 0.35 m/s = 35 cm/s
- 0.50 m/s = 50 cm/s
- 0.75 m/s = 75 cm/s
- 1.00 m/s = 100 cm/s

## 3. Speed analysis

| Speed (m/s) | Speed (cm/s) | Success Rate | Hard Success |
|-------------|--------------|--------------|--------------|
| 0.20 | 20 | 62.5% | 9 |
| 0.50 | 50 | 72.7% | 8 |

## 4. Horizon analysis

| Horizon | Success Rate | Hard Success |
|---------|--------------|--------------|
| 80 | 68.8% | 5 |
| 120 | 68.8% | 5 |
| 140 | 50.0% | 3 |
| 160 | 87.5% | 4 |

## 5. Template analysis

| Label | Split | Success Rate |
|-------|-------|--------------|
| ph07 |  | 0.0% |
| ph04 |  | 33.3% |
| bh00 |  | 57.1% |
| bh01 |  | 57.1% |
| pm06 |  | 71.4% |
| bm08 |  | 100.0% |
| bh09 |  | 100.0% |
| pm02 |  | 100.0% |

## 6. Recommendation

Based on this ablation:
- **Best main config:** s0.50_h120
- **Conservative config:** s0.20_h140 (lowest speed, highest horizon)
- **Aggressive config:** s0.20_h160

## 7. Data files

- Manifest: `runs/best_config_ablation_speed005_to_100_20260515_181130/manifest.csv`
- Videos: `runs/best_config_ablation_speed005_to_100_20260515_181130/videos/`
- Logs: `runs/best_config_ablation_speed005_to_100_20260515_181130/logs/`
