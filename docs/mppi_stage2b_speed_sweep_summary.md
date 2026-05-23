# MPPI Stage 2B Speed Sweep Summary

## 1. Purpose

本轮不是单纯追求最高成功率，而是判断 speed=0.75 是否存在无意义随机游走，
并寻找更干净的速度区间。扫描 3 temperatures × 5 speeds × 8 core8 模板 = 120 runs。

## 2. Best speed

- **Highest success rate**: speed=0.3 (83.3%)
- **Best path efficiency**: speed=0.1 (progress_eff_ee=0.1052)
- **Best composite score**: speed=0.3 (score=85.31)

- **Can lower speeds replace 0.75?**
  - speed=0.1: succ=33.3% (vs 58.3%), wasted_capped=18.5 (vs 60.0)
  - speed=0.2: succ=50.0% (vs 58.3%), wasted_capped=29.8 (vs 60.0)
  - speed=0.3: succ=83.3% (vs 58.3%), wasted_capped=27.5 (vs 60.0) ✅ higher or equal success, cleaner
  - speed=0.5: succ=75.0% (vs 58.3%), wasted_capped=38.1 (vs 60.0) ✅ higher or equal success, cleaner

### Speed comparison table

| Speed | Runs | 2mm10° | 5mm10° | FinalDist | EE_Path | WasteRatio | WasteCap | RW% | IneffSucc% | ProgEff | Runtime |
|-------|------|--------|--------|-----------|---------|-------------|----------|-----|------------|---------|--------|
| 0.1 | 24 | 33.3% | 33.3% | 0.1074 | 3.0595 | 18.4771 | 18.4771% | 0.0% | 0.0 | 0.1052s |
| 0.2 | 24 | 50.0% | 50.0% | 0.0726 | 5.3963 | 29.7785 | 29.7785% | 0.0% | 4.2 | 0.0827s |
| 0.3 | 24 | 83.3% | 83.3% | 0.0322 | 5.6485 | 30.2747 | 27.5456% | 0.0% | 16.7 | 0.0841s |
| 0.5 | 24 | 75.0% | 75.0% | 0.0395 | 9.3601 | 49.1123 | 38.0714% | 0.0% | 20.8 | 0.0737s |
| 0.75 | 24 | 58.3% | 58.3% | 0.0545 | 17.4964 | 90.096 | 60.017% | 0.0% | 33.3 | 0.0269s |

## 3. Speed × temperature interaction

### Temperature summary

| T | Runs | 2mm10° | 5mm10° | Reach10mm10° | Regress% | FinalDist | θErr | RW% | WasteRatio |
|---|------|--------|--------|-------------|----------|-----------|------|-----|------------|
| 0.1 | 40 | 65.0% | 65.0% | 67.5% | 0.0% | 0.052 | 8.6152 | 0.0% | 40.6276 |
| 0.2 | 40 | 57.5% | 57.5% | 57.5% | 0.0% | 0.069 | 11.9437 | 0.0% | 44.811 |
| 0.3 | 40 | 57.5% | 57.5% | 62.5% | 0.0% | 0.0628 | 11.7237 | 0.0% | 45.2045 |

### Speed × Temperature interaction table

| Speed \ T | T=0.1 | T=0.2 | T=0.3 |
|---|---|---|---|
| speed=0.1 | 37.5% | 37.5% | 25.0% |
| speed=0.2 | 50.0% | 37.5% | 62.5% |
| speed=0.3 | 100.0% | 75.0% | 75.0% |
| speed=0.5 | 62.5% | 87.5% | 75.0% |
| speed=0.75 | 75.0% | 50.0% | 50.0% |

**Sweet spots**: speed-temperature combinations where success ≥ average AND wasted_motion_ratio ≤ average.
- speed=0.3, T=0.1: succ=100.0%, waste_capped=14.5, score=157.3
- speed=0.5, T=0.2: succ=87.5%, waste_capped=29.4, score=86.9
- speed=0.2, T=0.3: succ=62.5%, waste_capped=22.7, score=60.5
- speed=0.3, T=0.2: succ=75.0%, waste_capped=31.5, score=59.8

## 4. Random-walk diagnostics

### speed=0.75 analysis

- ee_path_length_m: 17.4964
- wasted_motion_ratio: 90.096
- wasted_motion_ratio_capped: 60.017
- random_walk_rate: 0.0%
- inefficient_success_rate: 33.3%

- speed=0.1: EE_path=3.059m, waste=18.5, RW=0.0%, IneffSucc=0.0%
- speed=0.2: EE_path=5.396m, waste=29.8, RW=0.0%, IneffSucc=4.2%
- speed=0.3: EE_path=5.649m, waste=30.3, RW=0.0%, IneffSucc=16.7%
- speed=0.5: EE_path=9.360m, waste=49.1, RW=0.0%, IneffSucc=20.8%
- speed=0.75: EE_path=17.496m, waste=90.1, RW=0.0%, IneffSucc=33.3%

- speed=0.75 vs speed=0.1: EE_path=5.7× higher, waste=4.9× higher
- speed=0.75 vs speed=0.2: EE_path=3.2× higher, waste=3.0× higher
- speed=0.75 vs speed=0.3: EE_path=3.1× higher, waste=3.0× higher
- speed=0.75 vs speed=0.5: EE_path=1.9× higher, waste=1.8× higher

## 5. Family-wise behavior

| Family | Runs | 2mm10° | 5mm10° | Reach10mm10° | EE_Path | Waste | RW% | FinalDist | Top Failure |
|--------|------|--------|--------|-------------|---------|-------|-----|-----------|------------|
| blocking_hard | 30 | 66.7% | 66.7% | 70.0% | 6.3139 | 25.0381 | 0.0% | 0.0582 | not_reached |
| passage_bypass_medium | 15 | 100.0% | 100.0% | 100.0% | 3.6284 | 12.1486 | 0.0% | 0.0013 | ? |
| passage_bypass_narrow | 15 | 86.7% | 86.7% | 93.3% | 6.9392 | 25.585 | 0.0% | 0.0074 | not_reached |
| passage_bypass_wide | 15 | 100.0% | 100.0% | 100.0% | 3.3869 | 11.3394 | 0.0% | 0.0013 | ? |
| passage_direct_narrow | 45 | 20.0% | 20.0% | 22.2% | 12.985 | 83.0776 | 0.0% | 0.1211 | not_reached |

### blocking_hard: speed breakdown

| Speed | Runs | 2mm10° | EE_Path | Waste | ProgEff |
|-------|------|--------|---------|-------|---------|
| 0.1 | 6 | 0.0% | 3.877 | 29.6295 | 0.0343 |
| 0.2 | 6 | 50.0% | 5.0315 | 29.0937 | 0.1159 |
| 0.3 | 6 | 100.0% | 3.4441 | 10.2199 | 0.1317 |
| 0.5 | 6 | 100.0% | 4.0827 | 12.3167 | 0.1362 |
| 0.75 | 6 | 83.3% | 15.1342 | 43.9305 | 0.0353 |

### passage_bypass_medium: speed breakdown

| Speed | Runs | 2mm10° | EE_Path | Waste | ProgEff |
|-------|------|--------|---------|-------|---------|
| 0.1 | 3 | 100.0% | 1.2684 | 4.2507 | 0.2568 |
| 0.2 | 3 | 100.0% | 2.4139 | 8.0729 | 0.124 |
| 0.3 | 3 | 100.0% | 3.4026 | 11.402 | 0.0954 |
| 0.5 | 3 | 100.0% | 4.7029 | 15.7243 | 0.0747 |
| 0.75 | 3 | 100.0% | 6.3543 | 21.2933 | 0.0512 |

### passage_bypass_narrow: speed breakdown

| Speed | Runs | 2mm10° | EE_Path | Waste | ProgEff |
|-------|------|--------|---------|-------|---------|
| 0.1 | 3 | 66.7% | 2.923 | 9.9068 | 0.1092 |
| 0.2 | 3 | 100.0% | 5.0083 | 16.7809 | 0.0663 |
| 0.3 | 3 | 100.0% | 3.7485 | 12.5619 | 0.08 |
| 0.5 | 3 | 100.0% | 5.9009 | 19.7787 | 0.0511 |
| 0.75 | 3 | 66.7% | 17.1151 | 68.8967 | 0.0217 |

### passage_bypass_wide: speed breakdown

| Speed | Runs | 2mm10° | EE_Path | Waste | ProgEff |
|-------|------|--------|---------|-------|---------|
| 0.1 | 3 | 100.0% | 1.1153 | 3.7383 | 0.2759 |
| 0.2 | 3 | 100.0% | 1.8189 | 6.0837 | 0.1751 |
| 0.3 | 3 | 100.0% | 3.1285 | 10.453 | 0.115 |
| 0.5 | 3 | 100.0% | 4.1688 | 13.9517 | 0.085 |
| 0.75 | 3 | 100.0% | 6.7031 | 22.4701 | 0.0512 |

### passage_direct_narrow: speed breakdown

| Speed | Runs | 2mm10° | EE_Path | Waste | ProgEff |
|-------|------|--------|---------|-------|---------|
| 0.1 | 9 | 0.0% | 3.8051 | 23.5539 | 0.0437 |
| 0.2 | 9 | 0.0% | 7.9556 | 49.7011 | 0.0215 |
| 0.3 | 9 | 55.6% | 9.34 | 62.4468 | 0.0397 |
| 0.5 | 9 | 33.3% | 17.3144 | 106.2702 | 0.0356 |
| 0.75 | 9 | 11.1% | 26.51 | 173.4158 | 0.0069 |

## 6. Training data quality

- **Cleanest trajectories**: speed=0.1 (prog_eff=0.1052, RW=0.0%)
- **speed=0.75 caution**: wasted_motion_ratio=90.1, RW=0.0% — many trajectories contain excessive wandering.
- **Recommendation**: avoid speed=0.75 as primary training data. Use it as aggressive oracle / failure-rich augmentation instead.
- **Preferred speeds for learned dynamics**: lower speeds with high progress_efficiency_ee and low random_walk_rate.

## 7. Recommendation for Stage 2C

1. speed=0.3, T=0.1: succ=100.0%, waste=14.5, score=157.3
2. speed=0.5, T=0.2: succ=87.5%, waste=36.9, score=86.9
3. speed=0.2, T=0.3: succ=62.5%, waste=22.7, score=60.5

Stage 2C will scan num_samples=[1024,2048] and init_std=[0.5,0.7,1.0] around these top speed-temperature configurations.
**Do NOT auto-run Stage 2C. Await user decision.**

## Appendix: Top 10 speed-temperature combinations by score

| Rank | Speed | T | 2mm10° | 5mm10° | Reach10mm10° | RW% | IneffSucc% | Waste | ProgEff | Score |
|------|-------|---|--------|--------|-------------|-----|------------|-------|---------|-------|
| 1 | 0.3 | 0.1 | 100.0% | 100.0% | 100.0% | 0.0% | 25.0% | 14.5268 | 0.0923 | 157.3 |
| 2 | 0.5 | 0.2 | 87.5% | 87.5% | 87.5% | 0.0% | 25.0% | 36.9127 | 0.0855 | 86.9 |
| 3 | 0.2 | 0.3 | 62.5% | 62.5% | 62.5% | 0.0% | 0.0% | 22.7033 | 0.107 | 60.5 |
| 4 | 0.3 | 0.2 | 75.0% | 75.0% | 75.0% | 0.0% | 0.0% | 36.8028 | 0.1022 | 59.8 |
| 5 | 0.3 | 0.3 | 75.0% | 75.0% | 75.0% | 0.0% | 25.0% | 39.4943 | 0.0578 | 39.0 |
| 6 | 0.5 | 0.3 | 75.0% | 75.0% | 75.0% | 0.0% | 12.5% | 54.4917 | 0.0705 | 37.1 |
| 7 | 0.1 | 0.1 | 37.5% | 37.5% | 37.5% | 0.0% | 0.0% | 17.8054 | 0.1133 | 22.3 |
| 8 | 0.1 | 0.2 | 37.5% | 37.5% | 37.5% | 0.0% | 0.0% | 18.469 | 0.1028 | 20.2 |
| 9 | 0.2 | 0.1 | 50.0% | 50.0% | 62.5% | 0.0% | 12.5% | 29.7927 | 0.0723 | 13.8 |
| 10 | 0.1 | 0.3 | 25.0% | 25.0% | 37.5% | 0.0% | 0.0% | 19.1567 | 0.0994 | -3.2 |


*Raw data: runs/mppi_stage2b_speed_20260520_005142/manifest.csv*

*Generated: MPPI Stage 2B Speed Sweep*