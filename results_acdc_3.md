# ACDC 3 Labeled Results

Experiment setting:

- Dataset: ACDC
- Split: 3 labeled
- UniMatch baseline: 10000 iterations, batch size 8
- SAMatch and gate variants: 10000 iterations, batch size 2
- Model: `unet_drop`
- Metrics: Dice, HD95, ASD

## Main Results

| Method | RV Dice | RV HD95 | RV ASD | MYO Dice | MYO HD95 | MYO ASD | LV Dice | LV HD95 | LV ASD | Mean Dice | Mean HD95 | Mean ASD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniMatch | 0.8525 | 3.777 | 1.123 | 0.8109 | 3.746 | 0.901 | 0.8917 | 4.365 | 1.193 | 0.8517 | 3.962 | 1.072 |
| SAMatch | **0.8760** | **2.072** | **0.549** | **0.8448** | **1.437** | **0.443** | 0.9044 | **1.314** | **0.321** | **0.8751** | **1.608** | **0.438** |
| Class-wise Loose Gate | 0.8685 | 5.618 | 1.259 | 0.8447 | 2.297 | 0.910 | **0.9120** | 3.245 | 0.761 | 0.8750 | 3.720 | 0.977 |
| Sample-level Loose Gate | 0.8151 | 8.561 | 2.378 | 0.8173 | 3.575 | 0.975 | 0.9026 | 4.311 | 1.225 | 0.8450 | 5.483 | 1.526 |

## Gate Settings

| Method | Gate Mode | IoU Threshold | Area Ratio Min | Area Ratio Max | Min Area |
|---|---|---:|---:|---:|---:|
| Class-wise Loose Gate | class | 0.3 | 0.2 | 5.0 | 10 |
| Sample-level Loose Gate | sample | 0.3 | 0.2 | 5.0 | 10 |

## Summary

Compared with SAMatch:

- Class-wise Loose Gate keeps Mean Dice nearly unchanged: `0.8751 -> 0.8750`
- Class-wise Loose Gate worsens Mean HD95: `1.608 -> 3.720`
- Class-wise Loose Gate worsens Mean ASD: `0.438 -> 0.977`
- Sample-level Loose Gate clearly underperforms at 3 labeled: `0.8751 -> 0.8450` Mean Dice.

Current observation:

- With more labeled data, SAMatch already provides reliable refined pseudo labels.
- The reliability gate is most useful in the extremely low-label setting and is not consistently beneficial at 3 labeled.
