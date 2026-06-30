# ACDC 0_5 Results

Experiment setting:

- Dataset: ACDC
- Split: 0_5 labeled
- UniMatch baseline: 10000 iterations, batch size 8
- SAMatch and gate variants: 10000 iterations, batch size 2
- Model: `unet_drop`
- Metrics: Dice, HD95, ASD

## Main Results

| Method | RV Dice | RV HD95 | RV ASD | MYO Dice | MYO HD95 | MYO ASD | LV Dice | LV HD95 | LV ASD | Mean Dice | Mean HD95 | Mean ASD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniMatch | 0.7288 | 21.636 | 5.188 | 0.7526 | 5.951 | 1.576 | 0.8331 | 4.165 | 1.063 | 0.7715 | 10.584 | 2.609 |
| SAMatch | 0.7887 | 10.078 | 2.455 | 0.7697 | 2.038 | 1.002 | 0.8860 | 1.661 | 0.482 | 0.8148 | 4.592 | 1.313 |
| SAMatch + Gate v1 | 0.7666 | 10.513 | 2.421 | 0.7501 | 2.174 | 0.803 | 0.8804 | 3.190 | 1.047 | 0.7990 | 5.292 | 1.424 |
| SAMatch + Loose Gate | 0.7971 | 6.387 | 1.799 | 0.8149 | 1.988 | 0.897 | 0.9040 | 1.446 | 0.610 | 0.8387 | 3.274 | 1.102 |

## Gate Settings

| Method | IoU Threshold | Area Ratio Min | Area Ratio Max | Min Area | Notes |
|---|---:|---:|---:|---:|---|
| Gate v1 | 0.5 | 0.5 | 2.0 | 10 | Strict class-wise reliability gate |
| Loose Gate | 0.3 | 0.2 | 5.0 | 10 | Looser class-wise reliability gate |

## Summary

Compared with SAMatch, Loose Gate improves:

- Mean Dice: `0.8148 -> 0.8387` (+0.0239)
- Mean HD95: `4.592 -> 3.274` (-1.318)
- Mean ASD: `1.313 -> 1.102` (-0.211)

Compared with UniMatch, Loose Gate improves:

- Mean Dice: `0.7715 -> 0.8387` (+0.0672)
- Mean HD95: `10.584 -> 3.274` (-7.310)
- Mean ASD: `2.609 -> 1.102` (-1.507)

