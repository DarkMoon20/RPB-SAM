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
| No SAM Refinement | 0.5342 | 55.394 | 20.101 | 0.6495 | 4.724 | 1.562 | 0.7347 | 5.400 | 1.390 | 0.6395 | 21.839 | 7.684 |
| SAMatch | 0.7887 | 10.078 | 2.455 | 0.7697 | 2.038 | 1.002 | 0.8860 | 1.661 | 0.482 | 0.8148 | 4.592 | 1.313 |
| SAMatch + Gate v1 | 0.7666 | 10.513 | 2.421 | 0.7501 | 2.174 | 0.803 | 0.8804 | 3.190 | 1.047 | 0.7990 | 5.292 | 1.424 |
| SAMatch + Medium Gate | 0.7928 | 7.261 | 1.998 | 0.7821 | 2.803 | 1.304 | 0.8888 | 3.945 | 1.109 | 0.8212 | 4.670 | 1.470 |
| SAMatch + Class-wise Loose Gate | 0.7971 | 6.387 | 1.799 | 0.8149 | 1.988 | 0.897 | 0.9040 | 1.446 | 0.610 | 0.8387 | 3.274 | 1.102 |
| Sample-level Strict Gate | 0.7786 | 6.253 | 1.784 | 0.7303 | 3.343 | 1.183 | 0.8832 | 3.907 | 1.360 | 0.7974 | 4.501 | 1.442 |
| Sample-level Medium Gate | **0.8007** | 8.468 | 2.202 | 0.8066 | 3.121 | 1.160 | 0.8983 | 4.215 | 1.081 | 0.8352 | 5.268 | 1.481 |
| **Ours: Sample-level Loose Gate** | **0.8004** | **5.789** | **1.737** | **0.8140** | **1.667** | **0.807** | **0.9025** | 1.613 | **0.469** | **0.8390** | **3.023** | **1.004** |

## Gate Settings

| Method | IoU Threshold | Area Ratio Min | Area Ratio Max | Min Area | Notes |
|---|---:|---:|---:|---:|---|
| Gate v1 | 0.5 | 0.5 | 2.0 | 10 | Strict class-wise reliability gate |
| Medium Gate | 0.4 | 0.3 | 3.0 | 10 | Medium class-wise reliability gate |
| Class-wise Loose Gate | 0.3 | 0.2 | 5.0 | 10 | Looser class-wise reliability gate |
| Sample-level Strict Gate | 0.5 | 0.5 | 2.0 | 10 | Strict whole-sample fallback gate |
| Sample-level Medium Gate | 0.4 | 0.3 | 3.0 | 10 | Medium whole-sample fallback gate |
| Sample-level Loose Gate | 0.3 | 0.2 | 5.0 | 10 | Whole-sample fallback gate; selected as Ours for 0_5 labeled |

## Summary

Compared with SAMatch, Ours improves:

- Mean Dice: `0.8148 -> 0.8390` (+0.0242)
- Mean HD95: `4.592 -> 3.023` (-1.569)
- Mean ASD: `1.313 -> 1.004` (-0.309)

Compared with UniMatch, Ours improves:

- Mean Dice: `0.7715 -> 0.8390` (+0.0675)
- Mean HD95: `10.584 -> 3.023` (-7.561)
- Mean ASD: `2.609 -> 1.004` (-1.605)

No SAM refinement ablation:

- No SAM Refinement performs much worse than UniMatch and SAMatch: `0.6395` Mean Dice and `21.839` Mean HD95.
- This shows the gain is not from simply continuing training in the SAMatch framework.
- MedSAM refinement is necessary, and the reliability gate further improves its use in the 0_5 labeled setting.

Sample-level gate ablation:

- Strict gate is too conservative and drops Mean Dice to `0.7974`.
- Medium gate recovers Dice to `0.8352`, but HD95 and ASD remain worse than loose gate.
- Loose gate gives the best overall balance: `0.8390` Dice, `3.023` HD95, and `1.004` ASD.

Sample-level gate accept ratio:

| Gate Setting | Avg Accept | Reject Steps | Reject Ratio |
|---|---:|---:|---:|
| Strict | 0.609 | 81/128 | 0.633 |
| Medium | 0.812 | 41/128 | 0.320 |
| Loose | 0.883 | 26/128 | 0.203 |

The loose gate still filters unreliable MedSAM refinements, but it avoids the over-filtering behavior of the strict setting.
