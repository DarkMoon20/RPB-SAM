# ACDC 1 Labeled Results

Experiment setting:

- Dataset: ACDC
- Split: 1 labeled
- UniMatch baseline: 10000 iterations, batch size 8
- SAMatch and gate variants: 10000 iterations, batch size 2
- Model: `unet_drop`
- Metrics: Dice, HD95, ASD

## Main Results

| Method | RV Dice | RV HD95 | RV ASD | MYO Dice | MYO HD95 | MYO ASD | LV Dice | LV HD95 | LV ASD | Mean Dice | Mean HD95 | Mean ASD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniMatch | 0.8234 | 6.124 | 1.670 | 0.8323 | 3.728 | 1.178 | 0.8923 | 7.255 | 2.397 | 0.8493 | 5.702 | 1.748 |
| SAMatch | 0.8379 | 3.923 | 1.075 | 0.8466 | 7.051 | 1.641 | 0.9041 | 9.070 | 1.965 | 0.8629 | 6.681 | 1.560 |
| Sample-level Loose Gate | 0.8221 | 3.479 | 1.489 | 0.8376 | 8.646 | 1.872 | 0.8885 | 11.197 | 3.423 | 0.8494 | 7.774 | 2.261 |
| Class-wise Loose Gate | 0.8340 | 4.090 | 1.349 | 0.8380 | 5.250 | 1.407 | 0.8997 | 8.361 | 1.983 | 0.8572 | 5.901 | 1.580 |

## Gate Settings

| Method | Gate Mode | IoU Threshold | Area Ratio Min | Area Ratio Max | Min Area |
|---|---|---:|---:|---:|---:|
| Sample-level Loose Gate | sample | 0.3 | 0.2 | 5.0 | 10 |
| Class-wise Loose Gate | class | 0.3 | 0.2 | 5.0 | 10 |

## Summary

Compared with SAMatch:

- Sample-level Loose Gate decreases Mean Dice: `0.8629 -> 0.8494`
- Sample-level Loose Gate worsens Mean HD95: `6.681 -> 7.774`
- Class-wise Loose Gate slightly decreases Mean Dice: `0.8629 -> 0.8572`
- Class-wise Loose Gate improves Mean HD95: `6.681 -> 5.901`
- Class-wise Loose Gate keeps Mean ASD close: `1.560 -> 1.580`

Current observation:

- On ACDC 0_5 labeled, sample-level loose gate gives the best overall result.
- On ACDC 1 labeled, SAMatch keeps the best Mean Dice, while class-wise loose gate improves HD95.
- The reliability gate appears most beneficial in the lower-label regime.
- For ACDC 1 labeled, gate variants are treated as ablations rather than the selected main result.
