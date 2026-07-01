# ACDC Results Summary

Experiment setting:

- Dataset: ACDC
- Training length: 10000 iterations
- UniMatch baseline batch size: 8
- SAMatch and reliability gate batch size: 2
- Model: `unet_drop`
- Metrics: Dice, HD95, ASD

## Main Comparison

| Labeled Split | Method | Mean Dice | Mean HD95 | Mean ASD | Notes |
|---|---|---:|---:|---:|---|
| 0_5 | UniMatch | 0.7715 | 10.584 | 2.609 | Baseline |
| 0_5 | No SAM Refinement | 0.6395 | 21.839 | 7.684 | Continue training without MedSAM refined pseudo labels |
| 0_5 | SAMatch | 0.8148 | 4.592 | 1.313 | Refined pseudo labels from MedSAM |
| 0_5 | **Ours: Sample-level Loose Gate** | **0.8390** | **3.023** | **1.004** | Best 0_5 result |
| 1 | UniMatch | 0.8493 | 5.702 | 1.748 | Baseline |
| 1 | **SAMatch** | **0.8629** | 6.681 | **1.560** | Best Dice and ASD |
| 1 | Class-wise Loose Gate | 0.8572 | **5.901** | 1.580 | Better HD95 than SAMatch |
| 1 | Sample-level Loose Gate | 0.8494 | 7.774 | 2.261 | Not selected |
| 3 | UniMatch | 0.8517 | 3.962 | 1.072 | Baseline |
| 3 | **SAMatch** | **0.8751** | **1.608** | **0.438** | Best 3 labeled result |
| 3 | Class-wise Loose Gate | 0.8750 | 3.720 | 0.977 | Dice similar, boundary metrics worse |
| 3 | Sample-level Loose Gate | 0.8450 | 5.483 | 1.526 | Not selected |

## 0_5 Gate Ablation

| Method | Gate Mode | IoU Threshold | Area Ratio | Mean Dice | Mean HD95 | Mean ASD |
|---|---|---:|---|---:|---:|---:|
| No SAM Refinement | none | - | - | 0.6395 | 21.839 | 7.684 |
| SAMatch | none | - | - | 0.8148 | 4.592 | 1.313 |
| Gate v1 | class | 0.5 | 0.5-2.0 | 0.7990 | 5.292 | 1.424 |
| Medium Gate | class | 0.4 | 0.3-3.0 | 0.8212 | 4.670 | 1.470 |
| Class-wise Loose Gate | class | 0.3 | 0.2-5.0 | 0.8387 | 3.274 | 1.102 |
| Sample-level Strict Gate | sample | 0.5 | 0.5-2.0 | 0.7974 | 4.501 | 1.442 |
| Sample-level Medium Gate | sample | 0.4 | 0.3-3.0 | 0.8352 | 5.268 | 1.481 |
| **Sample-level Loose Gate** | sample | 0.3 | 0.2-5.0 | **0.8390** | **3.023** | **1.004** |

## Per-Class Best Results

| Labeled Split | Method | RV Dice | RV HD95 | RV ASD | MYO Dice | MYO HD95 | MYO ASD | LV Dice | LV HD95 | LV ASD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0_5 | Ours | 0.8004 | 5.789 | 1.737 | 0.8140 | 1.667 | 0.807 | 0.9025 | 1.613 | 0.469 |
| 1 | SAMatch | 0.8379 | 3.923 | 1.075 | 0.8466 | 7.051 | 1.641 | 0.9041 | 9.070 | 1.965 |
| 3 | SAMatch | 0.8760 | 2.072 | 0.549 | 0.8448 | 1.437 | 0.443 | 0.9044 | 1.314 | 0.321 |

## Current Interpretation

The reliability gate is most effective in the extremely low-label regime. On ACDC 0_5 labeled, the No SAM Refinement ablation drops to `0.6395` Mean Dice, showing that the improvement is not caused by simply continuing training in the SAMatch framework. MedSAM-refined pseudo labels are necessary, but they can also introduce unreliable refinements. The sample-level loose gate filters these cases by checking agreement and area consistency between the UniMatch pseudo label and the MedSAM refined mask, giving the best Dice, HD95, and ASD.

At 1 labeled, SAMatch keeps the best Dice and ASD, while class-wise loose gate improves HD95 slightly. At 3 labeled, SAMatch is already strong and the gate does not improve the boundary metrics. Sample-level loose gate also drops to `0.8450` Mean Dice at 3 labeled. This supports positioning the gate as a low-label reliability mechanism rather than a universally beneficial replacement for SAMatch refinement.

## Official Result Choice

Use the following as the main ACDC table:

| Labeled Split | Main Method To Report | Mean Dice | Mean HD95 | Mean ASD |
|---|---|---:|---:|---:|
| 0_5 | Ours: Sample-level Loose Gate | 0.8390 | 3.023 | 1.004 |
| 1 | SAMatch | 0.8629 | 6.681 | 1.560 |
| 3 | SAMatch | 0.8751 | 1.608 | 0.438 |

For the method contribution table, emphasize the 0_5 split where Ours improves over both UniMatch and SAMatch.
