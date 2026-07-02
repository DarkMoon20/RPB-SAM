# ACDC Reliability Gate 方法说明

本文档记录当前 ACDC 实验中 reliability gate 的设计动机、代码位置、训练流程和实验结论。该文档用于后续整理论文方法章节和实验分析。

## 1. 研究问题

SAMatch 的核心思路是利用 MedSAM 对半监督分割模型产生的伪标签进行 refine，然后将 refined pseudo label 用于无标签样本的训练。这个思路在低标注场景下有效，但存在一个问题：

> MedSAM refined mask 并不总是可靠。如果无条件使用 refined mask，错误 refinement 会把原本可用的 UniMatch pseudo label 覆盖掉，从而引入噪声。

因此，我们当前的改进目标是：

> 在使用 MedSAM refined pseudo label 之前，判断它是否可靠；可靠时使用 MedSAM refinement，不可靠时退回 UniMatch pseudo label。

这个判断模块就是 reliability gate。

## 2. ACDC 训练流程中的伪标签来源

当前主训练脚本为：

```text
train_unimatch_medsam_F2_ft_both_acdc.py
```

无标签样本的 UniMatch pseudo label 来自弱增强输入 `img_u_w` 的预测结果：

```python
pred_u_w = pred_u_w.detach()
conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
mask_u_w = pred_u_w.argmax(dim=1)
```

其中：

- `mask_u_w` 是 UniMatch 生成的无标签伪标签；
- `conf_u_w` 是对应的 confidence map；
- 后续 unsupervised loss 中仍然使用 `conf_u_w < conf_thresh` 作为 ignore mask。

## 3. MedSAM refined pseudo label 的生成

MedSAM refinement 的输入 prompt 来自 UniMatch pseudo label。对于 ACDC 的三个前景类别，脚本分别从 `mask_u_w == 1/2/3` 中生成 bounding box：

```python
img_pd_bbox = get_bbox256_cv(mask_u_w == 1, bbox_shift=args.bbox_shift)
medsam_logit_1, _ = sam(img_u_w_3ch, img_pd_bbox)

img_pd_bbox = get_bbox256_cv(mask_u_w == 2, bbox_shift=args.bbox_shift)
medsam_logit_2, _ = sam(img_u_w_3ch, img_pd_bbox)

img_pd_bbox = get_bbox256_cv(mask_u_w == 3, bbox_shift=args.bbox_shift)
medsam_logit_3, _ = sam(img_u_w_3ch, img_pd_bbox)
```

然后拼接背景和三个前景类别，得到 MedSAM refined prediction：

```python
medsam_logit_0 = 1 - (medsam_logit_1 + medsam_logit_2 + medsam_logit_3) / 3.0
a_tmp = torch.cat((medsam_logit_0, medsam_logit_1, medsam_logit_2, medsam_logit_3), dim=1)
pred_w_sam = a_tmp.softmax(dim=1)
mask_u_w_sam = pred_w_sam.argmax(dim=1)
```

其中：

- `mask_u_w_sam` 是 MedSAM refined pseudo label；
- 原始 SAMatch 默认直接用 `mask_u_w_sam` 参与无标签 loss。

## 4. Reliability Gate 的设计

Reliability gate 比较两个 mask：

- `mask_u_w`：UniMatch pseudo label；
- `mask_u_w_sam`：MedSAM refined pseudo label。

判断依据包括：

- 两者的类别区域 IoU；
- MedSAM mask 和 UniMatch mask 的面积比例；
- 前景区域是否过小。

当前实现包含两种模式：

```text
class mode: 逐类别判断是否接受 MedSAM refinement
sample mode: 整张样本级别判断是否接受 MedSAM refinement
```

对应函数在 `train_unimatch_medsam_F2_ft_both_acdc.py` 中：

```python
reliability_gate_pseudo_label(...)
reliability_gate_sample(...)
```

训练时通过参数控制：

```bash
--use_reliability_gate
--gate_mode class|sample
--gate_iou_thresh
--gate_area_min
--gate_area_max
--gate_min_area
```

最终选择逻辑：

```python
if args.disable_sam_refine:
    mask_u_w_final = mask_u_w
elif args.use_reliability_gate:
    mask_u_w_final, gate_accept_ratio = gate_fn(mask_u_w, mask_u_w_sam, ...)
else:
    mask_u_w_final = mask_u_w_sam
```

也就是说：

- 不使用 SAM refinement 时，直接使用 `mask_u_w`；
- 原始 SAMatch 使用 `mask_u_w_sam`；
- Ours 使用 gate 后的 `mask_u_w_final`。

## 5. Refined Mask 如何参与 Loss

最终用于无标签 loss 的是 `mask_u_w_final`：

```python
loss_u_w_fp = dice_loss(
    pred_u_w_fp.softmax(dim=1),
    mask_u_w_final.unsqueeze(1).float(),
    ignore=(conf_u_w < args.conf_thresh).float()
)
```

因此 reliability gate 的作用点非常明确：

> 它不改变模型结构，不改变 MedSAM prompt 生成方式，只改变最终无标签 loss 使用哪个 pseudo label。

这使得当前改动是最小侵入式的，也方便做消融实验。

## 6. 关键消融实验

ACDC 0_5 labeled 的主要结果如下：

| 方法 | Mean Dice | HD95 | ASD | 说明 |
|---|---:|---:|---:|---|
| UniMatch | 0.7715 | 10.584 | 2.609 | 原始半监督 baseline |
| No SAM Refinement | 0.6395 | 21.839 | 7.684 | 继续训练但不使用 MedSAM refined label |
| SAMatch | 0.8148 | 4.592 | 1.313 | 无条件使用 MedSAM refined label |
| Ours: Sample-level Loose Gate | 0.8390 | 3.023 | 1.004 | 使用 reliability gate 过滤 unreliable refinement |

这个结果说明：

1. MedSAM refinement 是有价值的，因为 SAMatch 明显优于 UniMatch；
2. 仅仅继续训练并不能带来提升，因为 No SAM Refinement 反而明显退化；
3. MedSAM refinement 仍然存在噪声，reliability gate 可以进一步提升结果。

## 7. Gate 阈值消融

ACDC 0_5 labeled 中，sample-level gate 的阈值消融如下：

| Gate 设置 | IoU 阈值 | 面积比例 | Mean Dice | HD95 | ASD | Avg Accept | Reject Steps |
|---|---:|---|---:|---:|---:|---:|---:|
| Strict | 0.5 | 0.5-2.0 | 0.7974 | 4.501 | 1.442 | 0.609 | 81/128 |
| Medium | 0.4 | 0.3-3.0 | 0.8352 | 5.268 | 1.481 | 0.812 | 41/128 |
| Loose | 0.3 | 0.2-5.0 | 0.8390 | 3.023 | 1.004 | 0.883 | 26/128 |

观察：

- strict gate 接受率只有 60.9%，过滤过多，导致 Dice 下降；
- medium gate 接受率提高到 81.2%，Dice 接近最终结果，但 HD95 和 ASD 不稳定；
- loose gate 接受率为 88.3%，仍然能过滤一部分不可靠 refinement，同时保留大多数有用的 MedSAM refinement，因此综合表现最好。

## 8. 不同标注比例下的表现

当前 ACDC 结果显示 reliability gate 最适合极低标注场景。

| Split | 最佳方法 | Mean Dice | HD95 | ASD | 结论 |
|---|---|---:|---:|---:|---|
| 0_5 | Ours | 0.8390 | 3.023 | 1.004 | gate 明显有效 |
| 1 | SAMatch | 0.8629 | 6.681 | 1.560 | gate 可改善部分 HD95，但 Dice 不占优 |
| 3 | SAMatch | 0.8751 | 1.608 | 0.438 | SAMatch 已足够稳定，gate 不再带来收益 |

因此，论文叙事应避免宣称 gate 在所有标注比例下都提升。更准确的表述是：

> Reliability gate is most effective in extremely low-label settings, where SAM refinement is useful but less stable.

中文表述可以写为：

> 在极低标注比例下，MedSAM refinement 能提供额外结构先验，但其结果并不总是可靠。Reliability gate 通过一致性和面积约束筛选 refined pseudo label，在保留大部分有效 refinement 的同时过滤错误 refinement，因此在 0_5 labeled ACDC 上取得最明显提升。

## 9. 当前方法边界

当前方法仍有几个边界：

1. Gate 是基于 mask-level 几何一致性，不直接利用像素级 confidence；
2. Confidence-aware prompt 已做初步实验，但当前结果不稳定，暂不纳入最终方法；
3. 在 1 labeled 和 3 labeled 下，gate 不稳定，说明阈值可能需要随标注比例或类别自适应；
4. BUSI 实验当前 baseline 不稳定，暂不作为主要证据。

这些边界可以作为后续改进方向，而不是当前阶段必须解决的问题。

### Confidence-aware Prompt 负结果

我们尝试过使用高置信区域生成 MedSAM box prompt：

```text
prompt mask = (mask_u_w == cls) & (conf_u_w >= threshold)
```

如果高置信区域太小，则退回原始 pseudo mask。实验结果如下：

| 方法 | Prompt 阈值 | Gate | Mean Dice | HD95 | ASD |
|---|---:|---|---:|---:|---:|
| SAMatch | - | no | 0.8148 | 4.592 | 1.313 |
| Ours: Sample-level Loose Gate | - | yes | 0.8390 | 3.023 | 1.004 |
| Confidence Prompt Only | 0.90 | no | 0.7904 | 5.343 | 1.308 |
| Confidence Prompt + Gate | 0.90 | yes | 0.7736 | 7.047 | 1.998 |
| Confidence Prompt + Gate | 0.95 | yes | 0.4346 | 13356.204 | 13340.792 |

该结果说明，直接用 confidence 过滤 prompt mask 会移除部分对 box prompt 有用的空间支持，导致 MedSAM prompt 不稳定。尤其当阈值提高到 0.95 时，高置信区域过小或不连续，最终造成严重退化。因此，当前最终方法只保留 reliability gate，不采用 confidence-aware prompt。

## 10. 后续可写入论文的主张

建议论文主张聚焦于以下三点：

1. SAM-based refinement improves pseudo labels but may introduce unreliable masks.
2. A lightweight reliability gate can select between the original pseudo label and the SAM-refined pseudo label.
3. The gate is especially beneficial in extremely low-label segmentation, as demonstrated on ACDC 0_5 labeled.

对应中文表述：

> 本研究针对 SAM 辅助半监督医学图像分割中的 refined pseudo label 可靠性问题，提出一种轻量级 reliability gate。该 gate 在不改变模型结构和 prompt 生成方式的前提下，根据原始伪标签与 SAM refined mask 的一致性决定是否接受 refinement。实验表明，在 ACDC 极低标注场景下，该机制能够有效过滤不可靠 refinement，并提升 Dice、HD95 和 ASD。
