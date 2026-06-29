# 硕士毕设方案：基于可靠 Prompt 与边界感知伪标签修正的 SAM 半监督医学图像分割

> 建议文件名：`SAMatch_RPB_SAM_Thesis_Plan.md`  
> 建议放置位置：项目根目录，例如 `/root/autodl-tmp/SAMatch/SAMatch_RPB_SAM_Thesis_Plan.md`  
> 用途：让 Codex 读取本文档后，理解课题目标、baseline、创新点、代码改进方向和实验计划。

---

## 0. 课题一句话概括

本课题拟以 **SAMatch** 作为主 baseline，研究 **SAM 辅助半监督医学图像分割中的可靠性问题**。核心思想不是简单使用 SAM 生成伪标签，而是从 **Prompt 生成前、SAM 修正中、伪标签采纳后** 三个阶段提升 SAM 伪监督信号的可靠性，最终形成：

**RPB-SAMatch: Reliable Prompt and Boundary-aware SAM-guided Semi-supervised Medical Image Segmentation**

中文题目可写为：

**基于可靠 Prompt 与边界感知伪标签修正的半监督医学图像分割方法研究**

---

## 1. 研究背景与问题定位

医学图像分割需要像素级或体素级标注，标注依赖专业医生，成本高、周期长。因此，半监督医学图像分割希望利用少量标注数据和大量未标注数据提升模型性能。

近年来，SAM / MedSAM 等 foundation model 具备较强的 promptable segmentation 能力，被用于辅助医学图像分割。SAMatch、CPC-SAM、SemiSAM 等方法已经证明 SAM 可以用于半监督医学图像分割。

但是，现有方法仍存在一个关键问题：

> 在半监督医学图像分割中，SAM 的 prompt 通常来自模型对无标签图像的预测，而这些预测本身可能不可靠。错误 prompt 会导致错误 SAM mask，错误 SAM mask 又会作为伪标签反过来训练模型，从而放大 confirmation bias。

因此，本课题的核心问题是：

**如何在半监督医学图像分割中更可靠地使用 SAM？**

具体拆成三个子问题：

1. **Prompt 生成是否可靠？**  
   无标签图像没有真值，直接从预测 mask 随机采点可能产生错误 prompt。

2. **SAM 应该修正哪些区域？**  
   医学图像主要错误集中在器官边界、低对比区域和形态模糊区域，不一定需要整张图都让 SAM 重分割。

3. **SAM 输出是否一定可信？**  
   SAM/MedSAM 不是 ground truth，在医学图像中也可能产生过分割、欠分割、空 mask、多连通域异常等问题。

---

## 2. 主 baseline 选择

### 2.1 首选 baseline：SAMatch

论文：**A SAM-guided and Match-based Semi-Supervised Segmentation Framework for Medical Imaging**  
代码：`https://github.com/apple1986/SAMatch`  
论文链接：`https://arxiv.org/abs/2411.16949`

选择原因：

1. 它已经是 **SAM + 半监督医学图像分割** 的直接相关工作。
2. 它基于 match-based 半监督框架，用 SAM/MedSAM 生成或修正伪标签。
3. 它包含 ACDC、BUSI、MRLiver 等实验，ACDC 是最适合中期快速出成果的数据集。
4. 它的改进入口比较清楚：prompt 生成、MedSAM 修正、pseudo label 回传训练。

SAMatch 原始流程可概括为：

```text
Match-based SSL model / UniMatch / FixMatch
        ↓
对无标签图像生成高置信预测
        ↓
由预测结果生成 prompt
        ↓
输入 SAM / MedSAM 得到 refined pseudo label
        ↓
将 refined pseudo label 反馈训练半监督分割模型
```

本课题的改进位置就在：

```text
prompt 生成阶段
SAM refined mask 选择阶段
伪标签损失计算阶段
```

---

## 3. 相关论文思想来源

本方法不是凭空设计，也不是简单堆模块，而是围绕“如何可靠使用 SAM”这一问题，从已有论文中吸收思想并重新组织。

| 模块 | 主要思想来源 | 借鉴内容 | 本课题中的改造 |
|---|---|---|---|
| 主框架 | SAMatch | SAM-guided + match-based 半监督医学分割 | 作为 baseline，重点改 prompt 和伪标签可靠性 |
| Prompt 机制 | CPC-SAM / SemiSAM | SAM 的 prompt-driven 分割能力可以用于 SSL | 从“使用 prompt”改为“选择可靠 prompt” |
| 置信度筛选 | UA-MT / ABD | 用 uncertainty / confidence 判断伪标签可靠性 | 把置信度用于 prompt 采样和 SAM mask 筛选 |
| 边界感知 | boundary-aware medical segmentation / MedSAM 分析 | 医学图像弱边界、低对比区域是主要难点 | 让 SAM 重点修正边界不确定区域 |
| 伪标签去噪 | pseudo-label filtering / consistency learning | 错误伪标签会产生 confirmation bias | 设计 SAM pseudo-label reliability gate |

---

## 4. 拟提出方法：RPB-SAMatch

方法全称：

**Reliable Prompt and Boundary-aware SAMatch**

核心目标：

> 将 SAM 从“直接生成伪标签的工具”，改造成“受置信度、边界不确定性和可靠性门控约束的选择性伪标签修正器”。

整体流程：

```text
Labeled data:   X_l, Y_l
Unlabeled data: X_u
        ↓
Train / load SAMatch baseline
        ↓
SSL model predicts probability map P_u on unlabeled image
        ↓
Compute confidence map C_u and entropy map H_u
        ↓
Confidence-aware prompt selection
        ↓
Boundary-aware prompt refinement
        ↓
MedSAM / SAM generates refined mask Y_sam
        ↓
Reliability gate checks whether Y_sam is trustworthy
        ↓
Use reliable SAM mask as pseudo label; otherwise fall back to teacher / baseline pseudo label
        ↓
Reliability-weighted unsupervised loss
```

---

## 5. 创新点 1：Confidence-aware Prompt Selection

### 5.1 问题

SAM 的分割质量高度依赖 prompt。如果 prompt 来自错误预测，SAM 可能输出更错误的 mask。

在半监督训练中，prompt 通常来自模型对无标签图像的预测。训练早期或低标注比例下，预测不稳定，随机从预测 mask 中采点容易导致错误 prompt。

错误链条：

```text
错误预测 → 错误 prompt → 错误 SAM mask → 错误伪标签 → 模型继续学习错误
```

### 5.2 方法

给定无标签图像预测概率图：

```text
P_u = model(x_u)
```

计算每个像素的最大类别概率作为置信度：

```text
C_u = max_c P_u[c]
```

计算预测熵：

```text
H_u = -sum_c P_u[c] * log(P_u[c])
```

Prompt 采样规则：

```text
positive point:
    从高置信前景区域采样，例如 C_u > tau_fg 且 pseudo_label != background

negative point:
    从高置信背景区域采样，例如 C_u > tau_bg 且 pseudo_label == background

avoid:
    不从低置信区域、边界模糊区域随机采点
```

建议默认阈值：

```text
tau_fg = 0.8
tau_bg = 0.9
entropy_threshold = 可选
```

也可以使用动态阈值：

```text
training early: tau = 0.6
training later: tau = 0.8 or 0.9
```

### 5.3 论文表述

本文提出置信度感知的 prompt 选择策略，根据模型预测概率和熵值从无标签图像中筛选高可靠前景点与背景点，避免低置信伪标签直接生成错误 prompt，从源头上降低 SAM 辅助伪标签修正过程中的噪声传播。

---

## 6. 创新点 2：Boundary-aware SAM Refinement

### 6.1 问题

医学图像分割的主要错误往往集中在边界区域，例如：

- 心肌边界模糊；
- 右心室形态变化大；
- 病灶与周围组织灰度接近；
- 超声图像噪声强；
- CT/MRI 低对比区域不清晰。

如果让 SAM 对整张图重新生成 mask，可能出现两个问题：

1. 器官内部本来预测正确，却被 SAM 改坏；
2. 真正需要修正的边界区域没有被重点处理。

### 6.2 方法

从 teacher 或当前模型预测 mask 中提取边界：

```text
B = Dilate(pseudo_mask) - Erode(pseudo_mask)
```

再结合不确定性：

```text
U = 1 - C_u
BoundaryUncertainRegion = B AND (U > tau_u)
```

在边界附近采样正负点：

```text
positive boundary prompt:
    从预测前景边界内侧的高置信区域采样

negative boundary prompt:
    从预测前景边界外侧的高置信背景区域采样
```

SAM 输入可以使用：

```text
box prompt + confidence-selected points + boundary points
```

### 6.3 边界损失

可以进一步加入边界加权损失：

```text
L_boundary = CE(pred, pseudo_label) * W_boundary
```

其中边界区域权重大，非边界区域权重小。

示例：

```text
W_boundary = 1 + alpha * boundary_mask
alpha = 1.0 or 2.0
```

### 6.4 论文表述

针对医学图像中器官边界模糊、伪标签边缘误差较大的问题，本文提出边界不确定性感知的 SAM 修正策略，通过形态学边界提取和预测不确定性估计定位高风险边界区域，并利用边界附近的正负 prompt 引导 SAM 进行局部结构修正，从而提升半监督分割的边界质量。

---

## 7. 创新点 3：SAM Pseudo-label Reliability Gate

### 7.1 问题

SAM / MedSAM 输出并不一定始终优于当前模型预测。可能出现：

- 空 mask；
- mask 面积异常过大或过小；
- 分割到错误组织；
- 连通域数量异常；
- 与 teacher pseudo label 差异过大；
- 过度平滑边界。

如果直接把 SAM 输出作为伪标签，可能会把错误监督信号引入训练。

### 7.2 方法

设计可靠性门控机制。给定：

```text
Y_teacher: teacher / SSL model pseudo label
Y_sam: SAM / MedSAM refined mask
```

计算：

```text
iou = IoU(Y_teacher, Y_sam)
area_ratio = area(Y_sam) / area(Y_teacher)
num_components = connected_components(Y_sam)
empty_flag = area(Y_sam) == 0
```

接受 SAM mask 的条件：

```text
iou > gamma_iou
area_min < area_ratio < area_max
empty_flag == False
num_components <= max_components
```

建议默认值：

```text
gamma_iou = 0.5
area_min = 0.5
area_max = 2.0
max_components = 3 或按类别设置
```

如果 SAM mask 通过门控：

```text
Y_final = Y_sam
weight = iou or 1.0
```

如果不通过：

```text
Y_final = Y_teacher
weight = lower_weight，例如 0.3 或 0.5
```

也可以直接丢弃不可靠 SAM 样本：

```text
ignore this unlabeled sample / region
```

### 7.3 论文表述

本文提出一种 SAM 伪标签可靠性门控机制，从区域一致性、面积比例和连通域结构等角度评估 SAM 修正结果的可信度，仅在 SAM 输出与当前模型预测保持合理一致时引入其作为伪监督信号，避免 foundation model 在医学图像上的错误修正被半监督训练进一步放大。

---

## 8. 损失函数设计

总损失：

```text
L = L_sup + lambda_u * L_unsup + lambda_b * L_boundary + lambda_s * L_sam
```

其中：

```text
L_sup = CE(pred_l, y_l) + DiceLoss(pred_l, y_l)
```

```text
L_unsup = CE(pred_u, Y_final) + DiceLoss(pred_u, Y_final)
```

```text
L_boundary = boundary_weighted_CE(pred_u, Y_final, boundary_mask)
```

```text
L_sam = reliability_weight * DiceLoss(pred_u, Y_sam)
```

如果为了中期快速实现，建议先实现简化版：

```text
L = L_sup + lambda_u * reliability_weight * L_unsup
```

再逐步加入 boundary loss。

---

## 9. 实验设计

### 9.1 数据集

主数据集：

```text
ACDC cardiac MRI segmentation
```

原因：

- 2D 任务，显存压力低于 3D；
- 半监督医学分割论文常用；
- SAMatch / CPC-SAM / ABD 都涉及 ACDC 或相关代码；
- 中期最快出结果。

可选补充数据集：

```text
BUSI breast ultrasound segmentation
```

原因：

- 超声图像边界模糊，适合验证 boundary-aware 模块；
- SAMatch 也提供 BUSI 相关支持。

### 9.2 标注比例

中期优先：

```text
10% labeled
```

毕业论文完整实验：

```text
5% labeled
10% labeled
20% labeled
```

如果 SAMatch 原文使用特殊比例，例如 0.5%、1%、3%、7%，可以按原仓库默认划分为准。

### 9.3 对比方法

最少对比：

```text
Fully supervised U-Net / VNet / baseline network
UniMatch / FixMatch
SAMatch
Ours
```

完整对比：

```text
U-Net supervised
Mean Teacher / UA-MT
CPS / MC-Net+ / BCP
UniMatch
SAMatch
CPC-SAM 或 ABD
Ours
```

中期不要求全部跑完，优先跑：

```text
SAMatch baseline
SAMatch + Reliability Gate
Ours full preliminary
```

### 9.4 指标

主指标：

```text
Dice ↑
Jaccard / IoU ↑
HD95 ↓
ASD ↓
```

如果时间有限，至少输出：

```text
Dice
HD95
```

因为 Dice 体现区域重叠，HD95 体现边界质量。

---

## 10. 消融实验设计

| Method | Confidence Prompt | Boundary Prompt | Reliability Gate | Boundary Loss | Dice ↑ | HD95 ↓ |
|---|---|---|---|---|---:|---:|
| SAMatch | × | × | × | × |  |  |
| Ours-A | × | × | √ | × |  |  |
| Ours-B | √ | × | √ | × |  |  |
| Ours-C | √ | √ | √ | × |  |  |
| Ours-Full | √ | √ | √ | √ |  |  |

最先实现：

```text
Ours-A = SAMatch + Reliability Gate
```

原因：

- 代码改动最小；
- 最容易验证 SAM 输出不是始终可靠；
- 最适合作为中期初步创新实验。

---

## 11. 可视化分析

建议输出以下图：

```text
Original image
Ground Truth
SAMatch prediction
Ours prediction
Error map
Boundary error map
```

重点展示：

1. SAMatch 边界外溢，Ours 更贴合 GT；
2. SAMatch 漏分细长结构，Ours 改善；
3. SAM 修正失败时，Reliability Gate 拒绝了错误 mask；
4. Confidence Prompt 采样点落在高置信前景/背景区域，而不是模糊边界上。

---

## 12. 代码实现建议：让 Codex 优先检查的位置

Codex 需要先阅读 SAMatch 仓库结构，重点找以下内容：

```text
1. ACDC 训练脚本
   例如：train_unimatch_acdc.py
        train_fixmatch_medsam_F2_ft_both_acdc.py
        train_unimatch_medsam_F2_ft_both_acdc.py

2. MedSAM / SAM 加载位置
   查找关键词：sam, medsam, checkpoint, prompt, mask_decoder, image_encoder

3. prompt 生成函数
   查找关键词：prompt, point, box, bbox, coord, label

4. pseudo label 生成位置
   查找关键词：pseudo, pseudo_label, mask, pred_u, logits_u, unlabeled

5. unsupervised loss 计算位置
   查找关键词：loss_u, unsup, consistency, criterion_u, dice_loss

6. evaluation / test 脚本
   查找关键词：dice, hd95, asd, test_ssl_acdc.py
```

如果找不到准确文件名，Codex 应先运行：

```bash
find . -maxdepth 3 -type f | sort
 grep -R "pseudo\|prompt\|medsam\|sam\|loss_u\|unlabeled" -n . | head -200
```

---

## 13. 最小代码改动方案：先实现 Reliability Gate

### 13.1 新增工具函数

建议新建文件：

```text
utils/reliability_gate.py
```

或者放入现有 `utils.py`。

伪代码：

```python
import torch
import torch.nn.functional as F


def binary_iou(mask_a, mask_b, eps=1e-6):
    """mask_a, mask_b: bool or 0/1 tensors, shape [B, H, W] or [H, W]."""
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    inter = (mask_a & mask_b).float().sum(dim=(-2, -1))
    union = (mask_a | mask_b).float().sum(dim=(-2, -1))
    return (inter + eps) / (union + eps)


def area_ratio(mask_sam, mask_teacher, eps=1e-6):
    area_sam = mask_sam.float().sum(dim=(-2, -1))
    area_teacher = mask_teacher.float().sum(dim=(-2, -1))
    return (area_sam + eps) / (area_teacher + eps)


def reliability_gate(mask_sam, mask_teacher,
                     iou_thr=0.5,
                     area_min=0.5,
                     area_max=2.0):
    """
    Return:
        accept: bool tensor [B]
        score: reliability score [B]
    """
    iou = binary_iou(mask_sam, mask_teacher)
    ratio = area_ratio(mask_sam, mask_teacher)
    non_empty = mask_sam.float().sum(dim=(-2, -1)) > 0
    accept = (iou > iou_thr) & (ratio > area_min) & (ratio < area_max) & non_empty
    score = iou.clamp(0.0, 1.0)
    return accept, score
```

### 13.2 接入训练流程

在得到：

```python
pseudo_teacher
pseudo_sam
```

之后加入：

```python
accept, score = reliability_gate(pseudo_sam, pseudo_teacher)

pseudo_final = torch.where(
    accept[:, None, None],
    pseudo_sam,
    pseudo_teacher
)

loss_weight = torch.where(
    accept,
    score,
    torch.full_like(score, 0.5)
)
```

然后无标签损失：

```python
loss_u = weighted_unsup_loss(pred_u, pseudo_final, loss_weight)
```

如果原始代码不支持 sample-wise weight，可以先做简化：

```python
loss_u = loss_fn(pred_u, pseudo_final)
```

但将被拒绝的样本权重设低。

---

## 14. 第二步代码改动：Confidence-aware Prompt Selection

### 14.1 置信度图

假设模型输出 logits：

```python
prob = torch.softmax(logits_u, dim=1)
conf, pseudo = torch.max(prob, dim=1)
```

### 14.2 采样 high-confidence positive / negative points

伪代码：

```python
def sample_confidence_points(prob, pseudo, num_pos=1, num_neg=1,
                             tau_fg=0.8, tau_bg=0.9,
                             background_id=0):
    conf, pred = torch.max(prob, dim=1)  # [B, H, W]
    points = []
    labels = []

    for b in range(prob.shape[0]):
        fg_candidates = torch.nonzero((pred[b] != background_id) & (conf[b] > tau_fg), as_tuple=False)
        bg_candidates = torch.nonzero((pred[b] == background_id) & (conf[b] > tau_bg), as_tuple=False)

        # fallback: if no high-confidence points, use center of predicted mask or random valid point
        pos = choose_points(fg_candidates, num_pos)
        neg = choose_points(bg_candidates, num_neg)

        # SAM usually expects points in x, y order
        pts = torch.cat([pos[:, [1, 0]], neg[:, [1, 0]]], dim=0)
        lbs = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))], dim=0)

        points.append(pts)
        labels.append(lbs)

    return points, labels
```

Codex 需要根据 SAMatch 当前 prompt 格式修改 shape。

---

## 15. 第三步代码改动：Boundary-aware Prompt / Boundary Loss

### 15.1 边界提取

用 max pooling 实现 dilation / erosion：

```python
def get_boundary(mask, kernel_size=5):
    # mask: [B, H, W], 0/1 or class mask
    mask = mask.float().unsqueeze(1)
    dilation = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)
    erosion = -F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size // 2)
    boundary = (dilation - erosion).squeeze(1)
    return boundary > 0
```

多类别时可以逐类提边界再合并。

### 15.2 边界加权损失

```python
def boundary_weighted_ce(logits, target, boundary_mask, alpha=2.0):
    ce = F.cross_entropy(logits, target.long(), reduction='none')
    weight = 1.0 + alpha * boundary_mask.float()
    return (ce * weight).mean()
```

---

## 16. 推荐执行路线

### 阶段 1：复现 baseline

```text
目标：跑通 SAMatch on ACDC
输出：baseline Dice / HD95 / 日志 / checkpoint
```

Codex 任务：

```text
阅读 SAMatch 仓库，整理 ACDC 训练与测试命令，确认数据路径、checkpoint 路径和输出结果路径。不要修改算法，先保证 baseline 能跑通。
```

### 阶段 2：实现 Reliability Gate

```text
目标：最小改动验证 SAM mask 不是始终可靠
输出：SAMatch + Gate 的 Dice / HD95
```

Codex 任务：

```text
在 SAMatch 的 ACDC 训练流程中找到 SAM refined pseudo label 和 teacher/model pseudo label 生成的位置，加入 reliability gate。当 SAM mask 与 teacher mask 的 IoU 小于 0.5，或面积比例小于 0.5/大于 2.0，或 SAM mask 为空时，不使用 SAM mask，回退到 teacher pseudo label。保持原训练框架不变，并记录 accept_rate。
```

### 阶段 3：实现 Confidence-aware Prompt

```text
目标：提高输入 SAM 的 prompt 可靠性
输出：SAMatch + Gate + Confidence Prompt
```

Codex 任务：

```text
修改 prompt 生成逻辑：不要从预测 mask 中随机采点，而是根据 softmax 置信度图，从高置信前景区域采 positive points，从高置信背景区域采 negative points。增加 tau_fg、tau_bg、num_pos、num_neg 参数，并在日志中记录采样失败 fallback 次数。
```

### 阶段 4：实现 Boundary-aware Prompt / Boundary Loss

```text
目标：提升边界指标 HD95 / ASD
输出：Ours-Full
```

Codex 任务：

```text
基于 pseudo label 提取边界区域，在边界附近采样额外 positive/negative prompt，并加入 boundary-weighted CE loss。要求新增参数 alpha_boundary，并能通过命令行开关启用或关闭该模块，方便消融实验。
```

---

## 17. 给 Codex 的总提示词

可以直接复制下面这段给 Codex：

```text
你现在在一个 SAMatch 半监督医学图像分割项目中。请先阅读当前仓库结构和本文件 SAMatch_RPB_SAM_Thesis_Plan.md。我的硕士毕设目标是以 SAMatch 为 baseline，提出 Reliable Prompt and Boundary-aware SAMatch，核心不是重新训练大模型，而是在 SAMatch 的 prompt 生成、SAM 伪标签采纳和无标签损失计算阶段加入可靠性机制。

请按以下优先级工作：
1. 找到 ACDC 训练脚本、测试脚本、SAM/MedSAM 加载位置、prompt 生成位置、pseudo label 生成位置、unsupervised loss 计算位置。
2. 不大改原始训练框架，先实现 SAM pseudo-label reliability gate：当 SAM mask 与 teacher/model pseudo label 的 IoU 小于 0.5，或面积比例小于 0.5/大于 2.0，或 SAM mask 为空时，不采用 SAM mask，回退到 teacher/model pseudo label。
3. 为 reliability gate 增加日志统计：accept_rate、mean_iou、mean_area_ratio、rejected_empty_count。
4. 代码要保留开关参数，例如 --use_reliability_gate，保证可以跑 ablation。
5. 初步跑通后，再实现 confidence-aware prompt selection：根据 softmax confidence 选择高置信前景 positive points 和高置信背景 negative points。
6. 最后实现 boundary-aware prompt 或 boundary-weighted loss，用于改善 HD95/ASD。

请每次修改前说明要改哪些文件和函数；修改后给出运行命令和预期输出文件。优先保证 ACDC 10% labeled setting 能跑通。
```

---

## 18. 中期汇报口径

可以这样讲：

```text
本课题研究少标注条件下的医学图像半监督分割问题。考虑到 SAM/MedSAM 等基础分割模型具有较强的 promptable segmentation 能力，本文拟以 SAMatch 为基础框架，将 SAM 用于无标签样本伪标签修正。与已有方法不同，本文并不直接默认 SAM 输出一定可靠，而是针对 SAM 辅助半监督训练中的 prompt 噪声和伪标签误传播问题，设计置信度感知 prompt 选择、边界不确定性感知修正和 SAM 伪标签可靠性门控机制，从 prompt 生成、边界修正和伪标签采纳三个阶段提升无标签监督信号质量。实验计划在 ACDC 心脏 MRI 分割数据集上进行，以 Dice、Jaccard、HD95 和 ASD 作为评价指标，并通过不同标注比例和消融实验验证各模块有效性。
```

---

## 19. 风险与替代方案

### 风险 1：SAMatch 代码跑不通

替代：

```text
改用 UniMatch / FixMatch + MedSAM 离线伪标签修正
```

即：

```text
先训练 UniMatch → 预测无标签图像 → 生成 box/point prompt → MedSAM 离线修正 → 保存 pseudo labels → 再训练最终模型
```

### 风险 2：显存不够

解决：

```text
batch_size 降到 1 或 2
冻结 MedSAM / SAM
离线生成 SAM pseudo labels
只跑 ACDC 2D
```

### 风险 3：创新模块提升不明显

优先观察：

```text
HD95 / ASD 是否下降
边界可视化是否更好
SAM mask 被拒绝样本是否确实质量差
accept_rate 是否合理
```

即使 Dice 提升不大，如果 HD95 明显改善，也能支撑边界感知模块。

---

## 20. 最终论文贡献写法

建议写成三点：

```text
1. 针对 SAM 辅助半监督医学图像分割中 prompt 来源不可靠的问题，提出置信度感知 prompt 选择策略，从高置信前景和背景区域采样正负提示，减少错误预测对 SAM 修正结果的误导。

2. 针对医学图像器官边界模糊、伪标签边界误差较大的问题，提出边界不确定性感知的 SAM 修正机制，利用边界区域正负 prompt 和边界加权损失增强模型对高风险区域的学习能力。

3. 针对 SAM 输出在医学图像中并非始终可靠的问题，提出 SAM 伪标签可靠性门控机制，根据 SAM mask 与 teacher pseudo label 的一致性、面积比例和结构合理性动态决定是否采用 SAM 伪标签，从而降低错误伪标签传播风险。
```

---

## 21. 当前最小可交付目标

中期前最低目标：

```text
1. SAMatch ACDC baseline 跑通；
2. 加入 Reliability Gate；
3. 输出 baseline vs Gate 的 Dice / HD95；
4. 画出方法框架图；
5. 做 2-3 张可视化：SAMatch、Ours、GT、Error Map；
6. 写出中期报告中的研究背景、方法设计和实验计划。
```

毕业论文完整目标：

```text
1. ACDC 上完成 5% / 10% / 20% labeled experiments；
2. 补充 BUSI 数据集；
3. 对比 SAMatch、UniMatch、Mean Teacher、BCP/ABD；
4. 完成 Confidence Prompt、Boundary Prompt、Reliability Gate 的消融实验；
5. 统计 SAM accept_rate 和 rejected samples；
6. 完成可视化与失败案例分析。
```

---

## 22. 参考文献与代码链接

1. SAMatch: A SAM-guided and Match-based Semi-Supervised Segmentation Framework for Medical Imaging  
   Paper: https://arxiv.org/abs/2411.16949  
   Code: https://github.com/apple1986/SAMatch

2. CPC-SAM: Cross Prompting Consistency with Segment Anything Model for Semi-supervised Medical Image Segmentation  
   Paper: https://arxiv.org/abs/2407.05416  
   Code: https://github.com/JuzhengMiao/CPC-SAM

3. SemiSAM: Enhancing Semi-Supervised Medical Image Segmentation via SAM-Assisted Consistency Regularization  
   Paper: https://arxiv.org/abs/2312.06316  
   Code: https://github.com/YichiZhang98/SemiSAM

4. ABD: Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation  
   Paper: https://arxiv.org/abs/2405.00378  
   Code: https://github.com/Star-chy/ABD

5. SSL4MIS: Semi-supervised Learning for Medical Image Segmentation Benchmark  
   Code: https://github.com/HiLab-git/SSL4MIS

---

## 23. 给自己和 Codex 的注意事项

1. 不要把创新点写成“本文使用 SAM”。  
   应写成“本文研究如何可靠地使用 SAM 辅助半监督医学图像分割”。

2. 不要一开始做太多模块。  
   先做 Reliability Gate，再做 Confidence Prompt，最后做 Boundary Loss。

3. 每个模块都必须有 ablation 开关。  
   例如：

```bash
--use_reliability_gate
--use_confidence_prompt
--use_boundary_prompt
--use_boundary_loss
```

4. 每个模块都要有日志。  
   例如：

```text
accept_rate
mean_sam_teacher_iou
mean_area_ratio
empty_mask_count
fallback_count
boundary_loss_value
```

5. 中期最重要的是跑通和有初步结果，不要陷入大规模重构。

