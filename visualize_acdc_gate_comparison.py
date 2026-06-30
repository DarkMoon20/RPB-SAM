import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom

from networks.net_factory import net_factory


CLASS_NAMES = ["BG", "RV", "MYO", "LV"]
LABEL_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 57, 70],
        [69, 123, 157],
        [42, 157, 143],
    ],
    dtype=np.float32,
) / 255.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/ACDC")
    parser.add_argument("--model", type=str, default="unet_drop")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="./visualizations/acdc_0_5_gate")
    parser.add_argument(
        "--unimatch_ckpt",
        type=str,
        default="./checkpoint/ACDC/Unimatch_0_5_labeled_bs8/unet_drop/unet_drop_best_model.pth",
    )
    parser.add_argument(
        "--samatch_ckpt",
        type=str,
        default="./checkpoint/ACDC/Unimatch_medsam_both_0_5_labeled_bs2/unet_drop/unet_drop_best_model.pth",
    )
    parser.add_argument(
        "--gate_ckpt",
        type=str,
        default="./checkpoint/ACDC/Unimatch_medsam_gate_sample_loose_0_5_labeled_bs2/unet_drop/unet_drop_best_model.pth",
    )
    return parser.parse_args()


def load_model(ckpt_path, args):
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()
    return model


def predict_volume(model, image, args):
    prediction = np.zeros_like(image, dtype=np.uint8)
    for ind in range(image.shape[0]):
        image_slice = image[ind]
        h, w = image_slice.shape
        resized = zoom(image_slice, (args.patch_size / h, args.patch_size / w), order=0)
        input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(args.device)
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
        pred = pred.cpu().numpy().astype(np.uint8)
        prediction[ind] = zoom(pred, (h / args.patch_size, w / args.patch_size), order=0).astype(np.uint8)
    return prediction


def dice_score(pred, gt, cls):
    pred_cls = pred == cls
    gt_cls = gt == cls
    denom = pred_cls.sum() + gt_cls.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred_cls, gt_cls).sum() / denom


def mean_foreground_dice(pred, gt, num_classes):
    scores = [dice_score(pred, gt, cls) for cls in range(1, num_classes)]
    return float(np.mean(scores))


def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-9)


def colorize_mask(mask):
    rgb = LABEL_COLORS[mask.astype(np.int64)]
    return rgb


def overlay_mask(image_slice, mask, alpha=0.45):
    image_rgb = np.repeat(image_slice[..., None], 3, axis=-1)
    mask_rgb = colorize_mask(mask)
    foreground = mask > 0
    out = image_rgb.copy()
    out[foreground] = (1 - alpha) * image_rgb[foreground] + alpha * mask_rgb[foreground]
    return np.clip(out, 0, 1)


def error_map(pred, gt):
    err = np.zeros((*gt.shape, 3), dtype=np.float32)
    fg_pred = pred > 0
    fg_gt = gt > 0
    false_positive = np.logical_and(fg_pred, ~fg_gt)
    false_negative = np.logical_and(~fg_pred, fg_gt)
    wrong_class = np.logical_and(fg_pred, fg_gt) & (pred != gt)
    err[false_positive] = [1.0, 0.2, 0.2]
    err[false_negative] = [0.2, 0.45, 1.0]
    err[wrong_class] = [1.0, 0.85, 0.15]
    return err


def choose_slice(label, samatch_pred, gate_pred):
    samatch_error = samatch_pred != label
    gate_error = gate_pred != label
    improvement = np.logical_and(samatch_error, ~gate_error).sum(axis=(1, 2))
    if improvement.max() > 0:
        return int(improvement.argmax())
    foreground = (label > 0).sum(axis=(1, 2))
    return int(foreground.argmax())


def save_case_figure(case, image, label, preds, out_path):
    slice_idx = choose_slice(label, preds["SAMatch"], preds["Gate"])
    image_slice = image[slice_idx]
    label_slice = label[slice_idx].astype(np.uint8)

    panels = [
        ("Image", np.repeat(image_slice[..., None], 3, axis=-1)),
        ("GT", overlay_mask(image_slice, label_slice)),
        ("UniMatch", overlay_mask(image_slice, preds["UniMatch"][slice_idx])),
        ("SAMatch", overlay_mask(image_slice, preds["SAMatch"][slice_idx])),
        ("Gate", overlay_mask(image_slice, preds["Gate"][slice_idx])),
        ("SAMatch error", error_map(preds["SAMatch"][slice_idx], label_slice)),
        ("Gate error", error_map(preds["Gate"][slice_idx], label_slice)),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{case} | slice {slice_idx}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.root_path, "test.list"), "r") as f:
        cases = sorted([line.strip().split(".")[0] for line in f if line.strip()])

    models = {
        "UniMatch": load_model(args.unimatch_ckpt, args),
        "SAMatch": load_model(args.samatch_ckpt, args),
        "Gate": load_model(args.gate_ckpt, args),
    }

    records = []
    cached = {}
    for case in cases:
        with h5py.File(os.path.join(args.root_path, "volumes", f"{case}.h5"), "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:].astype(np.uint8)
        image = normalize_image(image)

        preds = {name: predict_volume(model, image, args) for name, model in models.items()}
        dice = {name: mean_foreground_dice(pred, label, args.num_classes) for name, pred in preds.items()}
        gate_gain = dice["Gate"] - dice["SAMatch"]
        records.append((gate_gain, case, dice))
        cached[case] = (image, label, preds)
        print(
            f"{case}: UniMatch={dice['UniMatch']:.4f}, "
            f"SAMatch={dice['SAMatch']:.4f}, Gate={dice['Gate']:.4f}, "
            f"Gate-SAMatch={gate_gain:+.4f}"
        )

    records.sort(reverse=True, key=lambda item: item[0])
    selected = records[: args.top_k]

    summary_path = os.path.join(args.out_dir, "selected_cases.txt")
    with open(summary_path, "w") as f:
        f.write("gate_gain\tcase\tunimatch_dice\tsamatch_dice\tgate_dice\n")
        for gate_gain, case, dice in records:
            f.write(
                f"{gate_gain:.6f}\t{case}\t{dice['UniMatch']:.6f}\t"
                f"{dice['SAMatch']:.6f}\t{dice['Gate']:.6f}\n"
            )

    for rank, (gate_gain, case, dice) in enumerate(selected, start=1):
        image, label, preds = cached[case]
        out_path = os.path.join(args.out_dir, f"rank{rank:02d}_{case}_gain_{gate_gain:+.3f}.png")
        save_case_figure(case, image, label, preds, out_path)

    print(f"Saved {len(selected)} figures to {args.out_dir}")
    print(f"Saved ranking to {summary_path}")


if __name__ == "__main__":
    main()
