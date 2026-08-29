"""3x2 montage: (GT, Predicted) pairs for one representative real DUO image per
class, using the best-performing model (v9_flux2dev_duo_dr, duo_calibrated DR).
Images were hand-picked by grepping GT and predicted label files for cases
where the model actually predicts that class AND ground truth agrees.

    python -m imggen.analysis.plot_gt_vs_pred_montage
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

CLASS_NAMES = {0: "starfish", 1: "sea_urchin", 2: "scallop"}
GT_COLOR = (0, 90, 143)      # theme deep blue
PRED_COLOR = (255, 150, 100)  # theme orange

GT_LABELS = "dataset/real_eval/labels"
GT_IMAGES = "dataset/real_eval/images"
PRED_LABELS = "runs/predict_duo/v9_flux2dev_duo_dr_detections/labels"

# (class_id, image_stem) - hand-picked, see script docstring
ROWS = [
    (0, "1896_jpg.rf.34a5f908bc1e883d650bbf53eac05b7d"),
    (1, "1017_jpg.rf.fef4ca4fbb0d2497e95d47cfc668569f"),
    (2, "1929_jpg.rf.2b41f4276b027d74389d4e42f9f2bd88"),
]


def load_boxes(label_path: str, class_id: int, with_conf: bool = False):
    boxes = []
    try:
        lines = open(label_path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return boxes
    for line in lines:
        parts = line.split()
        if not parts or int(parts[0]) != class_id:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if with_conf and len(parts) > 5 else None
        boxes.append((cx, cy, w, h, conf))
    return boxes


def draw_boxes(image_path: str, boxes, color, class_name: str) -> Image.Image:
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for cx, cy, bw, bh, conf in boxes:
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = class_name if conf is None else f"{class_name} {conf:.2f}"
        draw.text((x1 + 3, max(0, y1 - 16)), label, fill=color)
    return im


fig, axes = plt.subplots(3, 2, figsize=(9, 13), dpi=150)

for row, (class_id, stem) in enumerate(ROWS):
    class_name = CLASS_NAMES[class_id]
    image_path = f"{GT_IMAGES}/{stem}.jpg"

    gt_boxes = load_boxes(f"{GT_LABELS}/{stem}.txt", class_id, with_conf=False)
    pred_boxes = load_boxes(f"{PRED_LABELS}/{stem}.txt", class_id, with_conf=True)

    gt_im = draw_boxes(image_path, gt_boxes, GT_COLOR, class_name)
    pred_im = draw_boxes(image_path, pred_boxes, PRED_COLOR, class_name)

    axes[row][0].imshow(gt_im)
    axes[row][0].set_title(f"GT {class_name}  (n={len(gt_boxes)})", fontsize=12)
    axes[row][1].imshow(pred_im)
    axes[row][1].set_title(f"Predicted {class_name}  (n={len(pred_boxes)})", fontsize=12)
    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle("Ground Truth vs. Predicted — v9_flux2dev_duo_dr (duo_calibrated) on real DUO test images", fontsize=14)
fig.tight_layout()
out_path = "assets/gt_vs_pred_montage_matplotlib.png"
fig.savefig(out_path)
print(f"saved -> {out_path}")
