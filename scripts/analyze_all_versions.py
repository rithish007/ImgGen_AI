"""One-off cross-version SAM3 analysis. Not meant to be reusable."""
import json, glob, os
from collections import defaultdict

CLASS_NAMES = {0: 'starfish', 1: 'sea urchin', 2: 'scallop'}

# (label, images_dir, labels_dir, sidecar_glob_suffix)
DATASETS = [
    ("flux2dev_v3", "outputs/flux2dev/v3", "outputs/flux2dev/v3/labels/sam3"),
    ("flux2dev_v4", "outputs/flux2dev/v4", "outputs/flux2dev/v4/labels/sam3"),
    ("flux2dev_v5", "outputs/flux2dev/v5", "outputs/flux2dev/v5/labels/sam3"),
    ("flux2dev_v6", "outputs/flux2dev/v6", "outputs/flux2dev/v6/labels/sam3"),
    ("flux2dev_v7", "outputs/flux2dev/v7", "outputs/flux2dev/v7/labels/sam3"),
    ("hunyuan_v1_3pilot", "outputs/hunyuan/v1/3-pilot", "outputs/hunyuan/v1/3-pilot/labels/sam3"),
    ("hunyuan_v1_3pilot_promptfix", "outputs/hunyuan/v1/3-pilot_promptfix", "outputs/hunyuan/v1/3-pilot_promptfix/labels/sam3"),
    ("hunyuan_v1_5pilot", "outputs/hunyuan/v1/5-pilot", "outputs/hunyuan/v1/5-pilot/labels/sam3"),
    ("hunyuan_v2_smoke", "outputs/hunyuan/v2_smoke", "outputs/hunyuan/v2_smoke/labels/sam3"),
    ("hunyuan_v3_smoke", "outputs/hunyuan/v3_smoke", "outputs/hunyuan/v3_smoke/labels/sam3"),
    ("hunyuan_v4", "outputs/hunyuan/v4", "outputs/hunyuan/v4/labels/sam3"),
]

results = {}

for label, img_dir, lab_dir in DATASETS:
    requested_total = defaultdict(int)
    detected_total = defaultdict(int)
    n_images = 0
    n_exact = 0
    zero_req_extra = defaultdict(int)  # instances detected of a class when 0 requested
    zero_req_images = defaultdict(set)  # image ids where class was falsely detected despite 0 requested

    for jf in sorted(glob.glob(f"{img_dir}/*.json")):
        meta = json.load(open(jf, encoding='utf-8'))
        image_id = meta['image_id']
        req = {int(k): v for k, v in meta['requested_counts'].items()}
        stem = os.path.splitext(os.path.basename(jf))[0]
        label_file = os.path.join(lab_dir, f"{stem}.txt")

        det = defaultdict(int)
        if os.path.exists(label_file):
            with open(label_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    det[cid] += 1

        for cid, c in req.items():
            requested_total[cid] += c
        for cid, c in det.items():
            detected_total[cid] += c
            if cid not in req:
                zero_req_extra[cid] += c
                zero_req_images[cid].add(image_id)

        match = all(det.get(c, 0) == n for c, n in req.items()) and not any(c not in req for c in det)
        if match:
            n_exact += 1
        n_images += 1

    results[label] = {
        "n_images": n_images,
        "n_exact": n_exact,
        "requested": dict(requested_total),
        "detected": dict(detected_total),
        "zero_req_extra": dict(zero_req_extra),
        "zero_req_image_count": {k: len(v) for k, v in zero_req_images.items()},
    }

# Print master table
print(f"{'dataset':<30} {'n':>3} {'exact%':>7} | {'sf_req':>6}{'sf_det':>7}{'sf_r':>6} | {'ur_req':>6}{'ur_det':>7}{'ur_r':>6} | {'sc_req':>6}{'sc_det':>7}{'sc_r':>6}  sc_leak_imgs")
for label, r in results.items():
    n = r['n_images']
    exact_pct = r['n_exact'] / n * 100 if n else 0
    row = [f"{label:<30} {n:>3} {exact_pct:>6.0f}%"]
    for cid in (0, 1, 2):
        req = r['requested'].get(cid, 0)
        det = r['detected'].get(cid, 0)
        ratio = det / req if req else float('inf') if det else 0.0
        ratio_str = f"{ratio:.2f}" if req else ("inf" if det else "n/a")
        row.append(f" | {req:>6}{det:>7}{ratio_str:>6}")
    leak_imgs = r['zero_req_image_count'].get(2, 0)
    row.append(f"  {leak_imgs}/{n}")
    print("".join(row))

with open("reports/cross_version_analysis.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nsaved -> reports/cross_version_analysis.json")
