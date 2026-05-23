"""
セグメンテーション学習用データ準備

入力: data/roboflow/floor-plan-annotation-1/
出力: data/floorplan_seg/

物体検出用(scripts/prepare_dataset.py)との違い:
- polygon 形式を維持(bbox 変換しない)
- room, wall を含む(物体検出版では除外していた)
- bathtub は引き続き除外(サンプル数極少)
"""
import random
import shutil
from pathlib import Path

import yaml

# ============================================================
# 設定
# ============================================================
SRC_ROOT = Path("data/roboflow/floor-plan-annotation-1")
DST_ROOT = Path("data/floorplan_seg")

# 元クラスID -> 新クラスID(None なら除外)
# 元: 0:bathtub, 1:door, 2:room, 3:shower, 4:sink, 5:staircase, 6:toilet, 7:wall, 8:window
CLASS_MAP = {
    0: None,  # bathtub  → 除外(サンプル数極少)
    1: 0,     # door
    2: 1,     # room      ← 復活(Phase 1+ で追加)
    3: 2,     # shower
    4: 3,     # sink
    5: 4,     # staircase
    6: 5,     # toilet
    7: 6,     # wall      ← 復活(Phase 1+ で追加)
    8: 7,     # window
}

NEW_CLASS_NAMES = [
    "door", "room", "shower", "sink", "staircase", "toilet", "wall", "window"
]

SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# ============================================================
# 出力先を初期化
# ============================================================
if DST_ROOT.exists():
    print(f"既存の {DST_ROOT} を削除します")
    shutil.rmtree(DST_ROOT)

for split in ["train", "val", "test"]:
    (DST_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

# ============================================================
# 元データの全画像をリストアップ(train + valid をマージ)
# ============================================================
all_images = []
for split in ["train", "valid"]:
    img_dir = SRC_ROOT / split / "images"
    if img_dir.exists():
        all_images.extend(sorted(img_dir.glob("*.jpg")))

print(f"元データ全画像数: {len(all_images)}")

random.seed(RANDOM_SEED)
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(n_total * SPLIT_RATIO["train"])
n_val = int(n_total * SPLIT_RATIO["val"])

splits = {
    "train": all_images[:n_train],
    "val": all_images[n_train:n_train + n_val],
    "test": all_images[n_train + n_val:],
}

# ============================================================
# 各画像とラベルを処理(polygon 形式は維持)
# ============================================================
stats = {split: {"images": 0, "labels": 0, "boxes_per_class": {}} for split in splits}

for split, image_list in splits.items():
    for img_path in image_list:
        src_split = img_path.parent.parent.name
        label_path = SRC_ROOT / src_split / "labels" / (img_path.stem + ".txt")
        
        if not label_path.exists():
            continue
        
        filtered_lines = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = CLASS_MAP.get(old_cls)
                if new_cls is None:
                    continue
                # 座標部分はそのまま(polygon 形式を維持)
                new_line = f"{new_cls} " + " ".join(parts[1:])
                filtered_lines.append(new_line)
                stats[split]["boxes_per_class"][new_cls] = \
                    stats[split]["boxes_per_class"].get(new_cls, 0) + 1
        
        if not filtered_lines:
            continue
        
        # 画像をコピー
        dst_img = DST_ROOT / split / "images" / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # ラベルを保存
        dst_label = DST_ROOT / split / "labels" / (img_path.stem + ".txt")
        with open(dst_label, "w") as f:
            f.write("\n".join(filtered_lines) + "\n")
        
        stats[split]["images"] += 1
        stats[split]["labels"] += len(filtered_lines)

# ============================================================
# data.yaml を生成
# ============================================================
yaml_data = {
    "path": str(DST_ROOT.resolve()),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": NEW_CLASS_NAMES,
    "nc": len(NEW_CLASS_NAMES),
}

yaml_path = DST_ROOT / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

# ============================================================
# 結果サマリ
# ============================================================
print()
print("=" * 60)
print("再分割結果(セグメンテーション用)")
print("=" * 60)
for split, info in stats.items():
    print(f"\n[{split}]")
    print(f"  画像: {info['images']} 枚")
    print(f"  polygon インスタンス合計: {info['labels']}")
    print(f"  クラス別:")
    for cls_id in range(len(NEW_CLASS_NAMES)):
        count = info["boxes_per_class"].get(cls_id, 0)
        print(f"    {cls_id} {NEW_CLASS_NAMES[cls_id]:12s}: {count:5d}")

print()
print(f"✅ 完了: {DST_ROOT}/")
print(f"   設定ファイル: {yaml_path}")
