"""ダウンロードしたデータセットの統計情報を表示"""
from pathlib import Path
import yaml

DATA_ROOT = Path("data/roboflow/floor-plan-annotation-1")

# data.yaml の中身を確認
print("=" * 60)
print("data.yaml の内容")
print("=" * 60)
yaml_path = DATA_ROOT / "data.yaml"
with open(yaml_path) as f:
    config = yaml.safe_load(f)
for key, val in config.items():
    print(f"  {key}: {val}")

print()
print("=" * 60)
print("ファイル数の集計")
print("=" * 60)

# train / valid / test の枚数
for split in ["train", "valid", "test"]:
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        print(f"  {split}: (存在しない)")
        continue
    images = list((split_dir / "images").glob("*"))
    labels = list((split_dir / "labels").glob("*.txt"))
    print(f"  {split}: 画像 {len(images)} 枚 / ラベル {len(labels)} 枚")

print()
print("=" * 60)
print("クラスごとの出現回数(train セット)")
print("=" * 60)

# YOLOラベルの形式は: class_id center_x center_y width height (正規化済み)
class_names = config.get("names", [])
class_counts = {i: 0 for i in range(len(class_names))}

train_labels_dir = DATA_ROOT / "train" / "labels"
total_boxes = 0
for label_file in train_labels_dir.glob("*.txt"):
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            total_boxes += 1

for cls_id, name in enumerate(class_names):
    count = class_counts.get(cls_id, 0)
    bar = "█" * int(count / max(class_counts.values(), default=1) * 30)
    print(f"  {cls_id:2d} {name:15s}: {count:5d}  {bar}")

print()
print(f"  合計バウンディングボックス数: {total_boxes}")

print()
print("=" * 60)
print("ラベルファイルのサンプル(最初の1ファイル)")
print("=" * 60)
first_label = next(train_labels_dir.glob("*.txt"), None)
if first_label:
    print(f"  ファイル: {first_label.name}")
    with open(first_label) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else "?"
            # 多角形(セグメンテーション)か bbox かを判定
            num_coords = len(parts) - 1
            if num_coords == 4:
                fmt = "bbox(cx,cy,w,h)"
            elif num_coords > 4 and num_coords % 2 == 0:
                fmt = f"polygon({num_coords//2}点)"
            else:
                fmt = f"unknown({num_coords}values)"
            print(f"  行{i+1}: クラス={cls_name}({cls_id}) 形式={fmt}")
            if i >= 4:
                print(f"  ... 以下省略")
                break
