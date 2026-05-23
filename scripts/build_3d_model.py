"""
M2 のアウトプット(floorplan_*.json)から 3D メッシュを生成

入力: outputs/floorplan_json/floorplan_*.json
出力: outputs/3d_models/house_*.glb
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.validation import make_valid


# ============================================================
# 設定
# ============================================================
INPUT_DIR = Path("outputs/floorplan_json")
OUTPUT_DIR = Path("outputs/3d_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3D 化のパラメータ
WALL_HEIGHT = 250        # 壁の高さ (px 単位、後でスケール調整)
FLOOR_THICKNESS = 5      # 床の厚み
WALL_THICKNESS = 8       # 壁の厚み(線分から立方体を作るときの幅)

# 色(RGBA, 0-255)
COLOR_FLOOR = [200, 200, 220, 255]
COLOR_WALL = [150, 150, 150, 200]
COLOR_DEVICES = {
    "door":      [100, 100, 200, 255],
    "window":    [150, 200, 250, 180],
    "shower":    [200, 100, 200, 255],
    "sink":      [200, 150, 50, 255],
    "staircase": [200, 200, 100, 255],
    "toilet":    [100, 50, 100, 255],
}


# ============================================================
# 2D ポリゴンを 3D 化する関数
# ============================================================
def polygon_2d_to_3d_room(
    polygon_2d: list[tuple[int, int]],
    image_height: int,
    floor_z: float = 0,
    wall_height: float = WALL_HEIGHT,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh] | None:
    """1つの部屋ポリゴンから「床」と「壁」のメッシュを作る
    
    Returns:
        (floor_mesh, wall_mesh) または None(失敗時)
    """
    # 2D 座標を 3D 用に変換
    # 画像座標(y下向き)→ 世界座標(y上向き)に変換するため、y を反転
    pts_2d = [(x, image_height - y) for x, y in polygon_2d]
    
    # Shapely Polygon を作って validity を確認
    try:
        poly = Polygon(pts_2d)
        if not poly.is_valid:
            # 自己交差している場合は修正
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                # 一番大きい部分を採用
                poly = max(poly.geoms, key=lambda p: p.area)
            elif poly.geom_type != "Polygon":
                return None
        if poly.area < 100:  # あまりに小さい部屋は無視
            return None
    except Exception as e:
        print(f"    Shapely 失敗: {e}")
        return None
    
    # 床(平面)
    try:
        # 平らなポリゴンを z=floor_z の平面に作る
        # trimesh.creation.extrude_polygon は2D Polygon を z方向に押し出す
        floor = trimesh.creation.extrude_polygon(poly, height=FLOOR_THICKNESS)
        floor.apply_translation([0, 0, floor_z - FLOOR_THICKNESS])
        floor.visual.face_colors = COLOR_FLOOR
    except Exception as e:
        print(f"    床メッシュ生成失敗: {e}")
        return None
    
    # 壁(部屋の外周を押し出し)
    try:
        # ポリゴンの外側だけ(穴がない単純な多角形)
        # 壁は「ポリゴンの外周線を太らせた領域」を押し出す
        outer = poly.exterior
        # buffer で外周を太らせて、内側をくり抜く
        wall_strip = outer.buffer(WALL_THICKNESS / 2, cap_style=2, join_style=2)
        wall_2d = wall_strip.difference(poly.buffer(-WALL_THICKNESS / 2))
        if wall_2d.geom_type == "MultiPolygon":
            wall_2d = max(wall_2d.geoms, key=lambda p: p.area)
        if wall_2d.area > 50:
            walls = trimesh.creation.extrude_polygon(wall_2d, height=wall_height)
            walls.apply_translation([0, 0, floor_z])
            walls.visual.face_colors = COLOR_WALL
        else:
            walls = None
    except Exception as e:
        print(f"    壁メッシュ生成失敗: {e}")
        walls = None
    
    return floor, walls


def make_device_box(
    bbox: list[int],
    image_height: int,
    device_class: str,
    floor_z: float = 0,
) -> trimesh.Trimesh | None:
    """設備をシンプルなボックスで表現"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h_2d = y2 - y1
    cx = (x1 + x2) / 2
    cy = image_height - (y1 + y2) / 2  # y 反転
    
    # 設備ごとの高さ
    heights = {
        "toilet": 50, "sink": 80, "shower": 200, "staircase": 100,
        "door": 200, "window": 100,
    }
    height_3d = heights.get(device_class, 50)
    
    # door / window は壁に貼り付ける薄い形にする
    z_offset = 0
    if device_class == "window":
        z_offset = 100  # 窓は床から 100 上に
    
    try:
        box = trimesh.creation.box(extents=[w, h_2d, height_3d])
        box.apply_translation([cx, cy, floor_z + height_3d / 2 + z_offset])
        color = COLOR_DEVICES.get(device_class, [128, 128, 128, 255])
        box.visual.face_colors = color
        return box
    except Exception as e:
        return None


# ============================================================
# メイン処理
# ============================================================
json_files = sorted(INPUT_DIR.glob("floorplan_*.json"))
if not json_files:
    print(f"❌ {INPUT_DIR}/ に floorplan JSON がありません。")
    sys.exit(1)

for json_path in json_files:
    print(f"\n{'='*60}")
    print(f"処理中: {json_path.name}")
    print(f"{'='*60}")
    
    with open(json_path) as f:
        data = json.load(f)
    
    image_h = data["image"]["height"]
    rooms = data["rooms"]
    devices = data["devices"]
    print(f"  部屋: {len(rooms)} 個, 設備: {len(devices)} 個")
    
    scene = trimesh.Scene()
    
    # ============================================================
    # 部屋を 3D 化
    # ============================================================
    n_floors, n_walls = 0, 0
    for room in rooms:
        result = polygon_2d_to_3d_room(
            [tuple(p) for p in room["polygon"]],
            image_height=image_h,
        )
        if result is None:
            print(f"  R{room['room_id']}: スキップ(ポリゴン無効または小さすぎ)")
            continue
        floor, walls = result
        scene.add_geometry(floor, node_name=f"floor_R{room['room_id']}")
        n_floors += 1
        if walls is not None:
            scene.add_geometry(walls, node_name=f"walls_R{room['room_id']}")
            n_walls += 1
    
    print(f"  3D 床: {n_floors}, 3D 壁: {n_walls}")
    
    # ============================================================
    # 設備を 3D 化
    # ============================================================
    n_devices = 0
    for dev in devices:
        box = make_device_box(
            dev["bbox"], image_h, dev["class"]
        )
        if box is not None:
            scene.add_geometry(box, node_name=f"device_{dev['class']}_{n_devices}")
            n_devices += 1
    
    print(f"  3D 設備: {n_devices}")
    
    # ============================================================
    # GLB エクスポート
    # ============================================================
    idx = json_path.stem.split("_")[1]
    glb_path = OUTPUT_DIR / f"house_{idx}.glb"
    scene.export(glb_path)
    print(f"  ✅ 出力: {glb_path}")

print(f"\n\n{'='*60}")
print(f"✅ 完了。{OUTPUT_DIR}/ に GLB ファイルが生成されました。")
print(f"   コマンド: open {OUTPUT_DIR}")
print()
print("確認方法(3つの選択肢):")
print("  A) Mac の Finder でダブルクリック → Preview で開く(回転不可)")
print("  B) ブラウザで https://gltf-viewer.donmccurdy.com/ にアップロード(推奨)")
print("  C) Python で確認: python -c 'import trimesh; trimesh.load(\"outputs/3d_models/house_1.glb\").show()'")
