"""
floorplan JSON → 3D メッシュ生成(設備をリッチな形状で表現)
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.validation import make_valid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 配色:建築パース風
# ============================================================
COLOR_FLOOR     = [235, 220, 195, 255]
COLOR_WALL      = [248, 246, 240, 255]
COLOR_TOILET_TANK   = [245, 245, 245, 255]
COLOR_TOILET_SEAT   = [255, 255, 255, 255]
COLOR_SINK_BASE     = [220, 220, 220, 255]
COLOR_SINK_BOWL     = [255, 255, 255, 255]
COLOR_SHOWER_TRAY   = [180, 190, 210, 255]
COLOR_SHOWER_HEAD   = [200, 200, 200, 255]
COLOR_STAIRCASE     = [160, 130, 100, 255]
COLOR_DOOR          = [120,  85,  70, 255]
COLOR_DOOR_HANDLE   = [255, 215,   0, 255]
COLOR_WINDOW_FRAME  = [140, 130, 110, 255]
COLOR_WINDOW_GLASS  = [180, 220, 235, 130]  # 半透明
COLOR_DEFAULT       = [170, 170, 170, 255]

# 3D パラメータ
WALL_HEIGHT = 250
FLOOR_THICKNESS = 5
WALL_THICKNESS = 8


# ============================================================
# ヘルパー:小さい箱を作る
# ============================================================
def _box(extents, position, color):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(position)
    box.visual.face_colors = color
    return box


def _cylinder(radius, height, position, color, sections=20):
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cyl.apply_translation(position)
    cyl.visual.face_colors = color
    return cyl


# ============================================================
# 部屋(床 + 壁)
# ============================================================
def _polygon_2d_to_3d_room(polygon_2d, image_height, floor_z=0, wall_height=WALL_HEIGHT):
    pts_2d = [(x, image_height - y) for x, y in polygon_2d]
    try:
        poly = Polygon(pts_2d)
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            elif poly.geom_type != "Polygon":
                return None
        if poly.area < 100:
            return None
    except Exception:
        return None
    
    # 床
    try:
        floor = trimesh.creation.extrude_polygon(poly, height=FLOOR_THICKNESS)
        floor.apply_translation([0, 0, floor_z - FLOOR_THICKNESS])
        floor.visual.face_colors = COLOR_FLOOR
    except Exception:
        return None
    
    # 壁
    walls = None
    try:
        outer = poly.exterior
        wall_strip = outer.buffer(WALL_THICKNESS / 2, cap_style=2, join_style=2)
        wall_2d = wall_strip.difference(poly.buffer(-WALL_THICKNESS / 2))
        if wall_2d.geom_type == "MultiPolygon":
            wall_2d = max(wall_2d.geoms, key=lambda p: p.area)
        if wall_2d.area > 50:
            walls = trimesh.creation.extrude_polygon(wall_2d, height=wall_height)
            walls.apply_translation([0, 0, floor_z])
            walls.visual.face_colors = COLOR_WALL
    except Exception:
        walls = None
    
    return floor, walls


# ============================================================
# 設備の 3D 形状(複数のメッシュを組み合わせ)
# ============================================================
def _build_toilet(bbox, image_height, floor_z=0):
    """トイレ:タンク + 便座"""
    x1, y1, x2, y2 = bbox
    w, d = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = image_height - (y1 + y2) / 2
    
    parts = []
    # タンク(背中側、奥1/3の幅)
    tank_w, tank_d, tank_h = w * 0.9, d * 0.4, 50
    parts.append(_box(
        [tank_w, tank_d, tank_h],
        [cx, cy - d * 0.3, floor_z + tank_h / 2],
        COLOR_TOILET_TANK,
    ))
    # 便座(前側、低い)
    seat_w, seat_d, seat_h = w * 0.9, d * 0.65, 20
    parts.append(_box(
        [seat_w, seat_d, seat_h],
        [cx, cy + d * 0.05, floor_z + seat_h / 2],
        COLOR_TOILET_SEAT,
    ))
    # 便器底(直方体で簡略表現)
    base_h = 40
    parts.append(_box(
        [seat_w * 0.7, seat_d * 0.7, base_h],
        [cx, cy + d * 0.05, floor_z + seat_h + base_h / 2 - 20],
        COLOR_TOILET_SEAT,
    ))
    return trimesh.util.concatenate(parts)


def _build_sink(bbox, image_height, floor_z=0):
    """シンク:台 + 水盤"""
    x1, y1, x2, y2 = bbox
    w, d = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = image_height - (y1 + y2) / 2
    
    parts = []
    # 台
    base_h = 70
    parts.append(_box(
        [w, d, base_h],
        [cx, cy, floor_z + base_h / 2],
        COLOR_SINK_BASE,
    ))
    # 水盤(白く、上に少し小さい)
    bowl_h = 10
    parts.append(_box(
        [w * 0.8, d * 0.8, bowl_h],
        [cx, cy, floor_z + base_h + bowl_h / 2],
        COLOR_SINK_BOWL,
    ))
    return trimesh.util.concatenate(parts)


def _build_shower(bbox, image_height, floor_z=0):
    """シャワー:トレイ + 柱 + ヘッド"""
    x1, y1, x2, y2 = bbox
    w, d = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = image_height - (y1 + y2) / 2
    # 小さい bbox は最低サイズに引き上げる
    w = max(w, 40)
    d = max(d, 40)
    
    parts = []
    # 床トレイ
    parts.append(_box(
        [w, d, 5],
        [cx, cy, floor_z + 2.5],
        COLOR_SHOWER_TRAY,
    ))
    # シャワー柱(背中側)
    parts.append(_cylinder(
        radius=3,
        height=210,
        position=[cx, cy - d / 2 + 5, floor_z + 105],
        color=COLOR_SHOWER_HEAD,
    ))
    # シャワーヘッド(上部、前方に傾けた感じで横の円柱)
    head = trimesh.creation.cylinder(radius=12, height=8, sections=16)
    head.apply_translation([cx, cy - d / 2 + 15, floor_z + 200])
    head.visual.face_colors = COLOR_SHOWER_HEAD
    parts.append(head)
    return trimesh.util.concatenate(parts)


def _build_staircase(bbox, image_height, floor_z=0):
    """階段:5段の段差"""
    x1, y1, x2, y2 = bbox
    w, d = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = image_height - (y1 + y2) / 2
    
    n_steps = 5
    step_height = 25
    step_depth = d / n_steps
    
    parts = []
    for i in range(n_steps):
        step_h = step_height * (i + 1)
        parts.append(_box(
            [w, step_depth, step_h],
            [cx, cy - d / 2 + step_depth * (i + 0.5), floor_z + step_h / 2],
            COLOR_STAIRCASE,
        ))
    return trimesh.util.concatenate(parts)


def _build_door(bbox, image_height, floor_z=0):
    """ドア:ドア板 + 取っ手"""
    x1, y1, x2, y2 = bbox
    w, d = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = image_height - (y1 + y2) / 2
    
    parts = []
    # 短辺方向を厚みに(ドアは「板」)
    if w < d:
        thickness, length = w, d
        ext = [thickness, length, 200]
    else:
        thickness, length = d, w
        ext = [length, thickness, 200]
    parts.append(_box(
        ext, [cx, cy, floor_z + 100], COLOR_DOOR
    ))
    # 取っ手
    parts.append(trimesh.creation.icosphere(radius=4, subdivisions=2))
    handle = parts[-1]
    handle.apply_translation([cx + min(w, d) / 2, cy, floor_z + 95])
    handle.visual.face_colors = COLOR_DOOR_HANDLE
    return trimesh.util.concatenate(parts)


def _build_window(bbox, image_height, floor_z=0):
    """窓:枠 + ガラス"""
    x1, y1, x2, y2 = bbox
    w, d = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = image_height - (y1 + y2) / 2
    
    # 窓は壁に貼り付ける薄いプレート
    parts = []
    if w < d:
        thickness, length = w, d
        frame_ext = [thickness, length, 90]
        glass_ext = [thickness * 1.05, length * 0.85, 80]
    else:
        thickness, length = d, w
        frame_ext = [length, thickness, 90]
        glass_ext = [length * 0.85, thickness * 1.05, 80]
    
    parts.append(_box(
        frame_ext, [cx, cy, floor_z + 145], COLOR_WINDOW_FRAME
    ))
    # ガラス部分(枠の中)
    parts.append(_box(
        glass_ext, [cx, cy, floor_z + 145], COLOR_WINDOW_GLASS
    ))
    return trimesh.util.concatenate(parts)


# 設備ビルダーのマップ
DEVICE_BUILDERS = {
    "toilet":    _build_toilet,
    "sink":      _build_sink,
    "shower":    _build_shower,
    "staircase": _build_staircase,
    "door":      _build_door,
    "window":    _build_window,
}


def _build_device(bbox, image_height, device_class, floor_z=0):
    builder = DEVICE_BUILDERS.get(device_class)
    if builder is None:
        return None
    try:
        return builder(bbox, image_height, floor_z)
    except Exception as e:
        print(f"  ⚠️ {device_class} 生成失敗: {e}")
        return None


# ============================================================
# シーン構築(主要 API)
# ============================================================
def build_scene_from_floorplan(floorplan_data):
    image_h = floorplan_data["image"]["height"]
    rooms = floorplan_data["rooms"]
    devices = floorplan_data["devices"]
    
    scene = trimesh.Scene()
    stats = {"n_floors": 0, "n_walls": 0, "n_devices": 0, "n_rooms_skipped": 0}
    
    for room in rooms:
        result = _polygon_2d_to_3d_room(
            [tuple(p) for p in room["polygon"]],
            image_height=image_h,
        )
        if result is None:
            stats["n_rooms_skipped"] += 1
            continue
        floor, walls = result
        scene.add_geometry(floor, node_name=f"floor_R{room['room_id']}")
        stats["n_floors"] += 1
        if walls is not None:
            scene.add_geometry(walls, node_name=f"walls_R{room['room_id']}")
            stats["n_walls"] += 1
    
    for i, dev in enumerate(devices):
        mesh = _build_device(dev["bbox"], image_h, dev["class"])
        if mesh is not None:
            scene.add_geometry(mesh, node_name=f"device_{dev['class']}_{i}")
            stats["n_devices"] += 1
    
    return scene, stats


def export_scene_to_glb_bytes(scene):
    buf = BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


# ============================================================
# スタンドアロン実行
# ============================================================
if __name__ == "__main__":
    INPUT_DIR = Path("outputs/floorplan_json")
    OUTPUT_DIR = Path("outputs/3d_models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(INPUT_DIR.glob("floorplan_*.json"))
    if not json_files:
        print(f"❌ {INPUT_DIR}/ に floorplan JSON がありません。")
        sys.exit(1)
    
    for json_path in json_files:
        print(f"\n=== {json_path.name} ===")
        with open(json_path) as f:
            data = json.load(f)
        
        scene, stats = build_scene_from_floorplan(data)
        print(f"  床: {stats['n_floors']}, 壁: {stats['n_walls']}, 設備: {stats['n_devices']}")
        if stats["n_rooms_skipped"] > 0:
            print(f"  ⚠️ スキップ部屋: {stats['n_rooms_skipped']}")
        
        idx = json_path.stem.split("_")[1]
        glb_path = OUTPUT_DIR / f"house_{idx}.glb"
        scene.export(glb_path)
        print(f"  ✅ 出力: {glb_path}")
    
    print(f"\n✅ 完了: {OUTPUT_DIR}/")
