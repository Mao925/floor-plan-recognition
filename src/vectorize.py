"""
ベクトル化モジュール

セグメンテーションマスクを後処理し、構造化された2D平面図データに変換する。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# データクラス
# ============================================================
@dataclass
class RoomPolygon:
    room_id: int
    polygon: list[tuple[int, int]]
    area_pixels: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    
    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "polygon": [list(p) for p in self.polygon],
            "area_pixels": self.area_pixels,
            "centroid": list(self.centroid),
            "bbox": list(self.bbox),
            "n_vertices": len(self.polygon),
        }


@dataclass
class WallSegment:
    """1本の壁線分"""
    p1: tuple[int, int]
    p2: tuple[int, int]
    length: float
    angle_deg: float   # 水平線を 0 度とした角度(-90〜90 度)
    
    def to_dict(self) -> dict:
        return {
            "p1": list(self.p1),
            "p2": list(self.p2),
            "length": round(self.length, 1),
            "angle_deg": round(self.angle_deg, 1),
        }


# ============================================================
# room マスク → ポリゴンへの変換
# ============================================================
def extract_room_polygons(
    room_mask: np.ndarray,
    min_area_ratio: float = 0.005,
    approx_epsilon_ratio: float = 0.025,
) -> list[RoomPolygon]:
    """room の二値マスクから個別の部屋ポリゴンを抽出"""
    h, w = room_mask.shape[:2]
    total_pixels = h * w
    min_area = total_pixels * min_area_ratio
    
    _, binary = cv2.threshold(room_mask, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons: list[RoomPolygon] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(cnt, closed=True)
        epsilon = approx_epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        polygon_pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = polygon_pts[0]
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        polygons.append(RoomPolygon(
            room_id=0,
            polygon=polygon_pts,
            area_pixels=int(area),
            centroid=(cx, cy),
            bbox=(x, y, x + bw, y + bh),
        ))
    
    polygons.sort(key=lambda p: p.area_pixels, reverse=True)
    for i, poly in enumerate(polygons):
        poly.room_id = i
    
    return polygons


def visualize_room_polygons(
    image: np.ndarray,
    polygons: list[RoomPolygon],
    show_id: bool = True,
    polygon_color: tuple[int, int, int] = (0, 255, 0),
    polygon_thickness: int = 2,
    text_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    img = image.copy()
    for poly in polygons:
        pts = np.array(poly.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=polygon_color, thickness=polygon_thickness)
        for x, y in poly.polygon:
            cv2.circle(img, (x, y), 4, polygon_color, -1)
        cv2.circle(img, poly.centroid, 6, (255, 0, 0), -1)
        if show_id:
            text = f"R{poly.room_id}"
            cv2.putText(
                img, text, (poly.centroid[0] - 15, poly.centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA
            )
    return img


# ============================================================
# wall マスク → 線分への変換
# ============================================================
def extract_wall_skeleton(wall_mask: np.ndarray) -> np.ndarray:
    """wall マスクを細線化して中心線を取得"""
    _, binary = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)
    # 0/1 に正規化(skimage 用)
    binary_01 = (binary > 0).astype(np.uint8)
    skeleton = skeletonize(binary_01)
    # uint8 (0/255) に戻す
    return (skeleton.astype(np.uint8)) * 255


def extract_wall_segments(
    wall_mask: np.ndarray,
    hough_threshold: int = 50,
    min_line_length: int = 30,
    max_line_gap: int = 10,
) -> list[WallSegment]:
    """wall マスクから線分を抽出
    
    Args:
        wall_mask: 二値マスク(uint8, 0 or 255)
        hough_threshold: HoughLinesP の票数しきい値(高いと検出減)
        min_line_length: 検出する最小線分長(px)
        max_line_gap: 線分の連結時に許容する隙間(px)
    Returns:
        WallSegment のリスト
    """
    skeleton = extract_wall_skeleton(wall_mask)
    
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    
    if lines is None:
        return []
    
    segments: list[WallSegment] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        # 角度(水平を0度、-90〜90)
        angle = math.degrees(math.atan2(dy, dx))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        
        segments.append(WallSegment(
            p1=(int(x1), int(y1)),
            p2=(int(x2), int(y2)),
            length=length,
            angle_deg=angle,
        ))
    
    return segments


def visualize_walls(
    image: np.ndarray,
    segments: list[WallSegment],
    show_endpoints: bool = True,
    color: tuple[int, int, int] = (0, 0, 255),  # 赤
    thickness: int = 2,
) -> np.ndarray:
    """壁線分を画像に描画"""
    img = image.copy()
    for seg in segments:
        cv2.line(img, seg.p1, seg.p2, color, thickness)
        if show_endpoints:
            cv2.circle(img, seg.p1, 3, (255, 255, 0), -1)
            cv2.circle(img, seg.p2, 3, (255, 255, 0), -1)
    return img


# ============================================================
# 動作確認(モジュール単体で実行した場合)
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    room_files = sorted(Path("outputs/seg_inference_test").glob("seg_*_room_mask.png"))
    wall_files = sorted(Path("outputs/seg_inference_test").glob("seg_*_wall_mask.png"))
    if not room_files or not wall_files:
        print("❌ outputs/seg_inference_test/ にマスクが見つかりません")
        sys.exit(1)
    
    OUTPUT_DIR = Path("outputs/vectorize_test")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for room_path, wall_path in zip(room_files, wall_files):
        idx = room_path.stem.split("_")[1]
        print(f"\n--- 画像 {idx} ---")
        
        room_mask = cv2.imread(str(room_path), cv2.IMREAD_GRAYSCALE)
        wall_mask = cv2.imread(str(wall_path), cv2.IMREAD_GRAYSCALE)
        h, w = room_mask.shape
        
        # 部屋ポリゴン抽出
        polygons = extract_room_polygons(room_mask)
        print(f"  部屋: {len(polygons)} 個")
        
        # 壁の細線化(中間結果として保存)
        skeleton = extract_wall_skeleton(wall_mask)
        cv2.imwrite(str(OUTPUT_DIR / f"walls_{idx}_skeleton.png"), skeleton)
        skel_pixels = (skeleton > 0).sum()
        print(f"  壁スケルトン: {skel_pixels} px (太さ1の中心線)")
        
        # 壁線分抽出
        segments = extract_wall_segments(wall_mask)
        print(f"  壁線分: {len(segments)} 本")
        if segments:
            lengths = [s.length for s in segments]
            print(f"    長さ: 最小={min(lengths):.0f}px, 最大={max(lengths):.0f}px, 平均={sum(lengths)/len(lengths):.0f}px")
            # 角度の分布
            angles = [s.angle_deg for s in segments]
            horizontals = sum(1 for a in angles if abs(a) < 10 or abs(a) > 170)
            verticals = sum(1 for a in angles if 80 < abs(a) < 100)
            print(f"    水平線(±10°): {horizontals} 本, 垂直線(80〜100°): {verticals} 本")
        
        # 統合可視化:room ポリゴン + wall 線分
        background = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白背景
        # 部屋を薄い水色で塗る
        for poly in polygons:
            pts = np.array(poly.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(background, [pts], (220, 240, 255))
        # 部屋ポリゴンの輪郭
        annotated = visualize_room_polygons(background, polygons, polygon_color=(0, 150, 0))
        # 壁線分を描画
        annotated = visualize_walls(annotated, segments, color=(0, 0, 200), thickness=2)
        
        out_path = OUTPUT_DIR / f"combined_{idx}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  保存: {out_path}")
    
    print(f"\n✅ 完了。{OUTPUT_DIR}/ に skeleton と combined の画像があります。")
    print(f"   コマンド: open {OUTPUT_DIR}")


# ============================================================
# 設備を部屋に紐付ける(改良版)
# ============================================================
def find_nearest_rooms(
    point: tuple[int, int],
    rooms: list,
    max_distance: float = 50.0,
    n: int = 1,
) -> list[int]:
    """点(x, y)に最も近い部屋(複数)の room_id を返す
    
    Args:
        point: 判定する点
        rooms: 部屋ポリゴンのリスト
        max_distance: これより遠い部屋は除外
        n: 取得する近い部屋の最大数
    Returns:
        room_id のリスト(距離の近い順、最大 n 個)
    """
    distances = []
    for room in rooms:
        pts = np.array(room.polygon, dtype=np.int32)
        # pointPolygonTest with measureDist=True で「符号付き距離」が返る
        # 正:内側、負:外側、絶対値が距離
        dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=True)
        # 内側なら距離0として扱う
        if dist >= 0:
            distances.append((0.0, room.room_id))
        else:
            distances.append((abs(dist), room.room_id))
    
    # 距離の近い順にソート
    distances.sort(key=lambda x: x[0])
    
    # max_distance 以下のものだけ、最大 n 個
    nearest = [room_id for dist, room_id in distances[:n] if dist <= max_distance]
    return nearest
