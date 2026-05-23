"""
Floor Plan Detector - Streamlit Web UI

起動: streamlit run src/app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# プロジェクトルートを sys.path に追加(src/inference.py を import するため)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FloorPlanDetector, DetectionResult


# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="Floor Plan Detector",
    page_icon="📐",
    layout="wide",
)


# ============================================================
# 利用可能なモデル
# ============================================================
AVAILABLE_MODELS = {
    "Best (Exp 2: YOLOv8n / 1024 / 100ep, mAP=0.966)": "models/exp2_long_yolov8n/weights/best.pt",
    "Exp 1 (YOLOv8n / 1024 / 50ep, mAP=0.903)":         "models/exp1_highres_yolov8n/weights/best.pt",
    "Baseline (YOLOv8n / 640 / 50ep, mAP=0.815)":       "models/baseline_yolov8n/weights/best.pt",
    "Exp 3 (YOLO26n / 1024 / 100ep, mAP=0.867)":        "models/exp3_yolo26n/weights/best.pt",
}

CLASS_NAMES = ["door", "shower", "sink", "staircase", "toilet", "window"]


# ============================================================
# キャッシュされたモデルローダー
# Streamlit のキャッシュ機能でモデルを1回だけロードする
# ============================================================
@st.cache_resource
def load_detector(model_path: str) -> FloorPlanDetector:
    """モデルをロードしてキャッシュ"""
    return FloorPlanDetector(model_path, warmup=True)


# ============================================================
# サンプル画像取得
# ============================================================
@st.cache_data
def get_sample_images() -> list[Path]:
    """test セットから数枚のサンプルを取得"""
    test_dir = PROJECT_ROOT / "data" / "floorplan_yolo" / "test" / "images"
    if not test_dir.exists():
        return []
    samples = sorted(test_dir.glob("*.jpg"))[:6]
    return samples


# ============================================================
# サイドバー(設定)
# ============================================================
with st.sidebar:
    st.title("⚙️ 設定")
    
    # モデル選択
    st.subheader("モデル")
    model_label = st.selectbox(
        "使用するモデル",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="ベスト(Exp 2)を使うのが推奨"
    )
    model_path = AVAILABLE_MODELS[model_label]
    
    # 推論ハイパラ
    st.subheader("推論パラメータ")
    conf_threshold = st.slider(
        "信頼度しきい値", min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="これ未満の信頼度の検出は破棄される"
    )
    iou_threshold = st.slider(
        "IoU しきい値 (NMS)", min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        help="bbox 同士がこれ以上重なると、1つに統合される(値が低いほど重複検出が減る)"
    )
    imgsz = st.selectbox(
        "推論サイズ", options=[640, 1024, 1280], index=1,
        help="大きいほど小物体検出に有利だが、推論が遅くなる"
    )
    
    # 表示クラス
    st.subheader("表示クラス")
    classes_to_show = []
    for cls in CLASS_NAMES:
        if st.checkbox(cls, value=True, key=f"show_{cls}"):
            classes_to_show.append(cls)
    
    st.markdown("---")
    st.markdown("**📊 Phase 3 ベストモデル**")
    st.markdown("- mAP@0.5: **0.966**")
    st.markdown("- 全6クラスで mAP ≥ 0.90")
    st.caption("詳細は [GitHub README](https://github.com/Mao925/floor-plan-recognition) 参照")


# ============================================================
# メインエリア
# ============================================================
st.title("📐 Floor Plan Detector")
st.markdown(
    "間取り図から設備記号(door, window, shower, sink, staircase, toilet)を検出するシステム。"
    "Phase 3 で達成した **mAP@0.5 = 0.966** のモデルを使用。"
)

# --- 画像入力 ---
st.subheader("1️⃣ 画像を選択")
input_method = st.radio(
    "入力方法",
    options=["📁 ファイルをアップロード", "🖼️ サンプル画像を使う"],
    horizontal=True,
)

uploaded_image: np.ndarray | None = None  # BGR

if input_method == "📁 ファイルをアップロード":
    uploaded_file = st.file_uploader(
        "間取り図画像を選択 (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        rgb_array = np.array(pil_image)
        uploaded_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
else:
    samples = get_sample_images()
    if samples:
        sample_names = [s.name[:30] + "..." for s in samples]
        sample_idx = st.selectbox(
            "サンプル画像",
            options=range(len(samples)),
            format_func=lambda i: f"Sample {i+1}: {sample_names[i]}",
        )
        uploaded_image = cv2.imread(str(samples[sample_idx]))
    else:
        st.warning("サンプル画像が見つかりません。データセットを準備してください。")


# --- 推論実行 ---
if uploaded_image is not None:
    st.subheader("2️⃣ 検出実行")
    
    if st.button("🚀 検出開始", type="primary"):
        with st.spinner("推論中..."):
            detector = load_detector(model_path)
            result = detector.predict(
                uploaded_image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
            )
            
            annotated = detector.visualize(
                uploaded_image,
                result,
                classes_to_show=classes_to_show,
            )
        
        # 結果をセッションに保存(再描画時のため)
        st.session_state["result"] = result
        st.session_state["annotated"] = annotated
        st.session_state["original"] = uploaded_image
    
    # --- 結果表示 ---
    if "result" in st.session_state:
        result: DetectionResult = st.session_state["result"]
        annotated = st.session_state["annotated"]
        original = st.session_state["original"]
        
        st.subheader("3️⃣ 検出結果")
        
        # メトリクス
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総検出数", len(result.detections))
        col2.metric("画像サイズ", f"{result.image_width}×{result.image_height}")
        col3.metric("推論時間", f"{result.inference_time_ms:.0f} ms")
        col4.metric("モデル", result.model_name)
        
        # 元画像 vs 結果画像
        st.markdown("#### 検出結果(元画像 vs アノテーション)")
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="元画像", use_container_width=True)
        with c2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="検出結果", use_container_width=True)
        
        # クラス別の検出数
        st.markdown("#### クラス別検出数")
        counts = result.class_counts
        all_counts = {cls: counts.get(cls, 0) for cls in CLASS_NAMES}
        st.bar_chart(pd.DataFrame.from_dict(all_counts, orient="index", columns=["count"]))
        
        # 検出物テーブル
        st.markdown("#### 検出物リスト")
        if result.detections:
            df = pd.DataFrame([
                {
                    "#": i + 1,
                    "class": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "x1": d.bbox_xyxy[0],
                    "y1": d.bbox_xyxy[1],
                    "x2": d.bbox_xyxy[2],
                    "y2": d.bbox_xyxy[3],
                    "width": d.width,
                    "height": d.height,
                }
                for i, d in enumerate(result.detections)
                if d.class_name in classes_to_show
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("検出物がありません。信頼度しきい値を下げてみてください。")
        
        # JSON 出力
        st.markdown("#### 4️⃣ 構造化データ出力")
        json_data = result.to_json_dict()
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.download_button(
                label="📥 JSON ダウンロード",
                data=json_str,
                file_name="detection_result.json",
                mime="application/json",
            )
        
        with st.expander("JSON プレビュー"):
            st.code(json_str, language="json")
else:
    st.info("👆 画像をアップロードまたはサンプルを選択してください")

# ============================================================
# フッター
# ============================================================
st.markdown("---")
st.caption(
    "📐 Floor Plan Detector | Phase 4-5: Viewer | "
    "[GitHub](https://github.com/Mao925/floor-plan-recognition)"
)
