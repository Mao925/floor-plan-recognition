"""
Floor Plan 3D Viewer - Streamlit Web App

間取り図画像 → ML → ベクトル化 → 3D の家(Google <model-viewer> 採用)
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_full_pipeline


# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="Floor Plan 3D Viewer",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 100%; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3D ビューア(Google <model-viewer> Web Component)
# ============================================================
MODEL_VIEWER_TEMPLATE = """
<style>
  .mv-container {
    width: 100%;
    height: __HEIGHT__px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
    background: linear-gradient(180deg, #e6ecf4 0%, #f5f1e8 50%, #ddd1bd 100%);
    position: relative;
  }
  model-viewer {
    width: 100%;
    height: 100%;
    --poster-color: transparent;
  }
  .mv-controls {
    position: absolute;
    top: 16px;
    right: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    z-index: 10;
  }
  .mv-btn {
    padding: 10px 16px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-weight: 500;
    color: #333;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    transition: all 0.15s ease;
    min-width: 100px;
    text-align: left;
  }
  .mv-btn:hover {
    background: #fff;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  }
  .mv-status {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 8px 16px;
    color: #555;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    background: #fafafa;
    border-radius: 0 0 8px 8px;
    margin-top: -4px;
  }
</style>

<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>

<div class="mv-container">
  <model-viewer
    id="mv___KEY__"
    src="data:model/gltf-binary;base64,__GLB_BASE64__"
    alt="3D model of generated house"
    camera-controls
    interaction-prompt="none"
    environment-image="neutral"
    exposure="1.0"
    shadow-intensity="1"
    shadow-softness="0.8"
    camera-orbit="35deg 65deg auto"
    field-of-view="35deg"
    min-camera-orbit="auto auto auto"
    max-camera-orbit="auto 90deg auto"
    style="background-color: transparent;"
  >
    <div slot="progress-bar" style="display: none;"></div>
  </model-viewer>
  
  <div class="mv-controls">
    <button class="mv-btn" onclick="setView___KEY__('persp')">🏠 パース</button>
    <button class="mv-btn" onclick="setView___KEY__('top')">🪂 真上</button>
    <button class="mv-btn" onclick="setView___KEY__('front')">👁️ 正面</button>
    <button class="mv-btn" onclick="setView___KEY__('side')">📐 側面</button>
  </div>
</div>

<div class="mv-status">
  <span>✅ Google &lt;model-viewer&gt; | 🖱️ ドラッグで回転、スクロールでズーム</span>
  <span>📦 __FILESIZE__ KB</span>
</div>

<script>
  function setView___KEY__(view) {
    const mv = document.getElementById('mv___KEY__');
    if (view === 'persp') {
      mv.cameraOrbit = '35deg 65deg auto';
    } else if (view === 'top') {
      mv.cameraOrbit = '0deg 0deg auto';
    } else if (view === 'front') {
      mv.cameraOrbit = '0deg 90deg auto';
    } else if (view === 'side') {
      mv.cameraOrbit = '90deg 75deg auto';
    }
  }
</script>
"""


def render_3d_viewer(glb_bytes: bytes, height: int = 700, key: str = "viewer"):
    """GLB バイト列を Google <model-viewer> Web Component で表示"""
    glb_base64 = base64.b64encode(glb_bytes).decode("ascii")
    file_size_kb = len(glb_bytes) / 1024
    
    html = (
        MODEL_VIEWER_TEMPLATE
        .replace("__KEY__", key)
        .replace("__HEIGHT__", str(height))
        .replace("__FILESIZE__", f"{file_size_kb:.1f}")
        .replace("__GLB_BASE64__", glb_base64)
    )
    components.html(html, height=height + 60)


# ============================================================
# サンプル画像
# ============================================================
@st.cache_data
def get_sample_images() -> list[Path]:
    test_dir = PROJECT_ROOT / "data" / "floorplan_seg" / "test" / "images"
    if not test_dir.exists():
        return []
    return sorted(test_dir.glob("*.jpg"))[:6]


# ============================================================
# UI
# ============================================================
st.title("🏠 Floor Plan 3D Viewer")
st.caption("間取り図画像 → ML(検出+セグメンテーション) → ベクトル化 → 3D の家")

with st.sidebar:
    st.subheader("⚙️ 設定")
    conf_threshold = st.slider(
        "信頼度しきい値", 0.05, 0.95, 0.25, 0.05,
        help="ML 推論で採用する最小信頼度"
    )
    st.markdown("---")
    st.subheader("📊 モデル情報")
    st.markdown(
        "- **検出+セグ**: YOLOv8n-seg\n"
        "- **クラス**: 8\n"
        "- **Mask mAP@0.5**: **0.903**\n"
        "- **room Mask AP**: 0.987\n"
        "- **wall Mask AP**: 0.683"
    )
    st.markdown("---")
    st.subheader("🛠️ 技術スタック")
    st.markdown(
        "- **ML**: Ultralytics YOLOv8\n"
        "- **ベクトル化**: OpenCV + scikit-image\n"
        "- **3D**: trimesh + shapely\n"
        "- **Viewer**: Google `<model-viewer>`"
    )
    st.markdown("---")
    st.caption("Phase 1〜6 完了 | [GitHub: Mao925/floor-plan-recognition](https://github.com/Mao925/floor-plan-recognition)")


st.subheader("1️⃣ 画像を選ぶ")
col_input1, col_input2 = st.columns([2, 3])

with col_input1:
    input_method = st.radio(
        "入力方法",
        options=["📁 ファイルをアップロード", "🖼️ サンプル画像"],
    )

uploaded_image: np.ndarray | None = None

with col_input2:
    if input_method == "📁 ファイルをアップロード":
        uploaded_file = st.file_uploader(
            "間取り図画像 (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            pil = Image.open(uploaded_file).convert("RGB")
            uploaded_image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        samples = get_sample_images()
        if samples:
            sample_idx = st.selectbox(
                "サンプル選択",
                options=range(len(samples)),
                format_func=lambda i: f"Sample {i+1}: {samples[i].name[:30]}...",
            )
            uploaded_image = cv2.imread(str(samples[sample_idx]))

if uploaded_image is not None:
    with st.expander("📷 入力画像プレビュー", expanded=False):
        st.image(cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB), use_container_width=True)


st.subheader("2️⃣ 3D を生成")

if uploaded_image is None:
    st.info("👆 画像を選択するとボタンが有効になります")
else:
    if st.button("🚀 3D を生成", type="primary"):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        def update_progress(msg: str, pct: float):
            progress_bar.progress(pct)
            progress_text.markdown(f"**{msg}**")
        
        try:
            t_total = time.time()
            result = run_full_pipeline(
                uploaded_image,
                conf_threshold=conf_threshold,
                progress_callback=update_progress,
            )
            total_ms = (time.time() - t_total) * 1000
            progress_bar.empty()
            progress_text.empty()
            st.session_state["result"] = result
            st.session_state["uploaded_image"] = uploaded_image
            st.session_state["total_ms"] = total_ms
            st.success(f"✅ 完了({total_ms:.0f} ms)")
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            st.error(f"❌ エラー: {e}")
            st.exception(e)


if "result" in st.session_state:
    result = st.session_state["result"]
    floorplan = result["floorplan_json"]
    
    st.subheader("3️⃣ 結果")
    
    summary = floorplan["summary"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("部屋", summary["n_rooms"])
    col2.metric("壁線分", summary["n_walls"])
    col3.metric("設備", summary["n_devices"])
    col4.metric("総処理時間", f"{st.session_state['total_ms']:.0f} ms")
    
    tab_3d, tab_detect, tab_json = st.tabs(["🏠 3D ビュー", "🔍 検出結果", "📄 JSON"])
    
    with tab_3d:
        render_3d_viewer(result["glb_bytes"], height=700)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button(
            "📥 GLB ダウンロード",
            data=result["glb_bytes"],
            file_name="house.glb",
            mime="model/gltf-binary",
            use_container_width=True,
        )
        col_dl2.download_button(
            "📥 JSON ダウンロード",
            data=json.dumps(floorplan, indent=2),
            file_name="floorplan.json",
            mime="application/json",
            use_container_width=True,
        )
    
    with tab_detect:
        original = st.session_state.get("uploaded_image")
        if original is not None:
            col_a, col_b = st.columns(2)
            col_a.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="元画像", use_container_width=True)
            col_b.image(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB), caption="ML 検出結果", use_container_width=True)
        st.markdown("#### クラス別検出数")
        st.json(summary["devices_by_class"])
        st.markdown("#### 処理時間内訳")
        for k, v in result["timing"].items():
            st.text(f"  {k}: {v:.0f} ms")
    
    with tab_json:
        st.json(floorplan)
