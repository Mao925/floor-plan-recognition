# Floor Plan Recognition

建築・間取り図面からの設備記号検出システム。物体検出 → 構造化データ(JSON)→ Viewer というパイプラインを、ミニチュア版で実装したプロジェクト。

## ステータス
✅ Phase 2 完了 (Baseline モデル学習完了 / mAP@0.5 = 0.815)
🚧 Phase 3 進行中:仮説検証サイクル

## 成果(現時点)

### Test セットでの最終スコア(YOLOv8n ベースライン)

| 指標 | 値 |
|---|---|
| mAP@0.5 | **0.815** |
| mAP@0.5:0.95 | 0.616 |
| Precision | 0.823 |
| Recall | 0.733 |

### クラス別 mAP@0.5

| クラス | mAP@0.5 | 評価 |
|---|---|---|
| window | 0.992 | 🏆 |
| door | 0.986 | 🏆 |
| toilet | 0.966 | 🏆 |
| sink | 0.823 | ✅ |
| staircase | 0.704 | ○ |
| shower | 0.417 | △ 課題 |

学習結果の詳細(学習曲線、混同行列、推論サンプル)は `models/baseline_yolov8n/` 配下に保存。

## プロジェクトの背景

このプロジェクトは「図面 → ML → 構造化データ → Viewer」というフローを、個人プロジェクトとして実装することを目的としている。建築・設備図面認識システムにおける典型的なパイプラインのミニ実装。

## 技術スタック

- **言語**: Python 3.11
- **物体検出**: YOLOv8 (Ultralytics)
- **画像処理**: OpenCV, Pillow
- **学習環境**: Google Colab (T4 GPU)
- **推論・開発環境**: ローカル macOS (Apple Silicon M2 + MPS)
- **Viewer**: Streamlit(Phase 5 予定)

## データセット

### 採用データセット
[Floor Plan Annotation v1](https://universe.roboflow.com/smartapp-3jazx/floor-plan-annotation-u6whl/dataset/1) (Roboflow Universe)

### データセット選定の経緯

当初 CubiCasa5K(5000枚)の採用を検討したが、以下の理由で Roboflow Universe の代替データセットに切り替えた:
- Zenodo からのダウンロードが極端に低速(35分で17%)
- CubiCasa5K のアノテーション形式が SVG ポリゴンで、YOLO 形式への変換に工数が発生
- 本プロジェクトの目的(動くシステム + 仮説検証サイクル)の達成には、整形済データセットを使う方が時間効率が高いと判断

### データの前処理(`scripts/prepare_dataset.py`)

データ実物を観察した結果、以下の前処理を実施:

1. **クラス選定**: 元データ9クラスのうち、以下を除外
   - `room`, `wall`: 「領域」であり物体検出には不適と判断
   - `bathtub`: サンプル数極少(全体0.7%)で評価信頼性が確保できないため除外
2. **6クラスで最終化**: `door`, `shower`, `sink`, `staircase`, `toilet`, `window`
3. **train/val/test 再分割**: 元データは valid のみで test なしのため、70%/15%/15% に再分割(seed=42 で再現可能)

### データセット統計

| split | 画像数 | bbox数 |
|---|---|---|
| train | 158 | 1,807 |
| val | 33 | 408 |
| test | 36 | 409 |
| 合計 | 227 | 2,624 |

## ベースライン学習(Phase 2)

### 設定
- モデル: YOLOv8n (約 300万 params)
- エポック数: 50 (EarlyStopping: patience=15)
- 画像サイズ: 640
- バッチサイズ: 16
- Optimizer: AdamW (自動選択)
- 学習時間: 約 4 分(Tesla T4 GPU)
- Seed: 42 (再現性確保)

### 観察された知見

混同行列の分析から、shower の低精度(0.42)は **「他クラスとの混同」ではなく「検出漏れ」が原因** であることを発見:
- 22個の shower のうち、20個が背景クラスとして見落とされていた
- 他クラスとの混同はゼロ

→ これは「クラス学習自体は正しいが、shower の特徴抽出が不十分」という仮説につながる。

学習曲線が50エポック時点でまだ完全に平坦化していないことから、**追加学習の余地がある** ことも観察された。

## 仮説検証サイクル(Phase 3 で進行)

ベースラインの観察から立てた検証可能な仮説(優先順):

| # | 仮説 | 実験設計 | ステータス |
|---|---|---|---|
| 1 | 入力解像度を 640 → 1024 に上げれば、小物体(shower)の検出漏れが減る | imgsz=1024 で再学習 | 🚧 |
| 2 | エポック数を 50 → 100 にすれば、学習曲線が平坦化し mAP がさらに上がる | epochs=100, patience=30 で再学習 | 計画中 |
| 3 | モデルサイズを n → s に上げれば、全体的に精度向上(速度とのトレードオフ評価) | YOLOv8s で再学習 | 計画中 |
| 4 | SAHI(タイル分割推論)で小物体検出を改善 | 推論時の工夫 | 計画中 |

## ディレクトリ構成
floor-plan-recognition/
├── data/                    # データセット(.gitignoreで除外)
│   ├── roboflow/           # Roboflowから取得した元データ
│   └── floorplan_yolo/     # 6クラスにフィルタ後の最終データ
├── notebooks/
│   └── 01_baseline_training.ipynb  # Colab 学習ノートブック
├── scripts/                 # 単発実行スクリプト
│   ├── check_env.py        # 環境確認
│   ├── download_roboflow.py # データセット取得
│   ├── inspect_data.py     # データ統計確認
│   ├── visualize_data.py   # 可視化
│   └── prepare_dataset.py  # クラスフィルタ + 再分割
├── src/                     # メインソース
├── models/
│   └── baseline_yolov8n/   # 学習結果(画像のみ Git に含む、.pt は除外)
├── outputs/                 # 実験結果
└── docs/

## セットアップと再現手順

```bash
# 仮想環境作成
conda create -n floorplan python=3.11 -y
conda activate floorplan

# 依存パッケージ
pip install -r requirements.txt

# Roboflow API キーを .env に設定
echo "ROBOFLOW_API_KEY=your_key_here" > .env

# データセット取得 + 前処理
python scripts/download_roboflow.py
python scripts/prepare_dataset.py

# 学習は Colab で実行(notebooks/01_baseline_training.ipynb)
```

## ライセンス
- データセット: CC BY 4.0 (Roboflow Universe)
- コード: MIT(予定)
