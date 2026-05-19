# Floor Plan Recognition

建築・間取り図面からの設備記号検出システム。物体検出 → 構造化データ(JSON)→ Viewer というパイプラインを、ミニチュア版で実装したプロジェクトです。

## ステータス
🚧 開発中(Phase 1 完了 / Phase 2: モデル学習へ)

## プロジェクトの背景

このプロジェクトは「図面 → ML → 構造化データ → Viewer」というフローを、個人プロジェクトとして実装することを目的としています。建築・設備図面認識システムにおける典型的なパイプラインのミニ実装です。

## 技術スタック

- **言語**: Python 3.11
- **物体検出**: YOLOv8 (Ultralytics)
- **画像処理**: OpenCV, Pillow
- **学習環境**: Google Colab (T4 GPU)
- **推論・開発環境**: ローカル macOS (Apple Silicon M2 + MPS)
- **Viewer**: Streamlit(予定)

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

## ディレクトリ構成
floor-plan-recognition/
├── data/                    # データセット(.gitignoreで除外)
│   ├── roboflow/           # Roboflowから取得した元データ
│   └── floorplan_yolo/     # 6クラスにフィルタ後の最終データ
├── scripts/                 # 単発実行スクリプト
│   ├── check_env.py        # 環境確認
│   ├── download_roboflow.py # データセット取得
│   ├── inspect_data.py     # データ統計確認
│   ├── visualize_data.py   # 可視化
│   └── prepare_dataset.py  # クラスフィルタ + 再分割
├── src/                     # メインソース
├── models/                  # 学習済みモデル(.gitignoreで除外)
├── outputs/                 # 実験結果
└── notebooks/               # 探索用ノートブック

## セットアップ

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
```

## 次のステップ(Phase 2 以降)

- [ ] YOLOv8n でベースライン学習(Colab)
- [ ] 評価指標(mAP, クラス別 AP)の確認
- [ ] Failure case 分析
- [ ] 仮説検証サイクル:モデルサイズ/解像度/SAHI による改善

## ライセンス
データセット: CC BY 4.0 (Roboflow Universe)
コード: MIT(予定)
