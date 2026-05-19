# Baseline YOLOv8n Training Results

このフォルダには Phase 2 のベースライン学習結果が含まれる。

## 内容

- `results.png` - 学習曲線(loss / mAP の推移)
- `confusion_matrix.png` - 混同行列
- `confusion_matrix_normalized.png` - 正規化された混同行列
- `Box*.png` - Precision/Recall/F1 のクラス別カーブ
- `labels.jpg` - 学習データのラベル分布
- `train_batch*.jpg` - 学習中のバッチサンプル
- `val_batch*_pred.jpg` - 検証データの予測結果
- `val_batch*_labels.jpg` - 検証データの正解ラベル
- `results.csv` - エポックごとのメトリクス(CSV形式)
- `args.yaml` - 学習時の全ハイパーパラメータ

## 主要メトリクス(test セット)

- mAP@0.5: **0.815**
- mAP@0.5:0.95: 0.616
- Precision: 0.823
- Recall: 0.733

## 重み

`weights/best.pt` および `weights/last.pt` は .gitignore で除外されている(サイズが大きいため)。
再学習するには `notebooks/01_baseline_training.ipynb` を Colab で実行する。
