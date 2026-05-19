"""環境確認スクリプト: PyTorch + MPS + YOLO の動作確認"""
import sys
import platform
import torch
import ultralytics

print("=" * 60)
print("環境情報")
print("=" * 60)
print(f"Python:           {sys.version.split()[0]}")
print(f"Platform:         {platform.platform()}")
print(f"PyTorch:          {torch.__version__}")
print(f"Ultralytics:      {ultralytics.__version__}")

print()
print("=" * 60)
print("GPU / MPS 確認")
print("=" * 60)
print(f"MPS available:    {torch.backends.mps.is_available()}")
print(f"MPS built:        {torch.backends.mps.is_built()}")
print(f"CUDA available:   {torch.cuda.is_available()}  (Apple Silicon では False が正常)")

print()
print("=" * 60)
print("MPS で簡単な計算ができるか")
print("=" * 60)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = x @ y
    print(f"行列計算成功! デバイス: {z.device}")
    print(f"結果の形状: {z.shape}")
else:
    print("MPS が使えません。CPU で動作します。")

print()
print("=" * 60)
print("完了")
print("=" * 60)
