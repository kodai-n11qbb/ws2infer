# ws2infer

WebSocketベースのONNX推論サーバー。リアルタイムビデオストリーム処理に対応。

## 特徴

- WebSocketサーバーで複数クライアント対応
- 環境自動検出と最適バックエンド選択
- マルチプラットフォーム対応 (CPU, CoreML, CUDA等)

## ビルド方法

```bash
# macOS
brew install opencv onnxruntime cmake
./build.sh

# Linux
sudo apt install build-essential cmake libopencv-dev
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 実行方法

```bash
# デモサーバー（基礎動作）
./demo_server ws://localhost:8080 model.onnx

# システムテスト
./simple_test model.onnx

# 本サーバー（要OpenCV）
./build/ws2infer ws://localhost:8080 model.onnx
```

## クライアント接続

1. `test_client.html` をブラウザで開く
2. WebSocket URL: `ws://localhost:8080`
3. 画像を選択して推論実行

## WebSocketメッセージ形式

### 画像送信
```json
{
  "type": "frame",
  "data": "base64_encoded_image"
}
```

### 推論結果
```json
{
  "type": "inference_result",
  "success": true,
  "inference_time_ms": 45.2
}
```

## プロジェクト構造

```
ws2infer/
├── src/                    # C++本実装
├── demo_server.cpp         # 基礎動作デモ（依存なし）
├── simple_test.cpp         # システムテスト
├── test_client.html        # Webテストクライアント
├── mmds/                   # Mermaid設計図
├── CMakeLists.txt         # ビルド設定
├── build.sh               # ビルドスクリプト
└── README.md              # このファイル
```

## 設計図

システム設計は `mmds/` ディレクトリのMermaid図に記載されています：

- `system.mmd` - 全体システムアーキテクチャ
- `ONNXsystem.mmd` - ONNX推論システムの詳細設計

## 依存関係

- C++17
- CMake 3.16+
- OpenCV 4.x
- ONNX Runtime 1.12+

## ライセンス

MIT License