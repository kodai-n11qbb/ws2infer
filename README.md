# ws2infer

WebSocketベースのONNX推論サーバー。リアルタイムビデオストリーム処理に対応。

## 概要

ws2inferは、WebSocketを介してリアルタイムでビデオストリームを受信し、ONNXモデルで推論を実行するサーバーシステムです。Webブラウザやその他のWebSocketクライアントから送信される映像データをリアルタイムに処理し、AI推論結果を返信することを目的としています。

### 主な用途
- **リアルタイムビデオ分析**: Webカメラからの映像をリアルタイムで分析
- **エッジAI推論**: ブラウザベースのAIアプリケーションにバックエンド推論を提供
- **ストリーミング処理**: 複数クライアントからの同時ビデオストリーム処理

## 特徴

- WebSocketサーバーで複数クライアント対応
- 環境自動検出と最適バックエンド選択
- マルチプラットフォーム対応 (CPU, CoreML, CUDA等)
- **設定ファイル対応** - JSONベースの柔軟な設定管理

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

## 使い方

### 基本的な使用フロー

1. **サーバー起動**: WebSocketサーバーを起動してONNXモデルをロード
2. **クライアント接続**: Webブラウザ等からWebSocketで接続
3. **ビデオストリーム送信**: クライアントからリアルタイムで映像フレームを送信
4. **推論実行**: サーバーが受信フレームをONNXモデルで推論
5. **結果返信**: 推論結果をクライアントに返信

### ステップバイステップガイド

#### 1. サーバーの起動

```bash
# デモサーバーで手軽に試す（OpenCV不要）
./demo_server ws://localhost:8080 model.onnx

# 本サーバーを起動（設定ファイル使用）
./build/ws2infer --config config.json
```

#### 2. Webクライアントからの接続

```html
<!-- test_client.html をブラウザで開く -->
<!-- WebSocket URL: ws://localhost:8080 -->
```

#### 3. ビデオストリームの送信と推論

JavaScriptクライアントでの実装例：

```javascript
const ws = new WebSocket('ws://localhost:8080');
const video = document.getElementById('video');

// ビデオフレームを定期的に送信
setInterval(() => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
    const base64Data = imageData.split(',')[1];
    
    ws.send(JSON.stringify({
        type: 'frame',
        data: base64Data
    }));
}, 100); // 10FPSで送信

// 推論結果を受信
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.type === 'inference_result') {
        console.log(`推論時間: ${result.inference_time_ms}ms`);
        // 結果をUIに表示
    }
};
```

## 実行方法

```bash
# デモサーバー（基礎動作、推論可能）
./demo_server ws://localhost:8080 model.onnx

# システムテスト
./simple_test model.onnx

# 本サーバー（要OpenCV、設定ファイル使用）
./build/ws2infer --config config.json

# コマンドライン引数で設定を上書き
./build/ws2infer --config config.json --ws-url ws://localhost:9000 --model custom_model.onnx
```

## クライアント接続

1. `test_client.html` をブラウザで開く
2. WebSocket URL: `ws://localhost:8080`
3. 画像を選択して推論実行

## WebSocketメッセージ形式

### クライアント → サーバー：画像送信

クライアントはビデオフレームをBase64エンコードして送信します：

```json
{
  "type": "frame",
  "data": "base64_encoded_image_data"
}
```

- `type`: `"frame"` 固定
- `data`: 画像のBase64エンコード文字列（data:image/jpeg;base64, のプレフィックスは不要）

### サーバー → クライアント：推論結果

```json
{
  "type": "inference_result",
  "success": true,
  "inference_time_ms": 45.2,
  "predictions": [
    {"class": "person", "confidence": 0.95, "bbox": [10, 20, 100, 200]},
    {"class": "car", "confidence": 0.87, "bbox": [150, 80, 300, 180]}
  ]
}
```

- `type`: `"inference_result"` 固定
- `success`: 推論が成功したかどうか
- `inference_time_ms`: 推論にかかった時間（ミリ秒）
- `predictions`: 推論結果の配列（モデルによって形式が異なります）

### エラーメッセージ

```json
{
  "type": "error",
  "message": "Invalid image format",
  "code": 400
}
```

## 設定ファイル

`config.json` でサーバー設定を管理：

```json
{
  "server": {
    "host": "localhost",
    "port": 8080,
    "max_connections": 10
  },
  "model": {
    "path": "model.onnx",
    "backend": "auto",
    "input_size": [224, 224]
  },
  "inference": {
    "device": "auto",
    "num_threads": 4
  },
  "preprocessing": {
    "normalize": true,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

### 設定項目

- **server**: WebSocketサーバー設定
- **model**: ONNXモデル設定
- **inference**: 推論デバイス設定
- **preprocessing**: 画像前処理設定
- **logging**: ログ出力設定

## プロジェクト構造

```
ws2infer/
├── src/                    # C++本実装
│   ├── config_loader.cpp   # 設定ファイル読み込み
│   └── ...
├── config.json             # 設定ファイル
├── requirements.txt        # 依存関係リスト (Homebrew)
├── demo_server.cpp         # 基礎動作デモ（依存なし）
├── simple_test.cpp         # システムテスト
├── test_client.html        # Webテストクライアント
├── third_party/            # 外部ライブラリ
│   └── json.hpp           # nlohmann/json
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
- OpenCV 4.x (本サーバーのみ)
- ONNX Runtime 1.12+

### インストール (macOS)

```bash
# Homebrewで依存関係をインストール
brew install cmake

# requirements.txtに記載の依存関係をインストール
brew install opencv onnxruntime protobuf openssl openvino gflags ffmpeg openexr

# または、requirements.txtから一括インストール
cat requirements.txt | xargs brew install
```

### 動作確認

```bash
# ビルド
./build.sh

# デモサーバーで動作確認（OpenCV不要）
./demo_server ws://localhost:8080 model.onnx
```

## ライセンス

MIT License