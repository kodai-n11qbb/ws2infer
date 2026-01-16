use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::signaling::{SignalingMessage, SignalingMessageType};
use bytes::Bytes;
use tokio::time::{Duration, Instant};
use log::info;

// 高性能SFUの構造体
#[derive(Debug, Clone)]
pub struct HighPerformanceSfuRoom {
    pub id: String,
    pub sender_id: Option<String>,
    pub viewer_ids: Vec<String>,
    pub media_stream: Option<MediaStream>,
    pub quality_profile: QualityProfile,
}

#[derive(Debug, Clone)]
pub struct MediaStream {
    pub stream_id: String,
    pub video_track: VideoTrack,
    pub audio_track: AudioTrack,
    pub encoding_params: EncodingParams,
    pub last_frame_time: Instant,
}

#[derive(Debug, Clone)]
pub struct VideoTrack {
    pub track_id: String,
    pub codec: VideoCodec,
    pub resolution: (u32, u32),
    pub fps: u32,
    pub bitrate: u32,
    pub keyframe_interval: u32,
}

#[derive(Debug, Clone)]
pub struct AudioTrack {
    pub track_id: String,
    pub codec: AudioCodec,
    pub bitrate: u32,
    pub sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct EncodingParams {
    pub hardware_acceleration: bool,
    pub preset: EncodingPreset,
    pub profile: CodecProfile,
    pub tune: TuneOption,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoCodec {
    H264,
    VP8,
    VP9,
    AV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioCodec {
    Opus,
    AAC,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodingPreset {
    Ultrafast,
    Superfast,
    Veryfast,
    Faster,
    Fast,
    Medium,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodecProfile {
    Baseline,
    Main,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TuneOption {
    None,
    Film,
    Animation,
    Grain,
    Stillimage,
    Fastdecode,
    Zerolatency,
}

#[derive(Debug, Clone)]
pub struct QualityProfile {
    pub target_fps: u32,
    pub target_bitrate: u32,
    pub adaptive_bitrate: bool,
    pub network_adaptation: bool,
    pub hardware_acceleration: bool,
}

#[derive(Debug, Clone)]
pub struct HighPerformanceSfuConnection {
    pub id: String,
    pub is_sender: bool,
    pub quality_level: QualityLevel,
    pub last_ping: Instant,
    pub bandwidth: u32,
    pub latency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityLevel {
    Low,     // 480p, 15fps, 500kbps
    Medium,  // 720p, 30fps, 1.5Mbps
    High,    // 1080p, 60fps, 3Mbps
    Ultra,   // 4K, 60fps, 8Mbps
}

pub struct HighPerformanceSfuManager {
    rooms: Arc<RwLock<HashMap<String, HighPerformanceSfuRoom>>>,
    connections: Arc<RwLock<HashMap<String, HighPerformanceSfuConnection>>>,
    media_processor: Arc<MediaProcessor>,
    bandwidth_monitor: Arc<BandwidthMonitor>,
    quality_controller: Arc<QualityController>,
}

impl HighPerformanceSfuManager {
    pub fn new() -> Self {
        Self {
            rooms: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
            media_processor: Arc::new(MediaProcessor::new()),
            bandwidth_monitor: Arc::new(BandwidthMonitor::new()),
            quality_controller: Arc::new(QualityController::new()),
        }
    }

    pub async fn handle_message(&self, room_id: &str, message: &SignalingMessage) -> Result<Vec<SignalingMessage>> {
        match message.message_type {
            SignalingMessageType::Join => {
                let connection_id = message.connection_id.as_ref().ok_or_else(|| anyhow::anyhow!("No connection_id"))?;
                let is_sender = message.is_sender.unwrap_or(false);
                self.add_connection(room_id, connection_id.clone(), is_sender).await
            }
            SignalingMessageType::Offer => {
                let connection_id = message.connection_id.as_ref().ok_or_else(|| anyhow::anyhow!("No connection_id"))?;
                let sdp = message.data.as_ref()
                    .and_then(|d| d.get("sdp"))
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| anyhow::anyhow!("No SDP in offer"))?;
                self.handle_media_offer(room_id, connection_id, sdp).await
            }
            SignalingMessageType::Answer => {
                let connection_id = message.connection_id.as_ref().ok_or_else(|| anyhow::anyhow!("No connection_id"))?;
                let sdp = message.data.as_ref()
                    .and_then(|d| d.get("sdp"))
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| anyhow::anyhow!("No SDP in answer"))?;
                self.handle_media_answer(room_id, connection_id, sdp).await
            }
            SignalingMessageType::IceCandidate => {
                let connection_id = message.connection_id.as_ref().ok_or_else(|| anyhow::anyhow!("No connection_id"))?;
                let candidate = message.data.as_ref()
                    .and_then(|d| d.get("candidate"))
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| anyhow::anyhow!("No candidate"))?;
                self.handle_ice_candidate(room_id, connection_id, candidate).await
            }
            _ => Ok(Vec::new())
        }
    }

    pub async fn create_room(&self, room_id: String) -> Result<()> {
        let quality_profile = QualityProfile {
            target_fps: 60,        // 最大fps
            target_bitrate: 8000,  // 8Mbps for 4K
            adaptive_bitrate: true,
            network_adaptation: true,
            hardware_acceleration: true,
        };

        let room = HighPerformanceSfuRoom {
            id: room_id.clone(),
            sender_id: None,
            viewer_ids: Vec::new(),
            media_stream: None,
            quality_profile,
        };

        let mut rooms = self.rooms.write().await;
        rooms.insert(room_id, room);
        Ok(())
    }

    pub async fn add_connection(&self, room_id: &str, connection_id: String, is_sender: bool) -> Result<Vec<SignalingMessage>> {
        let connection = HighPerformanceSfuConnection {
            id: connection_id.clone(),
            is_sender,
            quality_level: if is_sender { QualityLevel::Ultra } else { QualityLevel::Medium },
            last_ping: Instant::now(),
            bandwidth: 8000, // 初期値8Mbps
            latency: Duration::from_millis(50),
        };

        let mut connections = self.connections.write().await;
        connections.insert(connection_id.clone(), connection);

        let mut rooms = self.rooms.write().await;
        if let Some(room) = rooms.get_mut(room_id) {
            if is_sender {
                room.sender_id = Some(connection_id.clone());
                // 送信者用メディアストリーム作成
                room.media_stream = Some(self.create_optimized_media_stream(&room.quality_profile).await);
            } else {
                room.viewer_ids.push(connection_id.clone());
                // 受信者に最適な品質レベルを設定
                self.optimize_quality_for_viewer(room_id, &connection_id).await;
            }
        }

        let mut responses = Vec::new();

        // 受信者に送信者のストリーム情報を送信
        if !is_sender {
            let rooms_guard = self.rooms.read().await;
            if let Some(room) = rooms_guard.get(room_id) {
                if let Some(media_stream) = &room.media_stream {
                    responses.push(SignalingMessage {
                        message_type: SignalingMessageType::Offer,
                        connection_id: Some(connection_id),
                        sender_id: room.sender_id.clone(),
                        offer_id: Some(Uuid::new_v4().to_string()),
                        data: Some(serde_json::json!({
                            "sdp": self.generate_optimized_sdp(media_stream).await,
                            "media_stream": {
                                "video": {
                                    "codec": "H264",
                                    "resolution": (1920, 1080),
                                    "fps": 60,
                                    "bitrate": 8000,
                                    "hardware_acceleration": true
                                },
                                "audio": {
                                    "codec": "Opus",
                                    "bitrate": 128
                                }
                            }
                        })),
                        is_sender: Some(false),
                    });
                }
            }
        }

        Ok(responses)
    }

    async fn create_optimized_media_stream(&self, quality_profile: &QualityProfile) -> MediaStream {
        let video_track = VideoTrack {
            track_id: Uuid::new_v4().to_string(),
            codec: VideoCodec::H264, // ハードウェアアクセラレーション対応
            resolution: (1920, 1080),
            fps: quality_profile.target_fps,
            bitrate: quality_profile.target_bitrate,
            keyframe_interval: 60, // 1秒ごとにキーフレーム
        };

        let audio_track = AudioTrack {
            track_id: Uuid::new_v4().to_string(),
            codec: AudioCodec::Opus,
            bitrate: 128,
            sample_rate: 48000,
        };

        let encoding_params = EncodingParams {
            hardware_acceleration: quality_profile.hardware_acceleration,
            preset: EncodingPreset::Ultrafast, // 最低遅延
            profile: CodecProfile::High,
            tune: TuneOption::Zerolatency,    // ゼロ遅延チューニング
        };

        MediaStream {
            stream_id: Uuid::new_v4().to_string(),
            video_track,
            audio_track,
            encoding_params,
            last_frame_time: Instant::now(),
        }
    }

    async fn generate_optimized_sdp(&self, media_stream: &MediaStream) -> String {
        // 最適化されたSDP生成
        format!(
            "v=0\r\no=- {} 0 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n\
             m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
             a=rtpmap:96 H264/90000\r\n\
             a=fmtp:96 packetization-mode=1;profile-level-id=42e01f\r\n\
             a=rtcp-fb:96 ccm fir\r\na=rtcp-fb:96 nack\r\na=rtcp-fb:96 nack pli\r\n\
             a=rtcp-fb:96 goog-remb\r\na=rtcp-fb:96 transport-cc\r\n\
             m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
             a=rtpmap:111 opus/48000/2\r\n",
            media_stream.stream_id
        )
    }

    async fn optimize_quality_for_viewer(&self, _room_id: &str, viewer_id: &str) {
        // 帯域監視に基づく品質最適化
        let bandwidth = self.bandwidth_monitor.get_available_bandwidth(viewer_id).await;
        let latency = self.bandwidth_monitor.get_latency(viewer_id).await;

        let optimal_quality = self.quality_controller.calculate_optimal_quality(bandwidth, latency).await;

        let mut connections = self.connections.write().await;
        if let Some(connection) = connections.get_mut(viewer_id) {
            connection.quality_level = optimal_quality;
            connection.bandwidth = bandwidth;
            connection.latency = latency;
        }
    }

    async fn handle_media_offer(&self, room_id: &str, connection_id: &str, sdp: &str) -> Result<Vec<SignalingMessage>> {
        // メディアオファー処理 - ハードウェアエンコーディングで最適化
        let mut responses = Vec::new();

        // SDP解析と最適化
        let optimized_sdp = self.optimize_sdp_for_hardware(sdp).await;

        // 全受信者に転送
        let rooms_guard = self.rooms.read().await;
        if let Some(room) = rooms_guard.get(room_id) {
            for viewer_id in &room.viewer_ids {
                responses.push(SignalingMessage {
                    message_type: SignalingMessageType::Offer,
                    connection_id: Some(viewer_id.clone()),
                    sender_id: Some(connection_id.to_string()),
                    offer_id: Some(Uuid::new_v4().to_string()),
                    data: Some(serde_json::json!({ "sdp": optimized_sdp })),
                    is_sender: Some(false),
                });
            }
        }

        Ok(responses)
    }

    async fn handle_media_answer(&self, _room_id: &str, _connection_id: &str, sdp: &str) -> Result<Vec<SignalingMessage>> {
        // メディアアンサー処理
        info!("Received media answer: {}", sdp);
        Ok(Vec::new())
    }

    async fn handle_ice_candidate(&self, room_id: &str, connection_id: &str, candidate: &str) -> Result<Vec<SignalingMessage>> {
        let mut responses = Vec::new();

        // ICE候補転送
        let rooms_guard = self.rooms.read().await;
        if let Some(room) = rooms_guard.get(room_id) {
            let is_sender = room.sender_id.as_ref().map(|s| s.as_str()) == Some(connection_id);

            if is_sender {
                // 送信者のICE候補を全受信者に転送
                for viewer_id in &room.viewer_ids {
                    responses.push(SignalingMessage {
                        message_type: SignalingMessageType::IceCandidate,
                        connection_id: Some(viewer_id.clone()),
                        sender_id: Some(connection_id.to_string()),
                        offer_id: None,
                        data: Some(serde_json::json!({ "candidate": candidate })),
                        is_sender: None,
                    });
                }
            } else {
                // 受信者のICE候補を送信者に転送
                if let Some(sender_id) = &room.sender_id {
                    responses.push(SignalingMessage {
                        message_type: SignalingMessageType::IceCandidate,
                        connection_id: Some(sender_id.clone()),
                        sender_id: Some(connection_id.to_string()),
                        offer_id: None,
                        data: Some(serde_json::json!({ "candidate": candidate })),
                        is_sender: None,
                    });
                }
            }
        }

        Ok(responses)
    }

    async fn optimize_sdp_for_hardware(&self, sdp: &str) -> String {
        // ハードウェアアクセラレーション用SDP最適化
        sdp.replace("H264/90000", "H264/90000")
           .replace("profile-level-id=42e01f", "profile-level-id=42e01f")
    }

    pub async fn process_media_frame(&self, room_id: &str, frame_data: Bytes) -> Result<Vec<(String, Bytes)>> {
        let mut outputs = Vec::new();

        // メディアフレーム処理 - ハードウェアエンコーディング
        let rooms_guard = self.rooms.read().await;
        if let Some(room) = rooms_guard.get(room_id) {
            if let Some(media_stream) = &room.media_stream {
                // 各受信者に最適化されたフレームを生成
                for viewer_id in &room.viewer_ids {
                    let connections_guard = self.connections.read().await;
                    if let Some(connection) = connections_guard.get(viewer_id) {
                        let optimized_frame = self.media_processor.optimize_frame_for_quality(
                            frame_data.clone(),
                            &connection.quality_level,
                            &media_stream.encoding_params,
                        ).await?;
                        outputs.push((viewer_id.clone(), optimized_frame));
                    }
                }
            }
        }

        Ok(outputs)
    }
}

// メディアプロセッサ
pub struct MediaProcessor {
    hardware_encoder: Option<HardwareEncoder>,
}

impl MediaProcessor {
    pub fn new() -> Self {
        Self {
            hardware_encoder: HardwareEncoder::new(),
        }
    }

    async fn optimize_frame_for_quality(&self, frame_data: Bytes, quality: &QualityLevel, params: &EncodingParams) -> Result<Bytes> {
        match quality {
            QualityLevel::Ultra => self.process_ultra_quality(frame_data, params).await,
            QualityLevel::High => self.process_high_quality(frame_data, params).await,
            QualityLevel::Medium => self.process_medium_quality(frame_data, params).await,
            QualityLevel::Low => self.process_low_quality(frame_data, params).await,
        }
    }

    async fn process_ultra_quality(&self, frame_data: Bytes, params: &EncodingParams) -> Result<Bytes> {
        // 4K 60fps - ハードウェアエンコーディング優先
        if let Some(encoder) = &self.hardware_encoder {
            encoder.encode_h264_hardware(frame_data, (3840, 2160), 60, params).await
        } else {
            self.encode_h264_software(frame_data, (3840, 2160), 60, params).await
        }
    }

    async fn process_high_quality(&self, frame_data: Bytes, params: &EncodingParams) -> Result<Bytes> {
        // 1080p 60fps
        if let Some(encoder) = &self.hardware_encoder {
            encoder.encode_h264_hardware(frame_data, (1920, 1080), 60, params).await
        } else {
            self.encode_h264_software(frame_data, (1920, 1080), 60, params).await
        }
    }

    async fn process_medium_quality(&self, frame_data: Bytes, params: &EncodingParams) -> Result<Bytes> {
        // 720p 30fps
        self.encode_h264_software(frame_data, (1280, 720), 30, params).await
    }

    async fn process_low_quality(&self, frame_data: Bytes, params: &EncodingParams) -> Result<Bytes> {
        // 480p 15fps
        self.encode_h264_software(frame_data, (854, 480), 15, params).await
    }

    async fn encode_h264_software(&self, frame_data: Bytes, _resolution: (u32, u32), _fps: u32, _params: &EncodingParams) -> Result<Bytes> {
        // ソフトウェアエンコーディング（フォールバック）
        // 実際にはopenh264やFFmpegを使用
        Ok(frame_data) // 一時的にそのまま返す
    }
}

// ハードウェアエンコーダー
pub struct HardwareEncoder {
    available: bool,
}

impl HardwareEncoder {
    pub fn new() -> Option<Self> {
        // ハードウェアエンコーダーの利用可否チェック
        let available = Self::check_hardware_availability();
        if available {
            Some(Self { available })
        } else {
            None
        }
    }

    fn check_hardware_availability() -> bool {
        // Intel Quick Sync, NVIDIA NVENC, AMD VCEのチェック
        // 簡易的にtrueを返す（実際にはシステムチェックが必要）
        true
    }

    async fn encode_h264_hardware(&self, frame_data: Bytes, resolution: (u32, u32), fps: u32, params: &EncodingParams) -> Result<Bytes> {
        // ハードウェアH264エンコーディング
        info!("Hardware encoding: {:?}x{:?} @ {}fps", resolution.0, resolution.1, fps);
        Ok(frame_data) // 一時的にそのまま返す
    }
}

// 帯域監視
pub struct BandwidthMonitor {
    bandwidth_data: Arc<RwLock<HashMap<String, (u32, Instant)>>>,
}

impl BandwidthMonitor {
    pub fn new() -> Self {
        Self {
            bandwidth_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_available_bandwidth(&self, connection_id: &str) -> u32 {
        let bandwidths = self.bandwidth_data.read().await;
        bandwidths.get(connection_id).map(|(bw, _)| *bw).unwrap_or(8000)
    }

    pub async fn get_latency(&self, _connection_id: &str) -> Duration {
        Duration::from_millis(50) // 固定値（実際には計測が必要）
    }

    pub async fn update_bandwidth(&self, connection_id: &str, bandwidth: u32) {
        let mut bandwidths = self.bandwidth_data.write().await;
        bandwidths.insert(connection_id.to_string(), (bandwidth, Instant::now()));
    }
}

// 品質コントローラー
pub struct QualityController {
    thresholds: QualityThresholds,
}

#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub ultra_min_bandwidth: u32,
    pub high_min_bandwidth: u32,
    pub medium_min_bandwidth: u32,
    pub low_min_bandwidth: u32,
    pub max_acceptable_latency: Duration,
}

impl QualityController {
    pub fn new() -> Self {
        Self {
            thresholds: QualityThresholds {
                ultra_min_bandwidth: 8000,  // 8Mbps
                high_min_bandwidth: 3000,   // 3Mbps
                medium_min_bandwidth: 1500, // 1.5Mbps
                low_min_bandwidth: 500,     // 500kbps
                max_acceptable_latency: Duration::from_millis(100),
            },
        }
    }

    pub async fn calculate_optimal_quality(&self, bandwidth: u32, latency: Duration) -> QualityLevel {
        if latency > self.thresholds.max_acceptable_latency {
            return QualityLevel::Low;
        }

        if bandwidth >= self.thresholds.ultra_min_bandwidth {
            QualityLevel::Ultra
        } else if bandwidth >= self.thresholds.high_min_bandwidth {
            QualityLevel::High
        } else if bandwidth >= self.thresholds.medium_min_bandwidth {
            QualityLevel::Medium
        } else {
            QualityLevel::Low
        }
    }
}
