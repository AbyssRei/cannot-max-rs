use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("设备错误: {0}")]
    Device(String),

    #[error("识别错误: {0}")]
    Recognition(String),

    #[error("预测错误: {0}")]
    Prediction(String),

    #[error("采集错误: {0}")]
    Fetch(String),

    #[error("存储错误: {0}")]
    Storage(String),

    #[error("资源错误: {0}")]
    Resource(String),

    #[error("登录错误: {0}")]
    Login(String),

    #[error("OCR不可用: {0}")]
    OcrUnavailable(String),

    #[error("模型加载失败: {0}")]
    ModelLoad(String),

    #[error("推理错误: {0}")]
    Inference(String),

    #[error("配置错误: {0}")]
    Config(String),

    #[error("IO错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("序列化错误: {0}")]
    Serialize(#[from] ron::Error),

    #[error("JSON错误: {0}")]
    Json(#[from] serde_json::Error),

    #[error("图像错误: {0}")]
    Image(#[from] image::ImageError),

    #[error("CSV错误: {0}")]
    Csv(#[from] csv::Error),

    #[error("自定义ROI缺失")]
    CustomRoiMissing,

    #[error("ROI框选区域为零")]
    RoiSelectionZeroArea,

    #[error("示例图片加载失败: {0}")]
    ExampleImageLoadFailed(String),

    #[error("{0}")]
    General(String),
}

pub type AppResult<T> = Result<T, AppError>;
