use crate::capture::capture_frame;
use crate::config::AppConfig;
use crate::core::{AnalysisOutput, CaptureCatalog, CaptureSource};
use crate::prediction::{CandlePredictor, Predictor};
use crate::recognition::analyze_frame;
use crate::resources::ResourceStore;

#[derive(Debug, Clone)]
pub struct AnalysisPipeline {
    model_path: std::path::PathBuf,
}

impl AnalysisPipeline {
    pub fn new(model_path: std::path::PathBuf) -> Self {
        Self { model_path }
    }

    pub fn run(
        &self,
        source: &CaptureSource,
        catalog: &CaptureCatalog,
        config: &AppConfig,
    ) -> Result<AnalysisOutput, String> {
        let resources = ResourceStore::load(config)?;
        let frame = capture_frame(source, config, catalog)?;
        let snapshot = analyze_frame(source, &frame, config.roi, &resources, config);
        let predictor = CandlePredictor::new(self.model_path.clone());
        let prediction = predictor.predict(&snapshot)?;

        Ok(AnalysisOutput {
            frame,
            snapshot,
            prediction,
        })
    }
}