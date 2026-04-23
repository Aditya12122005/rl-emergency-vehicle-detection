"""YOLOv8 Training module for emergency vehicle detection."""

import logging
import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.config import Config
from src.utils.data_processing import remap_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmergencyVehicleTrainer:
    """Trainer class for emergency vehicle detection using YOLOv8."""

    def __init__(self, config: Config, project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path.cwd()
        self.model: Optional[YOLO] = None

    def setup_model(self) -> YOLO:
        """Initialize the YOLOv8 model."""
        model_name = self.config.model.name
        # Upgrade to small model if nano is selected for better performance
        if model_name == "yolov8n.pt":
            logger.info("Upgrading model from Nano to Small for better accuracy")
            model_name = "yolov8s.pt"
            
        logger.info(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        return self.model

    def train(self) -> dict:
        """Train the model on emergency vehicle dataset."""
        if self.model is None:
            self.setup_model()

        # Prepare Dataset (Remap 24 classes -> 4 classes)
        source_data_path = self.project_root / "data"
        target_data_path = self.project_root / "data_remapped"
        
        logger.info("Preparing dataset (remapping labels to 4 classes)...")
        try:
            data_yaml_path = remap_dataset(source_data_path, target_data_path)
        except Exception as e:
            logger.error(f"Failed to remap dataset: {e}")
            logger.warning("Falling back to original data config")
            data_yaml_path = self.project_root / self.config.paths.data

        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Data config not found: {data_yaml_path}")

        logger.info("Starting training...")
        logger.info(f"Dataset: {data_yaml_path}")
        logger.info(f"Epochs: {self.config.training.epochs}")
        
        # Enhanced hyperparameters for robust video inference
        results = self.model.train(
            data=str(data_yaml_path),
            epochs=self.config.training.epochs,
            batch=self.config.training.batch_size,
            imgsz=self.config.training.image_size,
            patience=20,
            workers=self.config.training.workers,
            device=self.config.training.device,
            project=str(self.project_root / self.config.paths.project),
            name=self.config.paths.name,
            
            # Optimized Hyperparameters
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Augmentation settings (Robust for video/real-world data)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,     # Added rotation for video stability
            translate=0.1,
            scale=0.5,
            shear=2.5,       # Added shear for perspective changes
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.0,
            
            # Additional settings
            pretrained=True,
            verbose=True,
            exist_ok=True,
            plots=True,
        )

        logger.info("Training completed!")
        return results

    def validate(self, model_path: Optional[str] = None) -> dict:
        """Validate the model on the validation set."""
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            # Try to load best model from current run
            best_path = self.project_root / self.config.paths.project / self.config.paths.name / "weights" / "best.pt"
            if best_path.exists():
                self.model = YOLO(str(best_path))
            else:
                raise ValueError("No model loaded. Train first or provide model_path.")

        # Use remapped data if available
        data_yaml_path = self.project_root / "data_remapped" / "data.yaml"
        if not data_yaml_path.exists():
             data_yaml_path = self.project_root / self.config.paths.data

        logger.info(f"Running validation on {data_yaml_path}...")

        results = self.model.val(
            data=str(data_yaml_path),
            imgsz=self.config.training.image_size,
            batch=self.config.training.batch_size,
            device=self.config.training.device,
            plots=True
        )

        logger.info(f"mAP50: {results.box.map50:.4f}")
        logger.info(f"mAP50-95: {results.box.map:.4f}")

        return results

    def export(self, model_path: str, format: str = "onnx") -> str:
        """Export the model to specified format."""
        model = YOLO(model_path)
        logger.info(f"Exporting model to {format} format...")

        export_path = model.export(format=format)
        logger.info(f"Model exported to: {export_path}")

        return export_path