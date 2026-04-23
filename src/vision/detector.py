import logging
from typing import Union, List, Optional
import numpy as np
import cv2
from ultralytics import YOLO
from src.vision.ocr import OCRPreFilter, EASYOCR_AVAILABLE

logger = logging.getLogger(__name__)

class EmergencyVehicleDetector:
    """Detector class for emergency vehicle detection using trained YOLOv8 model."""

    EMERGENCY_CLASSES = {
        "ambulance", 
        "fire_truck", 
        "police"
    }
    
    # Classes to show in the annotated video/image
    VISUALIZATION_CLASSES = {
        "ambulance", 
        "fire_truck", 
        "police", 
        "non_emergency_vehicle"
    }

    def __init__(
        self, 
        model_path: str, 
        confidence: float = 0.5, 
        device: Union[int, str] = "cpu",
        use_ocr: bool = False,
        ocr_languages: List[str] = ["en"],
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = self._load_model()
        
        # Initialize OCR pre-filter if requested
        self.use_ocr = use_ocr and EASYOCR_AVAILABLE
        self.ocr_filter = None
        if self.use_ocr:
            try:
                self.ocr_filter = OCRPreFilter(languages=ocr_languages, use_gpu=False)
                logger.info("OCR pre-filter enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize OCR: {e}")
                self.use_ocr = False

    def _load_model(self) -> YOLO:
        """Load the trained YOLO model."""
        logger.info(f"Loading model from: {self.model_path}")
        return YOLO(self.model_path)

    def predict(
        self,
        source: Union[str, np.ndarray, List[str]],
        save: bool = False,
        save_dir: Optional[str] = None,
        tracker: bool = False,
    ) -> list:
        """
        Run prediction on image(s) or video.

        Args:
            source: Image path, numpy array, or list of paths
            save: Whether to save results
            save_dir: Directory to save results
            tracker: Whether to use object tracking (for video)

        Returns:
            List of prediction results
        """
        if tracker:
            results = self.model.track(
                source=source,
                conf=self.confidence,
                device=self.device,
                save=save,
                project=save_dir,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml"
            )
        else:
            results = self.model.predict(
                source=source,
                conf=self.confidence,
                device=self.device,
                save=save,
                project=save_dir,
                verbose=False,
            )

        return results

    def detect(self, source: Union[str, np.ndarray], tracker: bool = False) -> dict:
        """
        Detect emergency vehicles and return structured results.
        
        If OCR is enabled, first checks for emergency text, then runs YOLO.

        Args:
            source: Image path or numpy array
            tracker: Whether to use tracking (for video frames)

        Returns:
            Dictionary with detection results and emergency status
        """
        if isinstance(source, str):
            image = cv2.imread(source)
        else:
            image = source
        
        # OCR pre-filter check
        # Disabled pre-filter OCR for performance. 
        # OCR is now run on crops in the pipeline after detection.
        ocr_emergency = False
        ocr_keywords = []
        ocr_texts = []
        
        # Run YOLO detection
        results = self.predict(image, tracker=tracker)[0]

        detections = []
        yolo_emergency = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            detection = {
                "class": cls_name,
                "confidence": conf,
                "bbox": xyxy.tolist(),
            }
            detections.append(detection)

            if cls_name in self.EMERGENCY_CLASSES:
                yolo_emergency = True

        # Combined result: emergency if YOLO OR OCR detects it
        is_emergency_vehicle = yolo_emergency or ocr_emergency

        # Filter boxes for visualization
        # We create a copy of results to not affect the original detection logic
        # but we want plot() to only show specific classes
        
        # Get indices of boxes that are in VISUALIZATION_CLASSES
        keep_indices = []
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            if cls_name in self.VISUALIZATION_CLASSES:
                keep_indices.append(i)
        
        # Plot only kept boxes
        # Note: results.plot() doesn't support filtering easily, so we might need to hack it
        # or just use the conf argument if we could, but we can't filter by class.
        # Instead, we can temporarily replace results.boxes with filtered boxes
        
        original_boxes = results.boxes
        if keep_indices:
            results.boxes = results.boxes[keep_indices]
        else:
            # If no relevant classes, show nothing (empty boxes)
            # We can't easily create empty Boxes object, so we might just plot nothing
            # or plot original if we want to be safe, but user wants to hide text.
            # Let's try to set it to empty if possible, or just accept that we might show nothing.
            # Actually, results.boxes[[]] should work if supported.
            try:
                results.boxes = results.boxes[keep_indices]
            except:
                pass # Fallback
                
        annotated_frame = results.plot()
        
        # Restore original boxes for logic
        results.boxes = original_boxes

        result = {
            "is_emergency": is_emergency_vehicle,
            "yolo_emergency": yolo_emergency,
            "detections": detections,
            "num_detections": len(detections),
            "annotated_frame": annotated_frame
        }
        
        if self.use_ocr:
            result["ocr_emergency"] = ocr_emergency
            result["ocr_keywords"] = ocr_keywords
            result["ocr_texts"] = [t["text"] for t in ocr_texts]
        
        return result
