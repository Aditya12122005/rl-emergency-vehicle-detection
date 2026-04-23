import logging
import re
from typing import List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Check if easyocr is available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed. OCR features disabled. Install with: pip install easyocr")

class OCRPreFilter:
    """OCR-based pre-filter to detect emergency text before running YOLO."""
    
    EMERGENCY_TEXT_PATTERNS = [
        r"ambulance",
        r"police",
        r"fire",
        r"emergency",
        r"rescue",
        r"108",   # Indian ambulance
        r"100",   # Police
        r"101",   # Fire
        r"102",   # Ambulance
        r"112",   # EU Emergency
        r"911",   # US emergency
        r"999",   # UK emergency
    ]
    
    def __init__(self, languages: List[str] = ["en"], use_gpu: bool = False):
        if not EASYOCR_AVAILABLE:
            raise ImportError("easyocr not installed. Run: pip install easyocr")
        
        logger.info(f"Initializing OCR with languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.EMERGENCY_TEXT_PATTERNS]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def detect_emergency_text(self, image: np.ndarray) -> Tuple[bool, List[str], List[dict]]:
        """
        Detect emergency-related text in image.
        
        Returns:
            Tuple of (has_emergency_text, found_keywords, all_text_regions)
        """
        # Optimization: If image is too large, resize it? 
        # Or rely on caller to pass crops.
        
        preprocessed = self.preprocess_image(image)
        
        try:
            # Use detail=0 for faster simple list output if we didn't need boxes
            # But we need boxes.
            results = self.reader.readtext(preprocessed)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return False, [], []
        
        found_keywords = []
        text_regions = []
        
        for (bbox, text, conf) in results:
            if conf < 0.3:
                continue
            
            text_regions.append({
                "text": text,
                "confidence": conf,
                "bbox": bbox,
            })
            
            for pattern in self.patterns:
                if pattern.search(text):
                    found_keywords.append(text)
                    break
        
        has_emergency = len(found_keywords) > 0
        return has_emergency, found_keywords, text_regions
