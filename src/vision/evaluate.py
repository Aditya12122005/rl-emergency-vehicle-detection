"""Evaluation module for YOLO emergency vehicle detection."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_images: int = 0
    processing_time: float = 0.0
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def avg_time_per_image(self) -> float:
        return self.processing_time / self.total_images if self.total_images > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "total_images": self.total_images,
            "accuracy": round(self.accuracy * 100, 2),
            "precision": round(self.precision * 100, 2),
            "recall": round(self.recall * 100, 2),
            "f1_score": round(self.f1_score * 100, 2),
            "avg_time_per_image_ms": round(self.avg_time_per_image * 1000, 2),
            "total_time_seconds": round(self.processing_time, 2),
        }


class EmergencyVehicleEvaluator:
    """Evaluates YOLO emergency vehicle detection accuracy."""
    
    EMERGENCY_CLASSES = {
        "ambulance", "ambulance_108", "ambulance_SOL", "ambulance_lamp", "ambulance_text",
        "fire_truck", "fireladder", "firelamp", "firesymbol", "firewriting",
        "police", "police_lamp", "police_lamp_ON"
    }
    
    EMERGENCY_CLASS_IDS = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17, 18, 19}
    
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        device: str = "cpu",
    ):
        logger.info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.metrics = EvaluationMetrics()
    
    def _get_ground_truth(self, label_path: Path) -> bool:
        """Determine if image contains emergency vehicle from label file."""
        if not label_path.exists():
            return False
        
        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in self.EMERGENCY_CLASS_IDS:
                            return True
        except Exception as e:
            logger.warning(f"Error reading label {label_path}: {e}")
        
        return False
    
    def _predict_emergency(self, image: np.ndarray) -> bool:
        """Predict if image contains emergency vehicle."""
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            if cls_name in self.EMERGENCY_CLASSES:
                return True
        
        return False
    
    def _update_metrics(self, predicted: bool, ground_truth: bool):
        """Update metrics based on prediction and ground truth."""
        if predicted and ground_truth:
            self.metrics.true_positives += 1
        elif predicted and not ground_truth:
            self.metrics.false_positives += 1
        elif not predicted and ground_truth:
            self.metrics.false_negatives += 1
        else:
            self.metrics.true_negatives += 1
    
    def evaluate_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        max_images: Optional[int] = None,
    ) -> Dict[str, dict]:
        """Evaluate model on a dataset."""
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Evaluating {len(image_files)} images...")
        
        self.metrics = EvaluationMetrics()
        
        for img_path in tqdm(image_files, desc="Evaluating"):
            label_path = labels_path / f"{img_path.stem}.txt"
            ground_truth = self._get_ground_truth(label_path)
            
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Cannot read image: {img_path}")
                continue
            
            start_time = time.time()
            predicted = self._predict_emergency(image)
            elapsed = time.time() - start_time
            
            self._update_metrics(predicted, ground_truth)
            self.metrics.total_images += 1
            self.metrics.processing_time += elapsed
        
        return {"metrics": self.metrics.to_dict()}
    
    def plot_results(self, results: dict, output_dir: str, training_dir: Optional[str] = None):
        """Generate and save visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = results["metrics"]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Metrics Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        bars = ax.bar(metric_names, values, color=colors, width=0.6)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Emergency Vehicle Detection - Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics.png', dpi=150)
        plt.close()
        
        # 2. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = np.array([
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ])
        
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Emergency', 'Emergency'], fontsize=11)
        ax.set_yticklabels(['Non-Emergency', 'Emergency'], fontsize=11)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=16)
        
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=150)
        plt.close()
        
        # 3. Training curves (if training_dir provided)
        if training_dir:
            self._plot_training_curves(training_dir, output_path)
        
        logger.info(f"Plots saved to: {output_path}")
    
    def _plot_training_curves(self, training_dir: str, output_path: Path):
        """Plot training loss and accuracy curves from CSV results."""
        import pandas as pd
        
        results_csv = Path(training_dir) / "results.csv"
        
        if not results_csv.exists():
            logger.warning(f"Training results not found: {results_csv}")
            return
        
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        epochs = df.index + 1
        
        # Training & Validation Loss
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1 = axes[0]
        if 'train/box_loss' in df.columns:
            train_loss = df['train/box_loss'] + df.get('train/cls_loss', 0) + df.get('train/dfl_loss', 0)
            val_loss = df['val/box_loss'] + df.get('val/cls_loss', 0) + df.get('val/dfl_loss', 0)
            
            ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        
        # mAP plot (accuracy proxy)
        ax2 = axes[1]
        if 'metrics/mAP50(B)' in df.columns:
            ax2.plot(epochs, df['metrics/mAP50(B)'] * 100, 'g-', label='mAP@50', linewidth=2)
            if 'metrics/mAP50-95(B)' in df.columns:
                ax2.plot(epochs, df['metrics/mAP50-95(B)'] * 100, 'purple', label='mAP@50-95', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('mAP (%)', fontsize=12)
            ax2.set_title('Model Accuracy (mAP) Over Training', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_path / 'training_curves.png', dpi=150)
        plt.close()
        
        # Precision & Recall over epochs
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'metrics/precision(B)' in df.columns:
            ax.plot(epochs, df['metrics/precision(B)'] * 100, 'b-', label='Precision', linewidth=2)
            ax.plot(epochs, df['metrics/recall(B)'] * 100, 'r-', label='Recall', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title('Precision & Recall Over Training', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_path / 'precision_recall_curve.png', dpi=150)
        plt.close()
        
        logger.info("Training curves plotted successfully")
    
    def print_results(self, results: dict):
        """Print formatted evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        metrics = results["metrics"]
        
        print(f"\n[Performance Metrics]")
        print("-" * 40)
        print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}%")
        print(f"  Recall:    {metrics['recall']:.2f}%")
        print(f"  F1 Score:  {metrics['f1_score']:.2f}%")
        
        print(f"\n[Confusion Matrix]")
        print("-" * 40)
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        print(f"\n[Performance]")
        print("-" * 40)
        print(f"  Total Images:    {metrics['total_images']}")
        print(f"  Avg Time/Image:  {metrics['avg_time_per_image_ms']:.2f}ms")
        print(f"  Total Time:      {metrics['total_time_seconds']:.2f}s")
        
        print("\n" + "=" * 60)
    
    def save_results(self, results: dict, output_path: str):
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
