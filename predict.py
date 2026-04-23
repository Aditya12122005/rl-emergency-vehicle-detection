#!/usr/bin/env python3
"""Prediction script for emergency vehicle detection."""

import sys
import os
import argparse
from pathlib import Path

from src.pipeline.predictor import EmergencyInferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run emergency vehicle detection on images or videos"
    )
    # UPDATED DEFAULT PATH
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--audio-model",
        type=str,
        default="models/audio_cnn.pth",
        help="Path to trained audio model",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to image, video, or directory",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (0, 1, 2... for GPU or 'cpu')",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR pre-filter to detect emergency text (requires easyocr)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/predictions",
        help="Directory to save results",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Process as video",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (for image+audio mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (for video processing)",
    )
    return parser.parse_args()


def interactive_mode():
    print("=" * 60)
    print("Emergency Vehicle Detection - Prediction Hub")
    print("=" * 60)
    
    # 1. Source
    default_source = "predict_data"
    print(f"\nEnter source path (default: {default_source}):")
    print(f"(You can enter a full path, or a filename inside '{default_source}')")
    source_input = input("> ").strip()
    
    if not source_input:
        source = default_source
    else:
        # Check if it's a direct path
        if os.path.exists(source_input):
            source = source_input
        # Check if it's a file in predict_data
        elif os.path.exists(os.path.join(default_source, source_input)):
            source = os.path.join(default_source, source_input)
        else:
            print(f"Error: Source '{source_input}' not found.")
            sys.exit(1)
        
    # 2. Mode (Image/Video)
    is_video = False
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video = True
            print(f"Detected video file: {source}")
        else:
            print(f"Detected image file: {source}")
    elif os.path.isdir(source):
        print(f"Detected directory: {source}")
    
    # Ask about audio for single image
    audio_path = None
    if not is_video and os.path.isfile(source):
        print("\nDo you want to provide an audio file for siren detection?")
        print("(Enter path, 'n' to skip, or filename in 'predict_data')")
        audio_input = input("> ").strip()
        
        if audio_input and audio_input.lower() != 'n':
            if os.path.exists(audio_input):
                audio_path = audio_input
            elif os.path.exists(os.path.join(default_source, audio_input)):
                audio_path = os.path.join(default_source, audio_input)
            else:
                print(f"Warning: Audio file '{audio_input}' not found. Proceeding without audio.")

    # 3. Models
    # UPDATED DEFAULT PATH
    default_vision = "models/best.pt"
    default_audio = "models/audio_cnn.pth"
    
    print(f"\nVision Model (default: {default_vision}):")
    vision_model = input("> ").strip() or default_vision
    
    print(f"Audio Model (default: {default_audio}):")
    audio_model = input("> ").strip() or default_audio
    
    # 4. Options
    use_ocr = input("Enable OCR? (y/n): ").lower().strip() == 'y'
    save_results = input("Save results? (y/n): ").lower().strip() == 'y'
    
    conf_input = input("Confidence Threshold (0.0-1.0) [default: 0.4]: ").strip()
    try:
        confidence = float(conf_input) if conf_input else 0.4
    except ValueError:
        print("Invalid number. Using default 0.4")
        confidence = 0.4
    
    return argparse.Namespace(
        source=source,
        model=vision_model,
        audio_model=audio_model,
        video=is_video,
        audio=audio_path,
        ocr=use_ocr,
        save=save_results,
        save_dir="results/predictions",
        confidence=confidence,
        device="cpu",
        output=None,
        show=False
    )

def main():
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        args = interactive_mode()

    # Pre-flight check for models
    if not os.path.exists(args.model):
        print(f"\nError: Vision model not found at '{args.model}'")
        print("Please check the path or run 'python train.py' first.")
        return

    device = args.device if args.device == "cpu" else int(args.device)
    
    try:
        pipeline = EmergencyInferencePipeline(
            vision_model_path=args.model,
            audio_model_path=args.audio_model,
            confidence=args.confidence,
            device=device,
            use_ocr=args.ocr,
        )
    except Exception as e:
        print(f"\nFailed to initialize pipeline: {e}")
        return

    if args.video:
        output_path = args.output
        if not output_path and args.save:
            # Generate output path if saving is enabled but no specific output path provided
            os.makedirs(args.save_dir, exist_ok=True)
            filename = Path(args.source).stem
            output_path = str(Path(args.save_dir) / f"{filename}_out.mp4")
            print(f"Saving video result to: {output_path}")

        summary = pipeline.process_video(
            video_path=args.source,
            output_path=output_path,
            show=args.show,
        )
        
        print("\n" + "="*40)
        print("FINAL REPORT")
        print("="*40)
        print(f"Total Frames Processed:              {summary['total_frames']}")
        print(f"Emergency Frames Detected:           {summary['emergency_frames']}")
        print("-" * 40)
        print(f"Emergency Vehicle Detected (Visual): {'Yes' if summary['vehicle_detected'] else 'No'}")
        print(f"Siren Detected (Audio):              {'Yes' if summary['sound_detected'] else 'No'}")
        
        status = "No Emergency Detected"
        if summary['vehicle_detected'] and summary['sound_detected']:
            status = "Confirmed Emergency Vehicle with Siren"
        elif summary['vehicle_detected']:
            status = "Emergency Vehicle Detected (Siren OFF)"
        elif summary['sound_detected']:
            status = "Siren Detected (Vehicle not visible)"
            
        print(f"Overall Status:                      {status}")
        print("="*40 + "\n")
        
    else:
        source_path = Path(args.source)

        if source_path.is_dir():
            sources = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        else:
            sources = [source_path]
            
        if args.save:
            os.makedirs(args.save_dir, exist_ok=True)

        for src in sources:
            result = pipeline.predict_single(str(src), audio_source=args.audio)
            print(f"\n{src.name}:")
            print(f"  Status: {result['emergency_status']}")
            
            # Explicitly show audio result if available
            if args.audio:
                audio_msg = "YES" if result['siren_detected'] else "NO"
                if 'siren_confidence' in result:
                    print(f"  Siren Detected: {audio_msg} (Conf: {result['siren_confidence']:.2f})")
                else:
                    print(f"  Siren Detected: {audio_msg}")
            
            if args.ocr and "ocr_emergency" in result:
                print(f"  YOLO detected: {result['yolo_emergency']}")
                print(f"  OCR detected:  {result['ocr_emergency']}")
                if result.get("ocr_keywords"):
                    print(f"  OCR keywords:  {', '.join(result['ocr_keywords'])}")
            
            print(f"  Detections: {result['num_detections']}")

            for det in result["detections"]:
                print(f"    - {det['class']}: {det['confidence']:.2f}")

            if args.save:
                import cv2
                filename = src.stem
                output_path = str(Path(args.save_dir) / f"{filename}_out.jpg")
                cv2.imwrite(output_path, result['annotated_frame'])
                print(f"  Saved result to: {output_path}")


if __name__ == "__main__":
    main()
