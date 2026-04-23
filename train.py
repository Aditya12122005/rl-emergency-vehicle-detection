#!/usr/bin/env python3
"""
Unified interactive training script for Emergency Vehicle Detection.
"""

import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 60)
    print("Emergency Vehicle Detection - Training Hub")
    print("=" * 60)

def run_command(command):
    try:
        # Add current directory to PYTHONPATH to ensure imports work
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        
        print(f"Executing: {command}\n")
        subprocess.run(command, shell=True, check=True, env=env)
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running command: {e}")
        # Don't exit immediately, allow user to see error
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)

def main():
    while True:
        clear_screen()
        print_header()
        
        print("\nWhat would you like to train?")
        print("1. Vision Model (YOLOv8 - Object Detection)")
        print("2. Audio Model (CNN - Siren Classifier)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '3':
            print("Goodbye!")
            sys.exit(0)
        
        if choice not in ['1', '2']:
            print("Invalid choice.")
            continue
            
        model_type = "vision" if choice == '1' else "audio"
        
        if model_type == "vision":
            print(f"\nWhere would you like to train the {model_type} model?")
            print("1. Local Machine (CPU/GPU)")
            print("2. Modal Cloud (High-performance GPU)")
            
            env_choice = input("\nEnter choice (1-2): ").strip()
            
            if env_choice not in ['1', '2']:
                print("Invalid choice.")
                continue
                
            is_modal = env_choice == '2'
            
            print("\n" + "-" * 60)
            print(f"Starting {model_type} training on {'Modal Cloud' if is_modal else 'Local Machine'}...")
            print("-" * 60 + "\n")
            
            if is_modal:
                cmd = "modal run training/modal/train_vision_clean.py::execute"
            else:
                cmd = f"{sys.executable} training/train_vision.py"
                
        else: # audio
            # Audio is now lightweight CNN, run locally
            print("\n" + "-" * 60)
            print("Starting audio training (Local CNN)...")
            print("-" * 60 + "\n")
            cmd = f"{sys.executable} training/train_audio.py"
                
        run_command(cmd)
        
        input("\nTraining finished. Press Enter to return to menu...")

if __name__ == "__main__":
    main()