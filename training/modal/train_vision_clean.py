"""
First-Principles Vision Training on Modal (GPU).

Core Logic:
1.  **Problem**: Object Detection. We need to find specific objects (Ambulance, Fire, Police) and differentiate them from generic 'Vehicles'.
2.  **Solution**: YOLO (You Only Look Once). It is the standard for real-time detection. Writing a detector from scratch (Backbone+FPN+Head+Loss) is inefficient and prone to errors compared to using a proven implementation like Ultralytics.
3.  **Data**: The raw data is messy (24 classes). The first step of ANY robust ML pipeline is Data Cleaning. We will normalize this to 4 classes: [Ambulance, FireTruck, Police, Vehicle].
4.  **Infrastructure**: GPU is required. We use Modal.
"""

import modal
import os
import shutil
from pathlib import Path

# Definition
app = modal.App("emergency-vehicle-vision-clean")
volume = modal.Volume.from_name("emergency-vehicle-vision-vol", create_if_missing=True)

# Environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("ultralytics", "torch", "pandas", "matplotlib", "seaborn")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 12,
    volumes={"/root/data_vol": volume},
    cpu=4.0,
    memory=16384
)
def train_vision_model(data_local_path: str):
    import torch
    from ultralytics import YOLO
    import yaml
    import shutil
    
    print("Initializing Vision Pipeline...")
    
    # --- 1. Data Preparation (The most important step) ---
    # We need to move data from the local mount (if using Mounts) or assume it's uploaded.
    # Since the dataset is large, typically in Modal we might mount it or download it.
    # For this script, we assume the dataset is uploaded via `add_local_dir` to `/root/data`.
    
    RAW_DATA_DIR = Path("/root/data")
    CLEAN_DATA_DIR = Path("/root/data_clean")
    
    if CLEAN_DATA_DIR.exists():
        shutil.rmtree(CLEAN_DATA_DIR)
    
    print("Step 1: Data Cleaning & Normalization (24 -> 4 Classes)")
    # We define our 'Ground Truth' classes
    TARGET_CLASSES = ['ambulance', 'fire_truck', 'police', 'vehicle']
    
    # Map legacy/noisy IDs to new clean IDs
    # Based on inspection of the dataset's data.yaml
    # 0: Ambulance, 1: Fire, 2: Police, 3: Vehicle
    ID_MAPPING = {
        # Ambulance variants -> 0
        2: 0, 3: 0, 4: 0,
        # Fire truck variants -> 1
        11: 1,
        # Police variants -> 2
        17: 2,
        # Generic vehicles -> 3
        0: 3, 1: 3, 7: 3, 8: 3, 9: 3, 10: 3, 21: 3, 22: 3
    }
    
    def process_dataset(split):
        src_img = RAW_DATA_DIR / split / "images"
        src_lbl = RAW_DATA_DIR / split / "labels"
        dst_img = CLEAN_DATA_DIR / split / "images"
        dst_lbl = CLEAN_DATA_DIR / split / "labels"
        
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        
        # Copy images (Symbolic link would be faster, but copy is safer for modification)
        # Actually, let's symlink images to save space/time
        for img in src_img.glob("*"):
            (dst_img / img.name).symlink_to(img)
            
        # Remap Labels
        for lbl in src_lbl.glob("*.txt"):
            if lbl.name == "classes.txt": continue
            
            new_lines = []
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    try:
                        cls_id = int(float(parts[0]))
                        if cls_id in ID_MAPPING:
                            new_id = ID_MAPPING[cls_id]
                            new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                    except:
                        pass
            
            if new_lines:
                with open(dst_lbl / lbl.name, 'w') as f:
                    f.writelines(new_lines)
                    
    for split in ['train', 'valid', 'test']:
        if (RAW_DATA_DIR / split).exists():
            print(f"  Processing {split} set...")
            process_dataset(split)
            
    # Create new data.yaml
    yaml_content = {
        'train': str(CLEAN_DATA_DIR / 'train' / 'images'),
        'val': str(CLEAN_DATA_DIR / 'valid' / 'images'),
        'test': str(CLEAN_DATA_DIR / 'test' / 'images'),
        'nc': 4,
        'names': TARGET_CLASSES
    }
    with open(CLEAN_DATA_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f)
        
    print("Data preparation complete.")
    
    # --- 2. Model Training ---
    print("\nStep 2: Training YOLOv8s (Small)")
    # We choose 'Small' (s) over 'Nano' (n) as a first-principle trade-off:
    # We need higher accuracy for safety-critical Emergency detection.
    # 's' is still fast enough for real-time (30fps+) on modern edge hardware.
    
    model = YOLO("yolov8s.pt") 
    
    PROJECT_DIR = "/root/runs"
    NAME = "emergency_detection"
    
    results = model.train(
        data=str(CLEAN_DATA_DIR / "data.yaml"),
        epochs=50,           # Sufficient for convergence
        imgsz=640,           # Standard resolution
        batch=32,            # Efficient for A10G
        device=0,
        project=PROJECT_DIR,
        name=NAME,
        patience=15,         # Stop if not improving
        save=True,
        verbose=True,
        pretrained=True,     # Transfer learning is standard practice
        
        # Robustness Augmentations (Standard but effective)
        degrees=5.0,         # Handle slight camera tilt
        shear=2.0,           # Handle perspective shift
        mosaic=1.0,          # Good for detecting objects in context
    )
    
    print("Training complete.")
    
    # --- 3. Artifact Handling ---
    # Move results to volume for download
    # We just return the path, and the local script will download
    return str(Path(PROJECT_DIR) / NAME)

@app.local_entrypoint()
def main():
    print("Uploading data and starting remote training...")
    
    # We mount the local 'data' directory to '/root/data' in the container
    data_path = Path("data")
    
    # Run the function
    # We use `with_options` to attach the mount dynamically if we wanted, 
    # but here we defined it in the app. 
    # Actually, `add_local_dir` in definition is static. 
    # Let's redefine the app structure slightly to ensure data upload.
    
    # Since we are redefining the flow, we need to pass the local dir to the remote function context
    # Re-declaring image to include local dir here for clarity of the script
    
    # Note: For large datasets, creating a Volume and uploading once is better.
    # But for simplicity/automation here, we upload context.
    
    remote_train = train_vision_model.with_options(
        image=image.add_local_dir(data_path, remote_path="/root/data")
    )
    
    run_dir = remote_train.remote(str(data_path))
    
    print(f"\nTraining finished. Results stored at: {run_dir}")
    
    # Download Results
    print("Downloading artifacts...")

    # Define a function to read files from the volume/container
    # Since the files are inside the container's ephemeral storage or volume...
    # We need a separate function to retrieve them.
    
    # Helper to download
    download_results.remote(run_dir)

@app.function(image=image, volumes={"/root/data_vol": volume})
def download_results(run_dir: str):
    from pathlib import Path
    run_path = Path(run_dir)
    
    # We return a dict of {filename: binary_content}
    # Limited to small files (graphs) and the best model
    
    artifacts = {}
    
    # 1. The Model
    best_pt = run_path / "weights" / "best.pt"
    if best_pt.exists():
        with open(best_pt, "rb") as f:
            artifacts["models/best.pt"] = f.read()
            
    # 2. The Graphs
    for plot in run_path.glob("*.png"):
        with open(plot, "rb") as f:
            artifacts[f"results/vision/{plot.name}"] = f.read()
            
    return artifacts

# We need to orchestrate the download in local_entrypoint
@app.local_entrypoint()
def run():
    print("="*60)
    print("Vision Model Training (First Principles Re-write)")
    print("="*60)
    
    # 1. Train
    # We attach the data directory here
    print(" syncing data and training...")
    # Note: 'data' folder in current dir
    train_func = train_vision_model.with_options(
        image=image.add_local_dir("data", remote_path="/root/data")
    )
    remote_run_path = train_func.remote("data")
    
    # 2. Download
    print(f" Training done. Downloading results from {remote_run_path}...")
    
    # We need a downloader function that runs on the SAME container/volume context 
    # OR simply returns the bytes.
    # The train function *could* return the bytes, but that might be too large for one object.
    # Let's define a specific downloader that takes the path.
    
    # Since the training used ephemeral storage (unless we used volume), 
    # we must actually return the data FROM the training function or use a Volume.
    # In the `train_vision_model` above, I didn't strictly write to Volume.
    # Let's modify `train_vision_model` to Write to Volume, then Download from Volume.
    pass

# --- RE-WRITING FOR CORRECT VOLUME USAGE ---
# To make this robust:
# 1. Train writes to /root/data_vol/runs/...
# 2. Download reads from /root/data_vol/runs/...

@app.function(
    image=image.add_local_dir("data", remote_path="/root/data"),
    gpu="A10G",
    timeout=3600 * 12,
    volumes={"/root/data_vol": volume},
)
def train_and_save():
    # ... (Imports and Data Cleaning Logic from above) ...
    import torch
    from ultralytics import YOLO
    import yaml
    import shutil
    
    # ... [Same Data Cleaning Code] ...
    RAW_DATA_DIR = Path("/root/data")
    CLEAN_DATA_DIR = Path("/root/data_clean")
    if CLEAN_DATA_DIR.exists(): shutil.rmtree(CLEAN_DATA_DIR)
    
    TARGET_CLASSES = ['ambulance', 'fire_truck', 'police', 'vehicle']
    ID_MAPPING = {2:0, 3:0, 4:0, 11:1, 17:2, 0:3, 1:3, 7:3, 8:3, 9:3, 10:3, 21:3, 22:3}
    
    # Quick inline processing
    for split in ['train', 'valid', 'test']:
        s_img, s_lbl = RAW_DATA_DIR/split/"images", RAW_DATA_DIR/split/"labels"
        d_img, d_lbl = CLEAN_DATA_DIR/split/"images", CLEAN_DATA_DIR/split/"labels"
        d_img.mkdir(parents=True, exist_ok=True); d_lbl.mkdir(parents=True, exist_ok=True)
        
        if s_img.exists():
            for i in s_img.glob("*"): (d_img/i.name).symlink_to(i)
            for l in s_lbl.glob("*.txt"):
                if l.name == "classes.txt": continue
                nl = []
                with open(l) as f:
                    for line in f:
                        p = line.split()
                        if p and int(float(p[0])) in ID_MAPPING:
                            nl.append(f"{ID_MAPPING[int(float(p[0]))]} {' '.join(p[1:])}\n")
                if nl: 
                    with open(d_lbl/l.name, 'w') as f: f.writelines(nl)

    yaml_data = {'train': str(CLEAN_DATA_DIR/'train'/'images'), 'val': str(CLEAN_DATA_DIR/'valid'/'images'), 
                 'test': str(CLEAN_DATA_DIR/'test'/'images'), 'nc': 4, 'names': TARGET_CLASSES}
    with open(CLEAN_DATA_DIR/"data.yaml", "w") as f: yaml.dump(yaml_data, f)
    
    # Training
    print("Starting YOLOv8s Training...")
    model = YOLO("yolov8s.pt")
    
    # WRITE TO VOLUME
    PROJECT_DIR = "/root/data_vol/runs"
    NAME = "emergency_detect_clean"
    
    model.train(
        data=str(CLEAN_DATA_DIR/"data.yaml"),
        epochs=50, batch=32, imgsz=640, device=0,
        project=PROJECT_DIR, name=NAME,
        patience=15, save=True, pretrained=True,
        degrees=5.0, shear=2.0, mosaic=1.0,
        verbose=True
    )
    
    # Commit volume
    volume.commit()
    return str(Path(PROJECT_DIR) / NAME)

@app.function(image=image, volumes={"/root/data_vol": volume})
def get_results(run_path_str):
    from pathlib import Path
    run_path = Path(run_path_str)
    artifacts = {}
    
    # Grab Best Model
    if (run_path/"weights"/"best.pt").exists():
        with open(run_path/"weights"/"best.pt", "rb") as f:
            artifacts["models/best.pt"] = f.read()
            
    # Grab Plots and Batch Images
    # We capture .png (charts), .jpg (batch samples), and .csv (metrics)
    for ext in ["*.png", "*.jpg", "*.csv"]:
        for p in run_path.glob(ext):
            with open(p, "rb") as f:
                artifacts[f"results/vision/{p.name}"] = f.read()
            
    return artifacts

@app.local_entrypoint()
def execute():
    print("--- Starting Remote Training ---")
    run_path = train_and_save.remote()
    print(f"--- Training Done. Path: {run_path} ---")
    print("--- Downloading Results ---")
    
    data = get_results.remote(run_path)
    
    for path, content in data.items():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(content)
        print(f"Saved: {path}")
