import os
import yaml
import glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def plot_dataset_statistics(data_yaml_path, output_dir):
    """
    Generates statistics and plots for the YOLO dataset.
    """
    print(f"Analyzing dataset from {data_yaml_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    class_names = data_config['names']
    train_path = data_config['train']
    
    # Handle relative paths if necessary (assuming relative to data.yaml location or absolute)
    # In Modal, paths are usually absolute /root/data/...
    
    # Collect labels
    label_counts = {name: 0 for name in class_names}
    bbox_sizes = []
    
    # Find label files (assuming standard YOLO structure: images/../labels)
    # train_path usually points to images directory
    images_dir = Path(train_path)
    # Try to find labels dir. Usually parallel to images dir.
    # Structure: data/train/images -> data/train/labels
    labels_dir = images_dir.parent / 'labels'
    
    if not labels_dir.exists():
        print(f"Warning: Could not find labels directory at {labels_dir}")
        return

    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files.")
    
    for label_file in tqdm(label_files, desc="Parsing labels"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    if 0 <= cls_id < len(class_names):
                        label_counts[class_names[cls_id]] += 1
                        bbox_sizes.append({'width': w, 'height': h, 'class': class_names[cls_id]})

    # 1. Class Distribution Plot
    plt.figure(figsize=(12, 8))
    df_counts = pd.DataFrame(list(label_counts.items()), columns=['Class', 'Count'])
    df_counts = df_counts.sort_values('Count', ascending=False)
    
    sns.barplot(data=df_counts, x='Count', y='Class', palette='viridis')
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Number of Instances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Bounding Box Size Distribution
    if bbox_sizes:
        df_bbox = pd.DataFrame(bbox_sizes)
        
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=df_bbox, x='width', y='height', alpha=0.3, hue='class', legend=False)
        plt.title('Bounding Box Sizes (Normalized)')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5) # Diagonal
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bbox_distribution.png'))
        plt.close()
        
        # 3. BBox Area Distribution
        df_bbox['area'] = df_bbox['width'] * df_bbox['height']
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df_bbox, x='area', bins=50, kde=True)
        plt.title('Bounding Box Area Distribution')
        plt.xlabel('Area (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bbox_area_distribution.png'))
        plt.close()

    print(f"Dataset analysis plots saved to {output_dir}")
