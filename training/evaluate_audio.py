import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.audio.model import AudioCNN, preprocess_audio

class SirenDataset(Dataset):
    def __init__(self, data_dir):
        self.files = []
        self.labels = []
        
        siren_dir = Path(data_dir) / 'siren'
        not_siren_dir = Path(data_dir) / 'not_siren'
        
        if siren_dir.exists():
            for f in siren_dir.glob('*.wav'):
                self.files.append(str(f))
                self.labels.append(1.0)
                
        if not_siren_dir.exists():
            for f in not_siren_dir.glob('*.wav'):
                self.files.append(str(f))
                self.labels.append(0.0)
                
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        spec = preprocess_audio(file_path)
        if spec is None:
            return torch.zeros(1, 64, 94), torch.tensor(label).float()
        return spec, torch.tensor(label).float()

def evaluate():
    DATA_DIR = "data/audio"
    MODEL_PATH = "models/audio_cnn.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(MODEL_PATH):
        # Fallback to results if models/ not found
        MODEL_PATH = "results/audio/audio_cnn.pth"
        
    print(f"Evaluating model: {MODEL_PATH}")
    
    dataset = SirenDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = AudioCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            outputs = model(specs)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total
    print(f"Overall Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate()
