# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from pathlib import Path
# from src.audio.model import AudioCNN, preprocess_audio
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.decomposition import PCA
# import numpy as np

# # --- 1. Dataset Class ---
# class SirenDataset(Dataset):
#     def __init__(self, data_dir):
#         self.files = []
#         self.labels = []
        
#         siren_dir = Path(data_dir) / 'siren'
#         not_siren_dir = Path(data_dir) / 'not_siren'
        
#         if not siren_dir.exists() and not not_siren_dir.exists():
#             raise FileNotFoundError(f"Audio data directory {data_dir} not found or empty.")

#         # Load Siren (Label 1)
#         if siren_dir.exists():
#             for f in siren_dir.glob('*.wav'):
#                 self.files.append(str(f))
#                 self.labels.append(1.0)
                
#         # Load Not Siren (Label 0)
#         if not_siren_dir.exists():
#             for f in not_siren_dir.glob('*.wav'):
#                 self.files.append(str(f))
#                 self.labels.append(0.0)
                
#         print(f"Found {len(self.files)} files for audio training.")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         file_path = self.files[idx]
#         label = self.labels[idx]
        
#         spec = preprocess_audio(file_path)
        
#         # Fallback for bad file or if preprocess_audio returns None
#         if spec is None:
#             # Return a dummy spectrogram of expected size (e.g., 64 mel bins, 94 time frames for 3s @ 16kHz)
#             return torch.zeros(1, 64, 94), torch.tensor(label).float()
            
#         return spec, torch.tensor(label).float()

# # --- 2. Training Loop ---
# def train():
#     # Setup
#     DATA_DIR = "data/audio"
#     EPOCHS = 20
#     BATCH_SIZE = 16
#     LR = 0.001
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     OUTPUT_DIR = Path("results/audio")
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
#     print(f"Using device: {DEVICE}")
#     print("Initializing First-Principles Audio CNN...")
    
#     # Data
#     dataset = SirenDataset(DATA_DIR)
    
#     # Simple split (80/20)
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
#     # Model
#     model = AudioCNN().to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.BCEWithLogitsLoss()
    
#     # Metrics tracking
#     history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
#     # Training
#     print("\nStarting Training...")
#     best_val_acc = 0.0
    
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
        
#         for specs, labels in train_loader:
#             specs, labels = specs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
#             optimizer.zero_grad()
#             outputs = model(specs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
            
#         # Validation
#         model.eval()
#         correct = 0
#         total = 0
#         val_loss = 0
#         all_preds = []
#         all_probs = []
#         all_labels = []
        
#         with torch.no_grad():
#             for specs, labels in val_loader:
#                 specs, labels = specs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
#                 outputs = model(specs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
                
#                 probs = torch.sigmoid(outputs)
#                 predicted = (probs > 0.5).float()
                
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)
                
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         val_acc = correct / total
#         print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2%}")
        
#         history['train_loss'].append(train_loss / len(train_loader))
#         history['val_loss'].append(val_loss / len(val_loader))
#         history['val_acc'].append(val_acc)
        
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), OUTPUT_DIR / "audio_cnn.pth")
            
#     print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2%}")
#     print(f"Model saved to {OUTPUT_DIR / 'audio_cnn.pth'}")
    
#     # --- Plotting ---
#     print("\nGenerating plots...")
    
#     # 1. Learning Curve
#     plt.figure(figsize=(10, 5))
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.plot(history['val_acc'], label='Validation Accuracy')
#     plt.title('Training and Validation Metrics')
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(OUTPUT_DIR / "learning_curve.png")
#     plt.close()
#     print(f"- Learning curve saved.")
    
#     # 2. Confusion Matrix
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=['Not Siren', 'Siren'], 
#                 yticklabels=['Not Siren', 'Siren'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
#     plt.close()
#     print(f"- Confusion matrix saved.")
    
#     # 3. Prediction Confidence Distribution
#     all_probs = np.array(all_probs).flatten()
#     all_labels = np.array(all_labels).flatten()
    
#     plt.figure(figsize=(10, 5))
#     sns.histplot(all_probs[all_labels==0], color='blue', label='Not Siren', kde=True, bins=20, alpha=0.5)
#     sns.histplot(all_probs[all_labels==1], color='red', label='Siren', kde=True, bins=20, alpha=0.5)
#     plt.axvline(0.5, color='black', linestyle='--', label='Threshold')
#     plt.title('Prediction Confidence Distribution')
#     plt.xlabel('Predicted Probability (Siren)')
#     plt.ylabel('Count')
#     plt.legend()
#     plt.savefig(OUTPUT_DIR / "prediction_confidence.png")
#     plt.close()
#     print(f"- Confidence plot saved.")
    
#     # 4. Feature Distribution (PCA of Embeddings)
#     print("Extracting embeddings for PCA...")
#     embeddings = []
#     embed_labels = []
    
#     model.eval()
#     with torch.no_grad():
#         # Use entire dataset for better visualization
#         full_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
#         for specs, labels in full_loader:
#             specs = specs.to(DEVICE)
#             # Get 64-dim embedding
#             emb = model(specs, return_embedding=True)
#             embeddings.extend(emb.cpu().numpy())
#             embed_labels.extend(labels.numpy())
            
#     embeddings = np.array(embeddings)
#     embed_labels = np.array(embed_labels)
    
#     if len(embeddings) > 1:
#         pca = PCA(n_components=2)
#         embeddings_2d = pca.fit_transform(embeddings)
        
#         plt.figure(figsize=(8, 6))
#         scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
#                              c=embed_labels, cmap='coolwarm', alpha=0.7, edgecolors='k')
#         plt.legend(handles=scatter.legend_elements()[0], labels=['Not Siren', 'Siren'])
#         plt.title('Learned Feature Space (PCA)')
#         plt.xlabel('PC1')
#         plt.ylabel('PC2')
#         plt.grid(True, alpha=0.3)
#         plt.savefig(OUTPUT_DIR / "feature_distribution.png")
#         plt.close()
#         print(f"- Feature distribution (PCA) saved.")
        
#     # 5. Average Spectrograms (Visual "Feature Importance")
#     # We will compute the mean spectrogram for each class from the first 100 samples to save time
#     siren_specs = []
#     not_siren_specs = []
#     count = 0
    
#     for specs, labels in full_loader:
#         specs = specs.numpy()
#         labels = labels.numpy()
#         for i in range(len(labels)):
#             if labels[i] == 1:
#                 siren_specs.append(specs[i, 0]) # remove channel dim
#             else:
#                 not_siren_specs.append(specs[i, 0])
#         count += len(labels)
#         if count > 200: break
        
#     if siren_specs and not_siren_specs:
#         avg_siren = np.mean(siren_specs, axis=0)
#         avg_not_siren = np.mean(not_siren_specs, axis=0)
        
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
#         im1 = axes[0].imshow(avg_siren, aspect='auto', origin='lower', cmap='inferno')
#         axes[0].set_title('Average Siren Spectrogram')
#         axes[0].set_xlabel('Time')
#         axes[0].set_ylabel('Mel Frequency')
#         plt.colorbar(im1, ax=axes[0])
        
#         im2 = axes[1].imshow(avg_not_siren, aspect='auto', origin='lower', cmap='inferno')
#         axes[1].set_title('Average Noise Spectrogram')
#         axes[1].set_xlabel('Time')
#         axes[1].set_ylabel('Mel Frequency')
#         plt.colorbar(im2, ax=axes[1])
        
#         plt.tight_layout()
#         plt.savefig(OUTPUT_DIR / "average_spectrograms.png")
#         plt.close()
#         print(f"- Average spectrograms saved.")

#     # Print Classification Report
#     print("\nClassification Report on Validation Set:")
#     print(classification_report(all_labels, all_preds, target_names=['Not Siren', 'Siren']))


# if __name__ == "__main__":
#     train()

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "audio_data/train"

X = []
y = []

labels = {"siren": 1, "normal": 0}

for label in labels:
    folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            audio, sr = librosa.load(path, sr=None)
            mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
            X.append(mfcc)
            y.append(labels[label])

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier()
model.fit(X, y)

print("✅ Audio training complete")