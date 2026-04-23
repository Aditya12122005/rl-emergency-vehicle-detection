class Config:
    # Vision (YOLO)
    MODEL = "yolov8n.pt"
    DATA = "corrected_dataset/data.yaml"
    EPOCHS = 5
    IMG_SIZE = 320
    BATCH = 2

    # Audio
    AUDIO_DATA = "audio_data"