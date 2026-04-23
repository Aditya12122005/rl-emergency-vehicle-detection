# from ultralytics import YOLO
from ultralytics import YOLO
from src.config import Config

def train():
    model = YOLO(Config.MODEL)

    model.train(
        data=Config.DATA,
        epochs=Config.EPOCHS,
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH
    )

if __name__ == "__main__":
    train()