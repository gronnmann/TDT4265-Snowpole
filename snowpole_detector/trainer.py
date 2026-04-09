from ultralytics import YOLO

model = YOLO("yolo11s.pt") 

model.train(
    data="dataset-augmented/data.yaml",
    epochs=100,
    imgsz=1280,
    batch=16,
    device=0,
    rect=True,
    mosaic=1.0,
    close_mosaic=10 ,
    project="TDT4265-Snowpole", # wandb?
)