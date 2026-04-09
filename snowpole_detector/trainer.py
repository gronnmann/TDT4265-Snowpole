from ultralytics import YOLO

model = YOLO("yolo11s.pt") 

model.train(
    data="dataset-augmented/data.yaml",
    epochs=1000,
    imgsz=1280,
    batch=24,
    device=0,
    rect=True,
    mosaic=1.0,
    close_mosaic=10 ,
    project="TDT4265-Snowpole", # wandb?
    save_period=50,
)