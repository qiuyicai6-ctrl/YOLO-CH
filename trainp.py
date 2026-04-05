# train_pcb.py
from ultralytics import YOLO
from ultralytics.utils.pcbsunshi import PcbSmallObjLoss  # ① 导入

def train():
    # 1. 加载官方模型（带预训练权重）
    model = YOLO(r"C:\Users\86156\Desktop\程序汇总\YOLOv8-model-improvement-P2cecasun\ultralytics\cfg\models\v8\yolov8m-P2.yaml")   # 会自动下载
    model.model.loss = PcbSmallObjLoss(model.model, small_weight=3.0, small_thresh=32)
    # 2. 训练参数（全部用官方默认损失）
    model.train(
        data=r"C:\Users\86156\Desktop\程序汇总\YOLOv8-model-improvement-P2cecasun\ultralytics\datasets\pcb.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device='cpu',          # 有 GPU 就写 0 或 [0,1,2,3]
        workers=0,             # Windows 上先设 0，Linux 可开大
        name='pcb_official',
        exist_ok=True,         # 重复运行覆盖
        pretrained=True,
        optimizer='auto',      # 默认 AdamW
        lr0=0.01,
        lrf=0.01,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        amp=False              # CPU 训练时关掉，GPU 可开 True
    )

if __name__ == '__main__':
    train()