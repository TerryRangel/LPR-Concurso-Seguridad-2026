from ultralytics import YOLO

# Todo el código de ejecución debe ir dentro de este bloque en Windows
if __name__ == '__main__':
    
    # 1. Load the fastest pre-trained model (YOLOv8 Nano)
    model = YOLO('yolov8n.pt') 

    # 2. Train the model using the dataset configuration
    results = model.train(
        data='dataset/data.yaml', 
        epochs=200, 
        imgsz=640,
        project='models',
        name='train_run',
        device=0  # Aquí usaremos la GPU si se desea usar la CPU, borra la línea o pon device='cpu'
        #en caso de usar gpu instalar torch con soporte para cuda, por ejemplo: pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    )