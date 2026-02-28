from ultralytics import YOLO

if __name__ == '__main__':
    # Tu ruta del modelo (asegúrate de que siga siendo la correcta)
    ruta_modelo = r"runs\detect\models\train_run2\weights\best.pt"
    modelo = YOLO(ruta_modelo)

    # Cambiamos show=True por save=True
    resultados = modelo.predict(
        source='auto_prueba.jpeg', 
        conf=0.5,  
        save=True  # <--- Esta es la clave. Ahora guardará la foto procesada.
    )
    
    print("¡Proceso terminado! Revisa la carpeta runs/detect/predict/")