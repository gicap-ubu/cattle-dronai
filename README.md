# cattle-dronai

Detección de ganado vacuno mediante Inteligencia Artificial en vídeos capturados desde dron.

Este proyecto implementa un prototipo de aplicación desarrollado en Python que permite cargar vídeos aéreos obtenidos mediante UAV (Unmanned Aerial Vehicle), visualizar el vídeo original y aplicar diferentes modelos YOLO para detectar objetos y ganado sobre cada frame del vídeo.

La aplicación ha sido desarrollada utilizando:

- Python 3.11
- Streamlit
- OpenCV
- PyTorch
- Ultralytics YOLO
- YOLOv5
- Conda

---

# Características

- Selección de vídeos desde el equipo local.
- Visualización simultánea del vídeo original y el vídeo procesado.
- Procesamiento frame a frame.
- Detección de objetos mediante distintos modelos YOLO.
- Visualización en tiempo real de detecciones.
- Descarga del vídeo procesado.
- Métricas de procesamiento:
  - Número de detecciones.
  - FPS de procesamiento.
  - Frame actual.
  - Máximo número de detecciones encontradas.

---

# Estructura del proyecto

```text
cattle-dronai/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── videos/
│   └── .gitkeep
│
└── yolov5/
```

---

# Requisitos previos

Antes de comenzar es necesario disponer de:

- Git
- Conda (Anaconda o Miniconda)
- Python 3.11

---

# Clonar el repositorio

```bash
git clone https://github.com/gicap-ubu/cattle-dronai.git
cd cattle-dronai
```

---

# Crear entorno Conda

```bash
conda create -n ganado-yolo python=3.11 -y
conda activate ganado-yolo
```

---

# Instalar PyTorch para CPU

El proyecto está preparado para funcionar en equipos sin GPU NVIDIA.

Instalar PyTorch para CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Comprobar instalación:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

Resultado esperado:

```text
False
```

---

# Instalar dependencias

```bash
pip install -r requirements.txt
```

Si fuese necesario:

```bash
pip install ultralytics
pip install opencv-python
pip install streamlit
pip install pillow
pip install numpy
pip install pandas
```

---

# Instalar YOLOv5

Algunos modelos utilizados en este proyecto están basados en YOLOv5 clásico.

Desde la raíz del proyecto:

```bash
git clone https://github.com/ultralytics/yolov5
```

Instalar dependencias adicionales:

```bash
pip install -r yolov5/requirements.txt
```

---

# Modelos soportados

Actualmente la aplicación permite utilizar:

| Modelo | Descripción |
|----------|----------|
| yolo11n.pt | Modelo general ligero |
| yolo11s.pt | Modelo general más preciso |
| mmla.pt | Modelo especializado en fauna aérea |
| yolov5-farm-cattle.pt | Modelo especializado en ganado vacuno |

Los modelos personalizados deben almacenarse dentro de:

```text
models/
```

Ejemplo:

```text
models/
├── mmla.pt
├── yolov5-farm-cattle.pt
```

---

# Ejecutar la aplicación

Con el entorno Conda activado:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador.

Normalmente estará disponible en:

```text
http://localhost:8501
```

---

# Uso de la aplicación

## 1. Seleccionar vídeo

Seleccionar un archivo de vídeo desde el equipo local.

Formatos soportados:

- mp4
- avi
- mov
- mkv

## 2. Seleccionar modelo

Elegir el modelo YOLO deseado.

## 3. Configurar confianza mínima

Ajustar el umbral de confianza para las detecciones.

Valores recomendados:

```text
0.25 - 0.40
```

## 4. Procesar vídeo

Pulsar:

```text
Procesar vídeo
```

Durante el procesamiento se muestran:

- Vídeo original.
- Vídeo procesado.
- Número de detecciones.
- FPS de procesamiento.
- Frame actual.

## 5. Descargar resultado

Al finalizar se podrá descargar el vídeo generado.

---

# Carpetas del proyecto

## videos/

Carpeta opcional para almacenar vídeos de prueba.

## outputs/

Carpeta donde se almacenan los vídeos procesados.

## models/

Carpeta donde se almacenan los modelos personalizados.
