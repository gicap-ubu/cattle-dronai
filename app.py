import os
import time
import cv2
import tempfile
import pathlib
import streamlit as st
import torch
from ultralytics import YOLO


# Arreglo para modelos YOLOv5 guardados en Linux y ejecutados en Windows
pathlib.PosixPath = pathlib.WindowsPath


st.set_page_config(
    page_title="Detección de vacas con YOLO",
    layout="wide"
)

st.title("Detección de vacas en vídeo de dron con YOLO")

OUTPUTS_DIR = "outputs"
YOLOV5_REPO = "yolov5"
YOLOV5_CATTLE_MODEL = "models/yolo5-farm-cattle.pt"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False


st.subheader("Seleccionar vídeo")

uploaded_file = st.file_uploader(
    "Selecciona un vídeo desde tu disco",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is None:
    st.info("Selecciona un vídeo para comenzar.")
    st.stop()


temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
temp_video.write(uploaded_file.read())
temp_video.close()

video_path = temp_video.name
video_name = uploaded_file.name


st.subheader("Configuración YOLO")

model_name = st.selectbox(
    "Modelo YOLO",
    [
        "yolo11n.pt",
        "yolo11s.pt",
        "models/mmla.pt",
        "YOLOv5 farm cattle"
    ],
    index=3
)

confidence = st.slider(
    "Confianza mínima",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05
)


@st.cache_resource
def load_model(name):

    if name == "YOLOv5 farm cattle":

        if not os.path.isdir(YOLOV5_REPO):
            raise FileNotFoundError(
                "No existe la carpeta 'yolov5'. "
                "Ejecuta: git clone https://github.com/ultralytics/yolov5"
            )

        if not os.path.isfile(YOLOV5_CATTLE_MODEL):
            raise FileNotFoundError(
                f"No existe el modelo: {YOLOV5_CATTLE_MODEL}"
            )

        model = torch.hub.load(
            YOLOV5_REPO,
            "custom",
            path=YOLOV5_CATTLE_MODEL,
            source="local",
            force_reload=True
        )

        model.to("cpu")

        return {
            "type": "yolov5",
            "model": model
        }

    else:

        if name.startswith("models/") and not os.path.isfile(name):
            raise FileNotFoundError(
                f"No existe el modelo: {name}"
            )

        model = YOLO(name)
        model.to("cpu")

        return {
            "type": "ultralytics",
            "model": model
        }


with st.spinner("Cargando modelo YOLO..."):
    model_data = load_model(model_name)

st.success("Modelo cargado correctamente")


# =========================================================
# VISUALIZADORES
# =========================================================

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Vídeo original")
    original_placeholder = st.empty()
    original_placeholder.video(video_path)

with right_col:
    st.subheader("Vídeo procesado / detección")
    processed_placeholder = st.empty()


# =========================================================
# BOTONES
# =========================================================

col1, col2 = st.columns(2)

with col1:
    process_button = st.button("Procesar vídeo")

with col2:
    stop_button = st.button("Parar análisis")

if stop_button:
    st.session_state.stop_processing = True


# =========================================================
# PROCESAMIENTO
# =========================================================

if process_button:

    st.session_state.stop_processing = False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("No se pudo abrir el vídeo.")
        st.stop()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25

    output_path = os.path.join(
        OUTPUTS_DIR,
        f"processed_{video_name}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    detections_metric = metric_col1.empty()
    fps_metric = metric_col2.empty()
    frame_metric = metric_col3.empty()

    progress_bar = st.progress(0)

    frame_idx = 0
    max_objects_detected = 0
    start_time = time.time()

    model_type = model_data["type"]
    model = model_data["model"]

    while True:

        if st.session_state.stop_processing:
            st.warning("Análisis detenido por el usuario.")
            break

        ret, frame = cap.read()

        if not ret:
            break

        # -------------------------------------------------
        # MOSTRAR FRAME ORIGINAL SINCRONIZADO
        # -------------------------------------------------

        original_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        original_placeholder.image(
            original_rgb,
            channels="RGB",
            caption=f"Original - Frame {frame_idx}"
        )

        # -------------------------------------------------
        # INFERENCIA
        # -------------------------------------------------

        if model_type == "ultralytics":

            results = model.predict(
                frame,
                conf=confidence,
                device="cpu",
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()

            if result.boxes is not None:
                num_objects = len(result.boxes)
            else:
                num_objects = 0

        else:

            model.conf = confidence

            frame_rgb_input = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = model(frame_rgb_input)

            detections = results.xyxy[0]
            num_objects = len(detections)

            annotated_rgb = results.render()[0]

            annotated_frame = cv2.cvtColor(
                annotated_rgb,
                cv2.COLOR_RGB2BGR
            )

        # -------------------------------------------------
        # MOSTRAR FRAME PROCESADO SINCRONIZADO
        # -------------------------------------------------

        processed_rgb = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        processed_placeholder.image(
            processed_rgb,
            channels="RGB",
            caption=f"Procesado - Frame {frame_idx}"
        )

        # -------------------------------------------------
        # GUARDAR FRAME
        # -------------------------------------------------

        writer.write(annotated_frame)

        max_objects_detected = max(
            max_objects_detected,
            num_objects
        )

        elapsed_time = time.time() - start_time

        current_fps = (
            frame_idx / elapsed_time
            if elapsed_time > 0
            else 0
        )

        detections_metric.metric(
            "Vacas detectadas",
            num_objects
        )

        fps_metric.metric(
            "FPS procesado",
            f"{current_fps:.2f}"
        )

        frame_metric.metric(
            "Frame",
            frame_idx
        )

        frame_idx += 1

        if total_frames > 0:
            progress_bar.progress(
                min(frame_idx / total_frames, 1.0)
            )

    cap.release()
    writer.release()

    st.success("Procesamiento finalizado")

    st.metric(
        "Máximo número de vacas detectadas",
        max_objects_detected
    )

    st.subheader("Comparativa final")

    final_left, final_right = st.columns(2)

    with final_left:
        st.markdown("### Original")
        st.video(video_path)

    with final_right:
        st.markdown("### Procesado")
        st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            "Descargar vídeo procesado",
            f,
            file_name=f"processed_{video_name}",
            mime="video/mp4"
        )