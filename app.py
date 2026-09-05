from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Modelo de IA para prendas",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).resolve().parent / "fashion_mnist_modelo.keras"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parents[1] / "Perceptron" / "fashion_mnist_model.keras"
CLASS_NAMES = [
    "Camiseta / top",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');

    :root { --ink: #14231f; --mint: #d9f5e9; --lime: #d8f36b; --paper: #f5f3ed; --line: #b9c9c1; }
    .stApp { background: var(--paper); color: var(--ink); font-family: Manrope, "Trebuchet MS", sans-serif; }
    .block-container { max-width: 760px; padding: 2rem 1rem 1rem; }
    h1, h2, h3, p, label, button, input, textarea, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { font-family: Manrope, "Trebuchet MS", sans-serif !important; letter-spacing: 0 !important; }
    h1 { font-size: clamp(2.1rem, 8vw, 4.6rem) !important; font-weight: 800 !important; line-height: 1.05 !important; max-width: 650px; color: var(--ink) !important; }
    h2, h3 { color: var(--ink) !important; line-height: 1.2 !important; }
    p, label, [data-testid="stMarkdownContainer"] { font-size: 1rem; line-height: 1.55; }
    .eyebrow { color: #527169; font-family: "DM Mono", Consolas, monospace !important; font-size: .75rem; font-weight: 500; line-height: 1.4; letter-spacing: .04em !important; text-transform: uppercase; }
    .objective { border-left: 4px solid var(--lime); background: var(--mint); padding: 1rem 1.1rem; margin: 1.5rem 0 1.8rem; border-radius: 2px; }
    .objective p { margin: 0; line-height: 1.55; }
    [data-baseweb="tab-list"] { gap: .35rem; }
    [data-baseweb="tab"] { font-family: Manrope, "Trebuchet MS", sans-serif !important; font-size: .95rem !important; font-weight: 700 !important; line-height: 1.3 !important; color: var(--ink) !important; white-space: nowrap; }
    [data-testid="stButton"] button { font-family: Manrope, "Trebuchet MS", sans-serif !important; font-size: .95rem !important; font-weight: 700 !important; line-height: 1.25 !important; min-height: 2.75rem; }
    [data-testid="stFileUploader"] label, [data-testid="stCameraInput"] label { line-height: 1.4 !important; }
    [data-testid="stAlert"] p { font-size: .95rem !important; line-height: 1.45 !important; }
    div[data-testid="stCanvas"] { display: flex; justify-content: center; }
    div[data-testid="stCanvas"] > div { border: 2px solid var(--ink); box-shadow: 8px 8px 0 var(--lime); }
    .canvas-note { color: #527169; font-family: "DM Mono", Consolas, monospace; font-size: .78rem; line-height: 1.4; text-align: center; margin: 1rem 0; }
    .result { border: 1px solid var(--line); padding: 1rem 1.1rem; background: #fffefa; border-radius: 2px; }
    .result-name { font-family: Manrope, "Trebuchet MS", sans-serif; font-size: 1.65rem; font-weight: 800; line-height: 1.2; margin: 0; }
    .result-score { font-family: "DM Mono", Consolas, monospace; font-size: .85rem; font-weight: 500; line-height: 1.4; color: #527169; margin: .35rem 0 0; }
    footer { text-align: center; color: #527169; margin-top: 3rem; font-family: "DM Mono", Consolas, monospace; font-size: .75rem; line-height: 1.5; }
    @media (max-width: 600px) {
        .block-container { width: 100%; padding: 1.1rem .75rem .5rem; }
        h1 { font-size: 2.35rem !important; line-height: 1.08 !important; }
        h2 { font-size: 1.45rem !important; }
        h3 { font-size: 1.2rem !important; }
        p, label, [data-testid="stMarkdownContainer"] { font-size: .95rem; }
        [data-baseweb="tab-list"] { width: 100%; gap: 0; }
        [data-baseweb="tab"] { flex: 1 1 0; padding-left: .25rem !important; padding-right: .25rem !important; font-size: .8rem !important; }
        [data-testid="stHorizontalBlock"] { gap: .65rem; }
        div[data-testid="stCanvas"] > div { box-shadow: 5px 5px 0 var(--lime); }
        .result-name { font-size: 1.35rem; }
        .objective { margin: 1.1rem 0 1.4rem; padding: .85rem .9rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Clasificación de imágenes · Fashion-MNIST</div>', unsafe_allow_html=True)
st.title("Clasificador de prendas con IA")
st.markdown(
    '<div class="objective"><p><strong>Objetivo:</strong> identifica una prenda usando un dibujo, una fotografía tomada con la cámara o una imagen cargada desde tu dispositivo.</p></div>',
    unsafe_allow_html=True,
)

draw_tab, camera_tab, image_tab = st.tabs(["Dibujo", "Cámara", "Imagen"])

source_image = None
source_name = ""

with draw_tab:
    st.subheader("Dibuja una prenda")
    st.write("Usa el lápiz blanco sobre el fondo negro. Procura dibujar la silueta centrada y grande.")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=16,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="fashion_canvas",
        display_toolbar=True,
    )
    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
        source_image = canvas_result.image_data
        source_name = "Dibujo"
    st.markdown('<div class="canvas-note">El dibujo se transforma a 28 × 28 píxeles para el modelo.</div>', unsafe_allow_html=True)

with camera_tab:
    st.subheader("Toma una fotografía")
    st.write("Centra una sola prenda, con buena iluminación y un fondo sencillo.")
    camera_image = st.camera_input("Abrir cámara", label_visibility="collapsed")
    if camera_image is not None:
        source_image = Image.open(camera_image).convert("RGBA")
        source_name = "Cámara"

with image_tab:
    st.subheader("Selecciona una imagen")
    st.write("Carga una imagen clara donde se vea una sola prenda.")
    uploaded_image = st.file_uploader(
        "Elegir archivo",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_image is not None:
        source_image = Image.open(uploaded_image).convert("RGBA")
        source_name = "Imagen cargada"

_, action_column = st.columns([2, 1])
with action_column:
    predict = st.button("Analizar imagen", type="primary", use_container_width=True)


@st.cache_resource
def load_model(model_path: str):
    import json
    import tempfile
    import zipfile

    import tensorflow as tf

    def remove_legacy_metadata(value):
        if isinstance(value, dict):
            value.pop("quantization_config", None)
            for child in value.values():
                remove_legacy_metadata(child)
        elif isinstance(value, list):
            for child in value:
                remove_legacy_metadata(child)

    with zipfile.ZipFile(model_path) as source:
        temporary_model = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        with zipfile.ZipFile(temporary_model.name, "w") as target:
            for item in source.infolist():
                content = source.read(item.filename)
                if item.filename == "config.json":
                    config = json.loads(content.decode("utf-8"))
                    remove_legacy_metadata(config)
                    content = json.dumps(config).encode("utf-8")
                target.writestr(item, content)
        temporary_model.close()

    return tf.keras.models.load_model(
        temporary_model.name,
        compile=False,
    )


def prepare_image(source_data) -> np.ndarray:
    if isinstance(source_data, Image.Image):
        grayscale = source_data.convert("L")
    else:
        grayscale = Image.fromarray(source_data.astype("uint8"), mode="RGBA").convert("L")

    grayscale_array = np.asarray(grayscale, dtype="uint8")
    border_pixels = np.concatenate(
        [grayscale_array[0, :], grayscale_array[-1, :], grayscale_array[:, 0], grayscale_array[:, -1]]
    )
    if border_pixels.mean() > 127:
        grayscale_array = 255 - grayscale_array
        grayscale = Image.fromarray(grayscale_array, mode="L")

    foreground = grayscale_array > 8
    if foreground.any():
        rows, columns = np.where(foreground)
        left, right = columns.min(), columns.max() + 1
        top, bottom = rows.min(), rows.max() + 1
        cropped = grayscale.crop((left, top, right, bottom))
        side = max(cropped.width, cropped.height)
        padded = Image.new("L", (side, side), 0)
        offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
        padded.paste(cropped, offset)
        resized = padded.resize((24, 24), Image.Resampling.LANCZOS)
        centered = Image.new("L", (28, 28), 0)
        centered.paste(resized, (2, 2))
    else:
        centered = Image.new("L", (28, 28), 0)

    image_28_uint8 = np.asarray(centered, dtype="uint8")
    image_28_uint8 = np.clip(image_28_uint8, 0, 255)
    image_28 = image_28_uint8.astype("float32") / 255.0
    return image_28_uint8, image_28


if predict:
    if source_image is None:
        st.warning("Dibuja, captura o carga una imagen antes de clasificarla.")
    else:
        model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
        if not model_path.exists():
            st.error(
                "No se encontró un modelo compatible. Coloca `fashion_mnist_modelo.keras` "
                "junto a `app.py`."
            )
        else:
            try:
                model = load_model(str(model_path))
                image_28_uint8, image_28 = prepare_image(source_image)
                input_shape = model.input_shape
                if len(input_shape) == 3:
                    model_input = image_28[np.newaxis, ...]
                elif len(input_shape) == 4:
                    model_input = image_28[np.newaxis, ..., np.newaxis]
                else:
                    model_input = image_28.reshape(1, -1)
                probabilities = np.asarray(model.predict(model_input, verbose=0))[0]
                predicted_index = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_index]) * 100

                st.divider()
                st.subheader("Resultado del análisis")
                result_column, preview_column = st.columns([1.4, 1])
                with result_column:
                    st.markdown(
                        f'<div class="result"><p class="result-name">{CLASS_NAMES[predicted_index]}</p><p class="result-score">fuente · {source_name} · confianza · {confidence:.1f}%</p></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(confidence / 100, 0.0), 1.0))
                with preview_column:
                    st.image(
                        image_28_uint8,
                        caption=f"Imagen en grises · 28 × 28 · máximo {image_28_uint8.max()}/255",
                        clamp=True,
                        use_container_width=True,
                    )
            except Exception as error:
                st.error(f"No fue posible realizar la predicción: {error}")

st.markdown("<footer>Autor: Javier Luna<br>UNAB 2026</footer>", unsafe_allow_html=True)
