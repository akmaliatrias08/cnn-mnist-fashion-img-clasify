import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Klasifikasi Pakaian", page_icon="👕", layout="centered")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
LABEL_ID = {"T-shirt/top": "Kaos/atasan", "Trouser": "Celana panjang", "Pullover": "Sweter",
            "Dress": "Gaun", "Coat": "Mantel/jaket", "Sandal": "Sandal", "Shirt": "Kemeja",
            "Sneaker": "Sepatu sneaker", "Bag": "Tas", "Ankle boot": "Sepatu bot"}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_cnn.keras")


# Harus sama persis dengan preprocessing di notebook (bagian 9)
def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("L")
    img.thumbnail((256, 256))
    arr = np.asarray(img, dtype=np.float32)

    def border(a):
        return np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if border(arr).mean() > 127:          # latar terang -> balik warna
        arr = 255.0 - arr
    arr = np.clip(arr - np.median(border(arr)), 0, 255)   # hilangkan sisa warna latar

    mask = arr > 0.15 * arr.max() if arr.max() > 0 else arr > 0
    if mask.any():
        rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = arr.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr

    out = np.asarray(Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS),
                     dtype=np.float32)
    if out.max() > 0:
        out = out / out.max()
    return out.reshape(1, 28, 28, 1)


model = load_model()

st.title("👕 Klasifikasi Pakaian dengan CNN")
st.write("Unggah atau potret satu barang fesyen. Model mengenali 10 kategori "
         "dari dataset Fashion-MNIST.")

with st.expander("Tips agar prediksi akurat"):
    st.write("Foto satu barang saja, dengan latar polos (lantai, kasur, atau dinding), "
             "diambil dari atas atau depan, dan barang terlihat utuh.")

tab_upload, tab_camera = st.tabs(["Unggah gambar", "Ambil foto"])
with tab_upload:
    file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    shot = st.camera_input("Potret barang")

source = file or shot
if source is None:
    st.info("Belum ada gambar. Unggah file atau ambil foto untuk melihat prediksi.")
    st.stop()

image = Image.open(source)
x = preprocess(image)
prob = model.predict(x, verbose=0)[0]
order = prob.argsort()[::-1]
best = order[0]

col1, col2 = st.columns(2)
col1.image(ImageOps.exif_transpose(image), caption="Gambar asli", width=250)
col2.image(x.squeeze(), caption="Yang dilihat model (28×28)", width=250, clamp=True)

st.subheader(f"Prediksi: {CLASS_NAMES[best]} ({LABEL_ID[CLASS_NAMES[best]]})")
st.metric("Tingkat keyakinan", f"{prob[best]:.1%}")
if prob[best] < 0.5:
    st.warning("Keyakinan model rendah. Coba foto ulang dengan latar lebih polos.")

st.write("**Tiga kemungkinan teratas**")
for i in order[:3]:
    st.progress(float(prob[i]), text=f"{CLASS_NAMES[i]} — {prob[i]:.1%}")
