# Klasifikasi Pakaian dengan CNN

Aplikasi web sederhana (Streamlit) yang mengenali satu barang fesyen dari foto. Modelnya adalah CNN yang dilatih pada dataset **Fashion-MNIST**, dengan 10 kategori:

T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Cara kerja

1. Pengguna mengunggah gambar atau memotret barang dengan kamera.
2. Gambar diproses agar mirip data Fashion-MNIST: grayscale, latar dibuat gelap, barang dipotong dan dipusatkan, lalu diubah ke ukuran 28×28.
3. Model memprediksi kategori. Aplikasi menampilkan hasil utama, tingkat keyakinan, dan 3 kemungkinan teratas.

Preprocessing di `app.py` harus sama persis dengan yang dipakai saat training di notebook.

## Isi folder

| File | Fungsi |
|------|--------|
| `app.py` | Aplikasi Streamlit |
| `requirements.txt` | Daftar dependensi |
| `fashion_cnn.keras` | Model hasil training (**harus kamu siapkan sendiri**) |

## Cara menjalankan

1. Taruh `fashion_cnn.keras` di folder ini. Buat dengan `model.save("fashion_cnn.keras")` setelah training.
2. Buat dan aktifkan virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
5. Buka `http://localhost:8501` di browser. Tekan `Ctrl+C` di terminal untuk berhenti.

## Tips agar prediksi akurat

- Foto satu barang saja.
- Gunakan latar polos (lantai, kasur, atau dinding).
- Ambil dari atas atau depan, dengan barang terlihat utuh.

## Catatan

- Di Mac, gunakan paket `tensorflow` (bukan `tensorflow-cpu`, yang tidak punya versi untuk macOS).
- Jika muncul error `No such file or directory: fashion_cnn.keras`, berarti file model belum ada di folder ini.
