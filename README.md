# 🎓 Feynman Student AI — Interactive Gap-Finder Tutor

Aplikasi web antarmuka **Gradio** untuk kompetisi **CAICC 2026 Track 2: Prompt to App**.

Aplikasi ini membantu mahasiswa menemukan kesalahpahaman konsep (*conceptual gaps*)
menggunakan prinsip **Feynman Technique terbalik**. Pengguna berperan sebagai
**Pengajar**, sedangkan AI berperan sebagai **"Murid Kritis"** yang belajar dari
pengguna.

---

## ✨ Konsep Utama

> "Jika kamu tidak bisa menjelaskan sesuatu dengan sederhana, kamu belum
> memahaminya." — Richard Feynman

Alih-alih AI yang menjelaskan ke pengguna, di sini alurnya **dibalik**:
pengguna menjelaskan materi dengan bahasanya sendiri, lalu AI (sebagai Murid
Kritis) mengajukan pertanyaan tajam untuk memancing pengguna menemukan sendiri
celah pemahamannya.

---

## 🧭 Alur Penggunaan

| Panel | Fungsi |
|-------|--------|
| **1. Input Materi Referensi** | Tempel catatan kuliah sebagai satu-satunya dasar penilaian AI. |
| **2. Sesi Mengajar Interaktif** | Chatbot — pengguna menjelaskan, AI bertanya kritis. |
| **3. Rangkuman Pemahaman** | Evaluasi akhir terstruktur setelah sesi selesai. |

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python 3.9 atau lebih baru.
- API token dari [ModelScope Inference API](https://api-inference.modelscope.ai/).

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Set environment variable

Pilih salah satu (di Windows gunakan `set`, di Linux/macOS gunakan `export`):

```bash
# Linux / macOS
export MODELSCOPE_TOKEN="token-anda-di-sini"

# Windows (Command Prompt)
set MODELSCOPE_TOKEN=token-anda-di-sini

# Windows (PowerShell)
$env:MODELSCOPE_TOKEN = "token-anda-di-sini"
```

> Token juga bisa dibaca dari `ANTHROPIC_AUTH_TOKEN` sebagai cadangan.

### 4. Jalankan aplikasi

```bash
python app.py
```

Buka alamat yang ditampilkan di terminal (biasanya `http://127.0.0.1:7860`).

---

## 🧠 Aturan Perilaku AI (System Prompt Internal)

1. **Persona Murid Kritis** — santai, ingin tahu, namun kritis. Mengajukan
   pertanyaan tingkat tinggi saat penjelasan melompat, tidak konsisten, atau
   bertentangan dengan referensi.
2. **Batasan Referensi Ketat** — AI hanya menilai berdasarkan materi referensi.
   Jika pengetahuan pendukung tidak ada di referensi, AI wajib jujur dan meminta
   pengguna melengkapi materi.
3. **Format Rangkuman Akhir** — tiga bagian: `[Poin Positif]`, `[Miskonsepsi/Gap]`,
   dan `[Saran Rekomendasi]`.

---

## 🧩 Arsitektur Kode

- **`app.py`** — antarmuka Gradio (`gr.Blocks`) + logika API.
- **`requirements.txt`** — daftar dependensi.
- **`README.md`** — dokumentasi ini.

Model dihubungkan melalui **OpenAI-compatible API** dari ModelScope
(`https://api-inference.modelscope.ai/v1`) menggunakan SDK `openai`, dengan
daftar model fallback agar aplikasi tangguh terhadap kegagalan.

---

## 🛡️ Penanganan Error

Setiap pemanggilan API dibungkus `try-except` sehingga antarmuka Gradio tidak
pernah *crash* akibat kesalahan koneksi. Jika seluruh model fallback gagal,
pesan error yang ramah akan ditampilkan langsung di dalam antarmuka.