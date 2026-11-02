"""
Feynman Student AI — Interactive Gap-Finder Tutor
Kompetisi CAICC 2026 Track 2: Prompt to App

Aplikasi ini membantu mahasiswa menemukan miskonsepsi konsep (conceptual gaps)
menggunakan prinsip Feynman Technique terbalik. Pengguna berperan sebagai
Pengajar, sedangkan AI berperan sebagai "Murid Kritis" yang belajar dari pengguna.

Struktur:
- Panel 1 : Input materi referensi (catatan kuliah)
- Panel 2 : Sesi mengajar interaktif (chatbot)
- Panel 3 : Rangkuman pemahaman (feedback dashboard)
"""

import os

import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

BASE_URL = "https://api-inference.modelscope.ai/v1"

# Daftar model resmi kompetisi CAICC 2026 (Track 2) beserta urutan fallback.
# Jika model pertama gagal, aplikasi akan mencoba model berikutnya secara
# berurutan.
MODEL_FALLBACKS = [
    "zai-org/GLM-5.2",               # Model utama
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    "Qwen/Qwen3.8-72B-Instruct",     # Fallback berperforma tinggi
    "Qwen/Qwen3.8-27B",
]


def get_api_token() -> str:
    """Ambil token API dari environment variable secara aman."""
    token = os.getenv("MODELSCOPE_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if not token:
        raise RuntimeError(
            "API token tidak ditemukan. Set environment variable "
            "`MODELSCOPE_TOKEN` (atau `ANTHROPIC_AUTH_TOKEN`) sebelum menjalankan aplikasi."
        )
    return token


def build_client(token: str = None) -> OpenAI:
    """Bangun klien OpenAI-compatible yang mengarah ke ModelScope."""
    token = token or get_api_token()
    return OpenAI(api_key=token, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# System prompt (persona AI)
# ---------------------------------------------------------------------------

def build_system_prompt(reference: str) -> str:
    """Susun system prompt yang membimbing perilaku AI sebagai Murid Kritis."""
    return f"""Kamu adalah "Murid Kritis", seorang mahasiswa yang antusias, santai, \
ingin tahu, namun kritis secara intelektual. Lawan bicaramu adalah seorang \
pengajar (mahasiswa lain) yang berusaha menjelaskan materi dengan bahasanya sendiri.

PERAN UTAMA:
Kamu belajar DARI pengguna. Tugasmu bukan menggurui, melainkan memancing \
pengguna menemukan sendiri celah pemahamannya (conceptual gaps) lewat pertanyaan \
yang tajam namun bersahabat.

ATURAN PERILAKU:
1. PERSONA MURID KRITIS
   - Bersikap santai dan terkesan benar-benar ingin paham.
   - Ajukan pertanyaan tingkat tinggi saat penjelasan pengguna terasa melompat, \
tidak konsisten, atau bertentangan dengan materi referensi. Contoh:
     "Kenapa hal itu bisa terjadi?"
     "Bukankah jika X terjadi, maka Y seharusnya Z?"
     "Bisa dijelaskan kenapa langkah itu terasa seperti melompat?"
   - Jangan langsung memberi jawaban. Arahkan pengguna untuk memikirkan ulang.

2. BATASAN REFERENSI KETAT
   - Kamu HANYA boleh menilai dan mempertanyakan penjelasan pengguna berdasarkan \
teks materi referensi yang diberikan (di bawah).
   - JANGAN menggunakan pengetahuan di luar materi referensi untuk menilai \
kebenaran penjelasan pengguna.
   - Jika penjelasan pengguna memerlukan pengetahuan pendukung yang TIDAK ADA \
di dalam materi referensi, katakan dengan jujur:
     "Materi referensi yang diberikan belum mencakup bagian ini untuk saya \
verifikasi, bisa tambahkan informasi terkait di materi referensi?"

3. FORMAT RANGKUMAN AKHIR
   Ketika pengguna meminta untuk mengakhiri sesi (misalnya menekan tombol \
"Selesaikan Sesi"), berikan rangkuman terstruktur dengan tepat 3 bagian:
   - [Poin Positif]: Konsep yang berhasil dijelaskan pengguna dengan sangat baik \
dan akurat.
   - [Miskonsepsi/Gap]: Bagian di mana penjelasan pengguna kurang tepat, melompat, \
atau membingungkan.
   - [Saran Rekomendasi]: Bagian spesifik dari materi referensi yang perlu dibaca \
ulang oleh pengguna.

MATERI REFERENSI (satu-satunya dasar penilaianmu):
---------------- BEGIN REFERENCE ----------------
{reference}
---------------- END REFERENCE ----------------
"""


# ---------------------------------------------------------------------------
# Logika pemanggilan API (dengan error handling)
# ---------------------------------------------------------------------------

def call_model(reference: str, history: list, user_message: str = None) -> str:
    """
    Panggil model melalui ModelScope Inference API.

    - Jika `user_message` diberikan, itu dipakai sebagai pesan pengguna terbaru.
    - Jika `user_message` adalah None, gunakan perintah internal untuk meminta
      rangkuman akhir sesi.
    """
    client = build_client()

    messages = [{"role": "system", "content": build_system_prompt(reference)}]

    # Normalisasi riwayat percakapan dari format Gradio (type="messages").
    for entry in history or []:
        role = entry.get("role")
        content = entry.get("content")
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

    if user_message is not None:
        messages.append({"role": "user", "content": user_message})
    else:
        # Perintah internal untuk menghasilkan rangkuman evaluasi akhir.
        messages.append(
            {
                "role": "user",
                "content": (
                    "Sesi belajar telah selesai. Tolong berikan rangkuman evaluasi "
                    "akhir yang terstruktur dengan tepat 3 bagian: [Poin Positif], "
                    "[Miskonsepsi/Gap], dan [Saran Rekomendasi]."
                ),
            }
        )

    # Coba setiap model pada daftar fallback sampai ada yang berhasil.
    last_error = None
    for model_name in MODEL_FALLBACKS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as exc:  # noqa: BLE001 - jaga agar UI tidak pernah crash
            last_error = exc
            continue

    # Seluruh model gagal.
    raise RuntimeError(
        "Gagal memanggil API ModelScope. "
        f"Error terakhir: {last_error}"
    )


# ---------------------------------------------------------------------------
# Callback Gradio
# ---------------------------------------------------------------------------

def start_session(reference: str) -> tuple:
    """
    Menyimpan materi referensi dan menginisialisasi sesi percakapan baru.
    Mengembalikan: (state string, pesan pembuka bot, riwayat chat kosong).
    """
    reference = (reference or "").strip()
    if not reference:
        warning = "⚠️  **Materi referensi masih kosong.**\n\n" \
                  "Silakan tempel catatan kuliah / teks materi pada Panel 1 terlebih dahulu, " \
                  "lalu klik **Mulai Sesi Belajar**."
        return (
            "",
            [{"role": "assistant", "content": warning}],
            [],
        )

    greeting = (
        "👋 **Halo, saya Murid Kritis!** Siap belajar dari kamu.\n\n"
        "Ceritakan / jelaskan materi referensimu dengan bahasamu sendiri. "
        "Saya akan mendengarkan dengan saksama sambil sesekali bertanya "
        "kalau ada yang kurang jelas atau terasa melompat. 😄\n\n"
        "Silakan mulai — apa yang ingin kamu jelaskan pertama?"
    )

    # state berupa string referensi + flag sesi aktif.
    state = f"ACTIVE\n{reference}"
    return state, [{"role": "assistant", "content": greeting}], []


def send_explanation(reference_state: str, history: list, user_input: str) -> tuple:
    """
    Proses penjelasan pengguna, panggil API, dan kembalikan riwayat yang diperbarui.
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return history, history, ""

    # Periksa apakah sesi aktif beserta referensinya tersimpan.
    if not reference_state or not reference_state.startswith("ACTIVE\n"):
        history = list(history or [])
        history.append(
            {"role": "assistant",
             "content": "⚠️  Sesi belum dimulai. Silakan klik **Mulai Sesi Belajar** di Panel 1 dulu."}
        )
        return history, history, ""

    reference = reference_state.split("\n", 1)[1]

    history = list(history or [])
    history.append({"role": "user", "content": user_input})

    try:
        reply = call_model(reference, history, user_message=user_input)
    except Exception as exc:  # noqa: BLE001
        reply = f"⚠️  Terjadi kesalahan saat memanggil model:\n\n`{exc}`"

    history.append({"role": "assistant", "content": reply})
    return history, history, ""


def finish_session(reference_state: str, history: list) -> str:
    """
    Akhiri sesi dan hasilkan rangkuman evaluasi terstruktur dalam Markdown.
    """
    if not reference_state or not reference_state.startswith("ACTIVE\n"):
        return (
            "### Rangkuman Pemahaman\n\n"
            "⚠️  Belum ada sesi belajar yang berjalan. Silakan mulai sesi di Panel 1 "
            "terlebih dahulu."
        )

    reference = reference_state.split("\n", 1)[1]
    history = history or []

    if len(history) == 0:
        return (
            "### Rangkuman Pemahaman\n\n"
            "Belum ada percakapan dalam sesi ini, jadi tidak ada yang bisa dievaluasi. "
            "Coba jelaskan materi terlebih dahulu pada Panel 2."
        )

    try:
        summary = call_model(reference, history, user_message=None)
    except Exception as exc:  # noqa: BLE001
        summary = f"⚠️  Terjadi kesalahan saat menyusun rangkuman:\n\n`{exc}`"

    return f"### Rangkuman Pemahaman\n\n{summary}"


# ---------------------------------------------------------------------------
# Antarmuka Gradio
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Feynman Student AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎓 Feynman Student AI\n"
            "### Interactive Gap-Finder Tutor — CAICC 2026 Track 2\n"
            "Jelaskan materi dengan bahasamu sendiri. AI berperan sebagai *Murid Kritis* "
            "yang menemukan celah pemahamanmu lewat pertanyaan."
        )

        # State tersembunyi untuk menyimpan referensi + status sesi.
        session_state = gr.State("")

        with gr.Tab("1. Input Materi Referensi"):
            gr.Markdown("Tempel catatan kuliah / teks materi di bawah ini.")
            reference_input = gr.Textbox(
                label="Materi Referensi",
                placeholder="Tempel teks/catatan kuliah sebagai referensi materi utama...",
                lines=12,
            )
            start_btn = gr.Button("🚀 Mulai Sesi Belajar", variant="primary")

        with gr.Tab("2. Sesi Mengajar Interaktif"):
            chatbot = gr.Chatbot(
                label="Percakapan (Mahasiswa = Pengajar, AI = Murid Kritis)",
                height=420,
                type="messages",
                allow_tags=False,
            )
            user_input = gr.Textbox(
                label="Penjelasanmu",
                placeholder="Jelaskan materi dengan bahasamu sendiri di sini...",
                lines=3,
            )
            with gr.Row():
                send_btn = gr.Button("📤 Kirim Penjelasan", variant="primary")
                finish_btn = gr.Button("✅ Selesaikan Sesi & Ambil Rangkuman")

        with gr.Tab("3. Rangkuman Pemahaman"):
            gr.Markdown(
                "Evaluasi akhir sesi ditampilkan di bawah setelah kamu menekan "
                "**Selesaikan Sesi & Ambil Rangkuman**."
            )
            summary_output = gr.Markdown(
                "### Rangkuman Pemahaman\n\n*Belum ada sesi yang diselesaikan.*"
            )

        # ------ Wiring event handler ------
        start_btn.click(
            fn=start_session,
            inputs=reference_input,
            outputs=[session_state, chatbot, chatbot],
        )

        send_btn.click(
            fn=send_explanation,
            inputs=[session_state, chatbot, user_input],
            outputs=[chatbot, chatbot, user_input],
        )

        finish_btn.click(
            fn=finish_session,
            inputs=[session_state, chatbot],
            outputs=summary_output,
        )

        # Tekan Enter untuk kirim (submit), tanpa langsung mengosongkan referensi.
        user_input.submit(
            fn=send_explanation,
            inputs=[session_state, chatbot, user_input],
            outputs=[chatbot, chatbot, user_input],
        )

    return demo


if __name__ == "__main__":
    # Pastikan token tersedia lebih awal agar error terlihat jelas saat startup,
    # bukan saat aplikasi sedang dipakai.
    try:
        get_api_token()
    except RuntimeError as exc:
        print(f"[PERINGATAN] {exc}")
        print(
            "Aplikasi tetap akan berjalan, namun panggilan API akan gagal "
            "hingga token tersedia."
        )

    app = build_ui()
    app.launch()