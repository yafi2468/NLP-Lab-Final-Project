# 🎙️ Voice Assistant Pro: Code-Switching Speech-to-Speech System

## 📝 Deskripsi Proyek
Proyek Akhir (UAS) Praktikum Natural Language Processing (NLP) ini adalah implementasi sistem **multilingual speech-to-speech end-to-end** yang dirancang secara individu. Sistem ini mampu menerima input ujaran berupa pencampuran bahasa (*code-switching*) antara **Bahasa Indonesia, Inggris, dan Arab**, memprosesnya melalui serangkaian arsitektur NLP, lalu menghasilkan output berupa respons suara kembali dari sistem secara natural.

Fokus utama dari proyek ini adalah penyusunan korpus *speech code-switching* yang terkontrol, konsisten, serta evaluasi performa *pipeline* secara menyeluruh.

---

## ⚙️ Alur Arsitektur & Pipeline
Sistem ini mengintegrasikan empat lapisan utama dalam satu siklus percakapan utuh:
`Speech Input` ➡️ `STT` ➡️ `Text Processing & Normalization` ➡️ `LLM` ➡️ `TTS` ➡️ `Speech Output`

1. **STT (Speech-to-Text):** Menggunakan implementasi lokal `whisper.cpp` (model *large-v3-turbo*) untuk mentranskripsi ujaran *code-switching* ke dalam teks.
2. **Text Processing (G2P):** Mengubah teks abjad biasa menjadi simbol fonetik International Phonetic Alphabet (IPA) secara dinamis agar kompatibel dengan kamus vokal lokal.
3. **LLM (Large Language Model):** Memanfaatkan **Google Gemini API** melalui Google Gen AI SDK resmi untuk menghasilkan respons kontekstual yang cerdas berdasarkan *system prompt* eksplisit.
4. **TTS (Text-to-Speech):** Menggunakan **Coqui TTS** dengan model *Indonesian-TTS VITS* lokal (suara "wibowo") untuk menyintesis kembali respons teks menjadi suara balasan.

Sistem mendukung dua mode operasional utama:
* **`preserve`**: Merespons dengan mempertahankan pola pencampuran bahasa asli (*code-switching*).
* **`normalize`**: Menormalisasi bahasa ke bentuk yang lebih seragam dan formal.

---

## 📂 Struktur Folder Proyek
```text
voice-cs-system/
├── app/
│   ├── main.py              # RestAPI Endpoint (FastAPI)
│   ├── stt.py               # Transkripsi Whisper.cpp
│   ├── llm.py               # Pemrosesan Konteks Gemini API
│   └── tts.py               # Sintesis Suara Coqui TTS (Native API)
├── coqui_tts/               # Aset lokal model TTS ("wibowo")
├── data/
│   └── corpus/              # Dataset rekaman audio sampel (.wav)
├── gradio_app/
│   └── app.py               # Antarmuka Web (Gradio Light UI Pro)
├── models/
│   └── whisper.cpp/         # Repositori & file binari model Whisper
├── analisis_pipeline.py     # Script otomatis pengujian seluruh korpus (WER & CER)
├── requirements.txt         # Daftar dependensi library Python
└── .env                     # File rahasia API Key Gemini (Diabaikan oleh Git)
