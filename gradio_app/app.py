import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

def voice_chat(audio, mode):
    if audio is None:
        return None
    
    sr, audio_data = audio
    # Proteksi anti-crash jika audio 0.00 detik
    if audio_data is None or len(audio_data) == 0:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr,audio_data)
        audio_path = tmpfile.name

    with open(audio_path, "rb") as f:
        files = {"file": ("voice.wav", f, "audio/wav")}
        data = {"mode": mode}
        response = requests.post("http://localhost:8000/voice-chat", files=files, data=data)

    if response.status_code == 200:
        output_audio_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        with open(output_audio_path, "wb") as f:
            f.write(response.content)
        return output_audio_path
    else:
        return None

custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    body_background_fill="#f8fafc",          # Warna latar off-white yang sangat lembut
    block_background_fill="#ffffff",         # Card berwarna putih bersih
    block_border_width="0px",                # Hilangkan garis tepi kaku
    block_shadow="0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05)", # Bayangan premium
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

custom_css = """
h1 {text-align: center; color: #0f172a; font-weight: 800; letter-spacing: -1.5px; font-size: 2.5em !important;}
p {text-align: center; color: #64748b; font-size: 1.1em;}
.gradio-container {max-width: 850px !important; margin: auto; padding-top: 40px;}
"""

with gr.Blocks(theme=custom_theme, css=custom_css) as demo:
    gr.Markdown("# 🎙️ Wibowo.AI")
    gr.Markdown("Berbicara langsung ke mikrofon dan dapatkan jawaban suara dari asisten AI Anda.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources="microphone", type="numpy", label="Rekam Pertanyaan Anda")
            mode_input = gr.Dropdown(
                choices=["normalize", "preserve"],
                value="normalize",
                label="Mode Respons",
                info="Pilih bagaimana AI menangani bahasa campuran"
            ) 
            submit_btn = gr.Button("Submit", variant="primary")
            
        with gr.Column(scale=1):
            audio_output = gr.Audio(type="filepath", label="Balasan dari Asisten", interactive=False)

    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, mode_input], 
        outputs=audio_output
    )

demo.launch()