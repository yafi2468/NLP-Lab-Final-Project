import os
import uuid
import tempfile
import torch
from TTS.api import TTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COQUI_DIR = os.path.join(BASE_DIR, "coqui_tts")
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")
COQUI_SPEAKER = "wibowo"

# 1. LOAD MODEL HANYA SATU KALI
print("[INFO] Memuat Model TTS ke dalam RAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_engine = TTS(model_path=COQUI_MODEL_PATH, config_path=COQUI_CONFIG_PATH).to(device)
print("[INFO] Model TTS Berhasil Dimuat!")

def text_to_ipa(text: str) -> str:
    """
    Mengubah abjad Indonesia biasa menjadi simbol fonetik IPA
    karena kamus model 'wibowo' mengharuskan format ini.
    """
    text = text.lower()
    
    text = text.replace("ng", "ŋ")
    text = text.replace("ny", "ɲ")
    text = text.replace("sy", "ʃ")
    
    text = text.replace("c", "tʃ")
    text = text.replace("j", "dʒ")
    text = text.replace("y", "j")
    
    text = text.replace("g", "ɡ") 
    
    text = text.replace("q", "k")
    text = text.replace("x", "ks")
    text = text.replace("v", "f")
    
    return text

def transcribe_text_to_speech(text: str) -> str:
    path = _tts_with_coqui(text)
    return path

def _tts_with_coqui(text: str) -> str:
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    ipa_text = text_to_ipa(text)
    
    print(f"[DEBUG] Teks Asli : {text}")
    print(f"[DEBUG] Teks IPA  : {ipa_text}")

    try:
        tts_engine.tts_to_file(text=ipa_text, speaker=COQUI_SPEAKER, file_path=output_path)
    except Exception as e:
        print(f"[ERROR] Native TTS failed: {e}")
        return "[ERROR] Failed to synthesize speech"

    return output_path