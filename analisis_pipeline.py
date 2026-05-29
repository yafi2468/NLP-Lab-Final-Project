import os
import json
import time
from datetime import datetime
from jiwer import wer, cer

from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "corpus", "audio")
REFERENCE_FILE = os.path.join(BASE_DIR, "data", "corpus", "transcripts", "reference.json")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_reference():
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def ekstrak_id_ujaran(filename):
    """
    Ekstrak ID tanpa peduli NPM siapa pun.
    Mengubah '2360_audio01.wav' atau '2359_audio1.wav' menjadi 'audio1'.
    """
    try:
        bagian = filename.split('_')
        if len(bagian) >= 2:
            raw_id = bagian[1].replace(".wav", "") 
            angka = raw_id.replace("audio", "")
            angka_bersih = str(int(angka)) # Menghilangkan angka 0 di depan (01 -> 1)
            return f"audio{angka_bersih}"
    except Exception:
        return None
    return None

def main():
    print("[INFO] Memulai Evaluasi Pipeline Massal...")
    
    ground_truth = load_reference()
    print(f"[INFO] Berhasil memuat {len(ground_truth)} skrip transkrip asli.")
    
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"[INFO] Menemukan {len(audio_files)} file audio di folder.\n")
    print(f"[WARNING] Perkiraan waktu eksekusi: {len(audio_files) * 30 / 60:.1f} menit.\n")
    
    hasil_evaluasi = []
    wer_list = []
    cer_list = []
    
    # Dictionary untuk menampung Summary per Utterance
    rekap_per_ujaran = {}
    
    for filename in audio_files:
        utterance_id = ekstrak_id_ujaran(filename)
        
        if not utterance_id:
            continue
            
        print("-" * 55)
        print(f"Memproses Rekaman: {filename}")
        
        teks_referensi = ground_truth.get(utterance_id)
        if not teks_referensi:
            print(f"   [WARNING] Kunci jawaban untuk {utterance_id} belum ada di JSON! Melewati...")
            continue
            
        audio_path = os.path.join(AUDIO_DIR, filename)
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        for mode in ["normalize", "preserve"]:
            print(f"\n>> Menguji Mode : {mode.upper()}")
            
            try:
                waktu_mulai = time.time()
                
                # STT
                waktu_stt_mulai = time.time()
                teks_prediksi = transcribe_speech_to_text(audio_bytes, ".wav")
                durasi_stt = time.time() - waktu_stt_mulai
                
                # LLM
                waktu_llm_mulai = time.time()
                teks_balasan = generate_response(teks_prediksi, mode)
                durasi_llm = time.time() - waktu_llm_mulai
                
                # TTS
                waktu_tts_mulai = time.time()
                audio_balasan_path = transcribe_text_to_speech(teks_balasan)
                durasi_tts = time.time() - waktu_tts_mulai
                
                durasi_total = time.time() - waktu_mulai
                
                ref_bersih = teks_referensi.lower()
                pred_bersih = teks_prediksi.lower()
                
                nilai_wer = wer(ref_bersih, pred_bersih)
                nilai_cer = cer(ref_bersih, pred_bersih)
                
                wer_list.append(nilai_wer)
                cer_list.append(nilai_cer)
                
                # Masukkan data ke Rekap Per Ujaran
                if utterance_id not in rekap_per_ujaran:
                    rekap_per_ujaran[utterance_id] = {'wer': [], 'cer': [], 'latency': []}
                rekap_per_ujaran[utterance_id]['wer'].append(nilai_wer)
                rekap_per_ujaran[utterance_id]['cer'].append(nilai_cer)
                rekap_per_ujaran[utterance_id]['latency'].append(durasi_total)
                
                print(f"   Teks Asli   : {teks_referensi}")
                print(f"   Tebakan STT : {teks_prediksi}")
                print(f"   WER / CER   : {nilai_wer:.2f} / {nilai_cer:.2f}")
                print(f"   Waktu Total : {durasi_total:.2f} detik")
                
                hasil_evaluasi.append({
                    "file": filename,
                    "utterance_id": utterance_id,
                    "mode": mode,
                    "wer": round(nilai_wer, 4),
                    "cer": round(nilai_cer, 4),
                    "latency_stt": round(durasi_stt, 2),
                    "latency_llm": round(durasi_llm, 2),
                    "latency_tts": round(durasi_tts, 2),
                    "latency_total": round(durasi_total, 2)
                })
                
            except Exception as e:
                print(f"   [ERROR] Sistem gagal memproses: {e}")

    # Menghitung Summary Keseluruhan
    rata_wer_global = sum(wer_list) / len(wer_list) if wer_list else 0
    rata_cer_global = sum(cer_list) / len(cer_list) if cer_list else 0
    
    print("\n" + "=" * 55)
    print("RINGKASAN PER UJARAN (UTTERANCE)")
    print("=" * 55)
    
    summary_per_ujaran_final = {}
    
    urutan_id = sorted(rekap_per_ujaran.keys(), key=lambda x: int(x.replace('audio', '')))
    
    for uid in urutan_id:
        data = rekap_per_ujaran[uid]
        avg_wer = sum(data['wer']) / len(data['wer']) if data['wer'] else 0
        avg_cer = sum(data['cer']) / len(data['cer']) if data['cer'] else 0
        avg_latency = sum(data['latency']) / len(data['latency']) if data['latency'] else 0
        
        summary_per_ujaran_final[uid] = {
            "rata_rata_wer": round(avg_wer, 4),
            "rata_rata_cer": round(avg_cer, 4),
            "rata_rata_waktu_total": round(avg_latency, 2)
        }
        print(f"[{uid.upper()}] -> WER: {avg_wer:.4f} | CER: {avg_cer:.4f} | Avg Waktu: {avg_latency:.2f}s")

    print("\n" + "=" * 55)
    print("RINGKASAN SKOR AKHIR GLOBAL")
    print("=" * 55)
    print(f"Total File Dievaluasi : {len(hasil_evaluasi)}")
    print(f"Rata-rata WER Global  : {rata_wer_global:.4f}")
    print(f"Rata-rata CER Global  : {rata_cer_global:.4f}")
    
    # Cetak laporan ke JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nama_laporan = os.path.join(RESULTS_DIR, f"laporan_akhir_{timestamp}.json")
    
    with open(nama_laporan, "w", encoding="utf-8") as f:
        json.dump({
            "summary_global": {
                "total_evaluasi": len(hasil_evaluasi),
                "rata_rata_wer": round(rata_wer_global, 4),
                "rata_rata_cer": round(rata_cer_global, 4)
            },
            "summary_per_utterance": summary_per_ujaran_final,
            "detail_per_audio": hasil_evaluasi
        }, f, indent=4)
        
    print(f"\n[INFO] Laporan evaluasi lengkap telah disimpan ke: {nama_laporan}")

if __name__ == "__main__":
    main()