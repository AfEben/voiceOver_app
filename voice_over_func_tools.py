import os
import io
import math
import tempfile
import uuid
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
import numpy as np

# Video/Audio
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment

# Transcription
from faster_whisper import WhisperModel

# Diarization
from pyannote.audio import Pipeline as PyannotePipeline

# Translation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

# TTS (voice cloning)
from TTS.api import TTS

DEFAULT_SR = 16000
TARGET_AUDIO_SR = 22050  # XTTS works nicely here
MAX_REF_SECONDS = 60      # reference voice audio total for cloning
MIN_REF_SECONDS = 15

# Common languages for UI (NLLB language codes can be broader)
LANG_CHOICES = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab",
    "Hindi": "hin_Deva",
    "Swahili": "swh_Latn",
    "Turkish": "tur_Latn",
    "Polish": "pol_Latn",
    "Dutch": "nld_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
}

# Mapping NLLB code -> XTTS language tag (best-effort)
# XTTS v2 supports many languages with simple ISO codes; for unsupported, fallback to English ("en").
NLLB_TO_XTTS_LANG = {
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "deu_Latn": "de",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "rus_Cyrl": "ru",
    "arb_Arab": "ar",
    "hin_Deva": "hi",
    "swh_Latn": "sw",
    "tur_Latn": "tr",
    "pol_Latn": "pl",
    "nld_Latn": "nl",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
}


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "large-v3") -> WhisperModel:
    # Use GPU if available
    compute_type = "float16"
    device = "cuda" if torch_cuda_available() else "cpu"
    if device == "cpu":
        compute_type = "int8"  # more memory efficient on CPU

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(Path.home() / ".cache" / "whisper")
    )
    return model


def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_diarization_pipeline(hf_token: str) -> PyannotePipeline:
    # Requires access approval to pyannote models on HF
    return PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)


@st.cache_resource(show_spinner=False)
def load_translator(model_name: str = "facebook/nllb-200-distilled-600M"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = hf_pipeline("translation", model=mod, tokenizer=tok, device=0 if torch_cuda_available() else -1)
    return translator


@st.cache_resource(show_spinner=False)
# def load_tts(model_name: str = "tts_models/fr/css10/vits") -> TTS:
def load_tts(model_name: str = "tts_models/multilingual/multi-dataset/your_tts") -> TTS:
# def load_tts(model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> TTS:
    #return TTS(model_name)
    return TTS(model_name).to("cpu")

# XTTS-v2:
# tts_models/multilingual/multi-dataset/xtts_v2
# Hugging Face
# docs.coqui.ai
#
# XTTS-v1:
# tts_models/multilingual/multi-dataset/xtts_v1
# Hugging Face
# docs.coqui.ai
#
# YourTTS:
# tts_models/multilingual/multi-dataset/your_tts
# GitHub
# docs.coqui.ai
#
# 3. Fairseq-based multilingual models
# Model name format: tts_models/<lang-iso_code>/fairseq/vits
# Languages: ~1100 supported via Fairseq



def save_uploaded_file(uploaded_file, tmpdir: str) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    dst = Path(tmpdir) / f"input_{uuid.uuid4().hex}{suffix}"
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dst)


def download_youtube(url: str, tmpdir: str) -> Optional[str]:
    try:
        import yt_dlp
        outtmpl = str(Path(tmpdir) / f"yt_{uuid.uuid4().hex}.%(ext)s")
        ydl_opts = {
            "outtmpl": outtmpl,
            "format": "mp4/bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            # Ensure .mp4 extension after merge
            if not filepath.endswith(".mp4"):
                filepath = Path(filepath).with_suffix(".mp4")
            return str(filepath)
    except Exception as e:
        st.error(f"YouTube download failed: {e}")
        return None


def generate_audio(text="hello world", model_name='tts_models/en/ljspeech/glow-tts', output_path='output.wav'):
    tts = TTS(model_name=model_name).to("cpu")
    tts.tts_to_file(text, file_path=output_path)

def extract_audio(video_path: str, out_wav: str, sr: int = DEFAULT_SR) -> str:
    clip = VideoFileClip(video_path)
    audio = clip.audio
    if audio is None:
        raise RuntimeError("No audio track found in video.")
    audio.write_audiofile(out_wav, fps=sr, codec="pcm_s16le", logger=None)
    clip.close()
    return out_wav

def transcribe_audio(whisper: WhisperModel, wav_path: str, language: Optional[str] = None) -> Dict:
    segments, info = whisper.transcribe(
        wav_path,
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
        language=language,  # None -> auto-detect
        task="transcribe"
    )
    results = []
    for seg in segments:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    return {
        "segments": results,
        "language": info.language,
        "language_probability": info.language_probability
    }

def diarize_audio(pyannote: PyannotePipeline, wav_path: str) -> List[Dict]:
    diarization = pyannote(wav_path, min_speakers=1, max_speakers=8)
    # Convert to list of dict with timestamps and speaker labels
    diarized = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarized.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker)
        })
    diarized.sort(key=lambda x: x["start"])
    return diarized

def pick_main_speaker(diarized: List[Dict]) -> str:
    # Sum durations per speaker
    durations = {}
    for d in diarized:
        durations[d["speaker"]] = durations.get(d["speaker"], 0.0) + (d["end"] - d["start"])
    if not durations:
        return "SPEAKER_00"
    # Longest total speaking time == main character
    return max(durations.items(), key=lambda kv: kv[1])[0]

def cut_segments(wav_path: str, segments: List[Tuple[float, float]], out_path: str) -> str:
    audio = AudioSegment.from_file(wav_path)
    pieces = []
    for s, e in segments:
        s_ms = int(max(0, s) * 1000)
        e_ms = int(max(0, e) * 1000)
        piece = audio[s_ms:e_ms]
        pieces.append(piece)
    if not pieces:
        # fallback: take first N seconds
        pieces = [audio[:MAX_REF_SECONDS * 1000]]
    combined = sum(pieces)
    combined.export(out_path, format="wav")
    return out_path

def build_reference_voice(wav_path: str, diarized: List[Dict], main_speaker: str, tmpdir: str) -> str:
    # Select up to MAX_REF_SECONDS of main speaker audio
    ref_segments = [(d["start"], d["end"]) for d in diarized if d["speaker"] == main_speaker]
    # Greedy take until MAX_REF_SECONDS
    chosen = []
    total = 0.0
    for s, e in ref_segments:
        dur = e - s
        if total + dur <= MAX_REF_SECONDS or total < MIN_REF_SECONDS:
            chosen.append((s, e))
            total += dur
        if total >= MAX_REF_SECONDS:
            break
    ref_path = str(Path(tmpdir) / f"ref_{uuid.uuid4().hex}.wav")
    return cut_segments(wav_path, chosen, ref_path)

def assign_speakers_to_transcript(transcript: Dict, diarized: List[Dict]) -> List[Dict]:
    # For each transcript segment, assign the speaker overlapping most
    diarized_arr = diarized
    result = []
    for seg in transcript["segments"]:
        s, e = seg["start"], seg["end"]
        overlaps = {}
        for d in diarized_arr:
            os_ = max(s, d["start"])
            oe_ = min(e, d["end"])
            ov = max(0.0, oe_ - os_)
            if ov > 0:
                overlaps[d["speaker"]] = overlaps.get(d["speaker"], 0.0) + ov
        speaker = None
        if overlaps:
            speaker = max(overlaps.items(), key=lambda kv: kv[1])[0]
        else:
            speaker = "UNKNOWN"
        result.append({**seg, "speaker": speaker})
    return result

def chunk_text(text: str, max_chars: int = 400) -> List[str]:
    # Simple sentence-aware splitter (fallback)
    import re
    sentences = re.split(r'(?<=[\.\?\!\n])\s+', text.strip())
    chunks = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]

def translate_text(translator, text: str, src_lang_nllb: str, tgt_lang_nllb: str, max_chars=900) -> str:
    # chunk long text for NLLB
    pieces = chunk_text(text, max_chars=max_chars)
    outputs = []
    for p in pieces:
        out = translator(p, src_lang=src_lang_nllb, tgt_lang=tgt_lang_nllb, max_length=2000)
        outputs.append(out[0]["translation_text"])
    return " ".join(outputs)

def synthesize_tts_xtts(tts_model: TTS, text: str, speaker_wav: str, language_code: str, out_path: str):
# def synthesize_tts_xtts(tts_model: TTS, text: str, language_code: str, out_path: str, speaker_wav: str = 'rien'):
    # chunk long text to avoid memory spikes
    chunks = chunk_text(text, max_chars=400)
    audio = None
    for i, c in enumerate(chunks):
        tmp_chunk = Path(out_path).with_name(f"{Path(out_path).stem}_part{i}.wav")
        tts_model.tts_to_file(
            text=c,
            speaker_wav=speaker_wav,
            # language=language_code,
            language='fr-fr',
            file_path=str(tmp_chunk)
        )
        seg = AudioSegment.from_file(tmp_chunk)
        audio = seg if audio is None else (audio + seg)
    audio.export(out_path, format="wav")
    return out_path

def mux_audio_to_video(video_path: str, audio_path: str, out_path: str):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video = video.set_audio(new_audio)
    video.write_videofile(out_path, audio_codec="aac", codec="libx264", logger=None)
    video.close()
    new_audio.close()
    return out_path