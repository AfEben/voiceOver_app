
import os
import io
import math
import tempfile
import uuid
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np




# Video/Audio
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment

# Transcription
from faster_whisper import WhisperModel

# Diarization
#from pyannote.audio import Pipeline as PyannotePipeline

# Translation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

# TTS (voice cloning)
from TTS.api import TTS


DEFAULT_SR = 16000
TARGET_AUDIO_SR = 22050  # XTTS works nicely here
MAX_REF_SECONDS = 60      # reference voice audio total for cloning
MIN_REF_SECONDS = 15


def extract_audio(video_path: str, out_wav: str, sr: int = DEFAULT_SR) -> str:
    clip = VideoFileClip(video_path)
    audio = clip.audio
    if audio is None:
        raise RuntimeError("No audio track found in video.")
    audio.write_audiofile(out_wav, fps=sr, codec="pcm_s16le",
                          logger=None)
    clip.close()
    # return out_wav


def download_youtube(url: str, tmpdir: str = r'C:\Users\Hp\Desktop\Cours\repos_githubs\voiceOver_app\downloaded_vids') -> Optional[str]:
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
        raise RuntimeError(f"YouTube download failed: {e}")


def speech_to_text(path_input_audio, path_output_audio=None):
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path_input_audio, beam_size=5)

    doc = """"""
    for segment in segments:
        # doc = doc + "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text) + '\n'
        doc = doc + segment.text + ' '
    return doc


def load_translator(model_name: str = "facebook/nllb-200-distilled-600M"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = hf_pipeline("translation", model=mod, tokenizer=tok, device=-1)
    return translator


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


