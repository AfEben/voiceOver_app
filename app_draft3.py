# app.py
import os
import io
import math
import tempfile
import uuid
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
from streamlit_player import st_player
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

import voice_over_func_tools as VO


# -----------------------------
# Configuration & Constants
# -----------------------------
st.set_page_config(page_title="Multilingual Dubbing Studio", page_icon="🎙️", layout="wide")

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
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("🎙️ Multilingual Dubbing Studio (Voice Clone)")


with st.sidebar:
    st.markdown("## Settings")
    st.caption("This app transcribes a video, finds the main speaker, clones the voice, translates, and dubs.")
    model_size = st.selectbox("Whisper model size", ["large-v3", "large-v3-turbo", "medium", "small"], index=0)
    hf_token = st.text_input("Hugging Face Token (for diarization)", type="password", help="Needed for pyannote speaker diarization.")
    skip_diarization = st.checkbox("Skip diarization (assume single speaker)", value=True)
    st.caption("Tip: GPU strongly recommended for faster-whisper & TTS.")

tab_input, tab_process, tab_results = st.tabs(["1) Input", "2) Process", "3) Results"])


with tab_input:
    st.subheader("Upload a video or paste a YouTube URL")
    uploaded = st.file_uploader("Upload MP4/MOV/MKV", type=["mp4", "mov", "mkv"])
    yt_url = st.text_input("...or YouTube URL (optional)")

    #if st.button("Preview video", type="secondary"):
     #   st_player(yt_url)

    target_langs = st.multiselect("Target languages", list(LANG_CHOICES.keys()), default=["French"])
    produce_dubbed_video = st.checkbox("Also produce dubbed video (replace audio track)", value=False)

    start_btn = st.button("Start Pipeline 🚀")

if start_btn:
    if not uploaded and not yt_url:
        st.error("Please upload a video or provide a YouTube URL.")
        st.stop()
    if not skip_diarization and not hf_token:
        st.warning("Diarization requires a Hugging Face token. Either provide one or check 'Skip diarization'.")
    if len(target_langs) == 0:
        st.error("Choose at least one target language.")
        st.stop()

    with tab_process:
        with st.spinner("Preparing..."):
            tmpdir = tempfile.mkdtemp(prefix="dubbing_", dir=os.getcwd())
            input_video_path = None
            if uploaded:
                input_video_path = VO.save_uploaded_file(uploaded, tmpdir)
            elif yt_url:
                input_video_path = VO.download_youtube(yt_url, tmpdir)
                if input_video_path is None:
                    st.stop()

            st.success("Video loaded.")
            st.video(input_video_path)

            # 1) Extract audio
            st.info("Extracting audio track...")
            wav_path = str(Path(tmpdir) / "audio.wav")
            VO.extract_audio(input_video_path, wav_path)
            st.success("Audio extracted.")

            # 2) Transcribe
            st.info(f"Loading Whisper ({model_size})...")
            whisper = VO.load_whisper(model_size)
            st.info("Transcribing (Whisper)...")
            transcript = VO.transcribe_audio(whisper, wav_path)
            base_lang = transcript.get("language", "eng")
            st.success(f"Transcribed ({base_lang}).")

            # 3) Diarize & choose main speaker
            diarized = []
            main_speaker = "SPEAKER_00"
            transcript_with_speakers = []
            ref_voice_path = None

            if skip_diarization:
                st.warning("Skipping diarization; assuming single speaker for reference voice.")
                # Use first 30-60s as reference
                ref_voice_path = VO.build_reference_voice(wav_path, [], main_speaker, tmpdir)
                transcript_with_speakers = [{**s, "speaker": main_speaker} for s in transcript["segments"]]
            else:
                try:
                    st.info("Loading diarization pipeline (pyannote)...")
                    dia = VO.load_diarization_pipeline(hf_token)
                    st.info("Finding speakers...")
                    diarized = VO.diarize_audio(dia, wav_path)
                    if len(diarized) == 0:
                        st.warning("No diarization segments found. Falling back to single-speaker assumption.")
                        ref_voice_path = VO.build_reference_voice(wav_path, [], main_speaker, tmpdir)
                        transcript_with_speakers = [{**s, "speaker": main_speaker} for s in transcript["segments"]]
                    else:
                        main_speaker = VO.pick_main_speaker(diarized)
                        st.success(f"Main speaker detected: {main_speaker}")
                        transcript_with_speakers = VO.assign_speakers_to_transcript(transcript, diarized)
                        ref_voice_path = VO.build_reference_voice(wav_path, diarized, main_speaker, tmpdir)
                except Exception as e:
                    st.error(f"Diarization failed: {e}")
                    st.warning("Falling back to single-speaker reference.")
                    ref_voice_path = VO.build_reference_voice(wav_path, [], main_speaker, tmpdir)
                    transcript_with_speakers = [{**s, "speaker": main_speaker} for s in transcript["segments"]]

            st.audio(ref_voice_path, format="audio/wav", start_time=0)
            st.caption("Reference audio used for voice cloning (main speaker).")

            # Join transcript into one text string for translation (option: only main speaker text)
            st.info("Preparing text for translation...")
            main_speaker_text = " ".join([s["text"] for s in transcript_with_speakers if s["speaker"] == main_speaker]).strip()
            if len(main_speaker_text) < 5:
                # fallback to full transcript
                main_speaker_text = " ".join([s["text"] for s in transcript["segments"]]).strip()
        #
            # updated original transcript should be stored in a variable
            transcribed_text = st.text_area("Original Transcript (detected language)", value=" ".join([s["text"] for s in transcript["segments"]]), height=200)
        #
        #     # 4) Translation
            st.info("Loading translator (NLLB-200 distilled 600M)...")
            translator = VO.load_translator()
        #
        #     # Map Whisper language (like 'en') to NLLB code; if unknown, assume eng
        #     # A simple mapping for common languages:
            WHISPER_TO_NLLB = {
                "en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn",
                "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab",
                "hi": "hin_Deva", "sw": "swh_Latn", "tr": "tur_Latn", "pl": "pol_Latn",
                "nl": "nld_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
            }
            src_lang_nllb = WHISPER_TO_NLLB.get(transcript.get("language", "en"), "eng_Latn")

            translations = {}
            for lang_name in target_langs:
                tgt_code = LANG_CHOICES[lang_name]
                st.info(f"Translating to {lang_name} ...")
                try:
                    # translations[lang_name] = VO.translate_text(translator, main_speaker_text, src_lang_nllb, tgt_code)
                    translations[lang_name] = VO.translate_text(translator, transcribed_text, src_lang_nllb, tgt_code)
                except Exception as e:
                    st.error(f"Translation to {lang_name} failed: {e}")

            st.success("Translation complete.")

            translated_text = st.text_area(f"Translated Transcript {lang_name}",
                                            #value=" ".join([s["text"] for s in translations[lang_name]]), height=200)
                                            value=translations[lang_name], height=200)
        #
            # 5) TTS with cloned voice (XTTS)
            st.info("Loading TTS (XTTS v2)...")
            tts_model = VO.load_tts()

            st.info(f"Synthesizing {lang_name} TTS with cloned voice...")
            out_wav = str(Path(tmpdir) / f"dub_{lang_name}_{uuid.uuid4().hex}.wav")
            lang_code_xtts = NLLB_TO_XTTS_LANG.get(LANG_CHOICES[lang_name], "en")

            # VO.generate_audio(text=translated_text, model_name=tts_model, output_path=out_wav)
            # VO.generate_audio(text=translated_text, model_name="tts_models/fr/css10/vits", output_path=out_wav)

            # st.audio(out_wav, format="audio/wav")
            outputs = []

            for lang_name, translated_text in translations.items():
                st.info(f"Synthesizing {lang_name} TTS with cloned voice...")
                out_wav = str(Path(tmpdir) / f"dub_{lang_name}_{uuid.uuid4().hex}.wav")
                lang_code_xtts = NLLB_TO_XTTS_LANG.get(LANG_CHOICES[lang_name], "en")
                try:
                   # VO.synthesize_tts_xtts(tts_model, translated_text, ref_voice_path, lang_code_xtts, out_wav)
                    VO.synthesize_tts_xtts(tts_model, translated_text, ref_voice_path, lang_code_xtts, out_wav)
                    outputs.append((lang_name, out_wav))
                    st.audio(out_wav, format="audio/wav")
                except Exception as e:
                    st.error(f"TTS for {lang_name} failed: {e}")
        #     #
        #     # 6) Optional: mux dubbed audio to video (create a dubbed video per language)
        #     dubbed_videos = []
        #     if produce_dubbed_video and outputs:
        #         st.info("Muxing dubbed audio into video...")
        #         for lang_name, dub_wav in outputs:
        #             out_mp4 = str(Path(tmpdir) / f"video_dub_{lang_name}_{uuid.uuid4().hex}.mp4")
        #             try:
        #                 VO.mux_audio_to_video(input_video_path, dub_wav, out_mp4)
        #                 dubbed_videos.append((lang_name, out_mp4))
        #             except Exception as e:
        #                 st.error(f"Muxing failed for {lang_name}: {e}")
        #
        #     st.session_state["pipeline_results"] = {
        #         "input_video": input_video_path,
        #         "audio_wav": wav_path,
        #         "transcript": transcript,
        #         # "diarized": diarized,
        #         # "main_speaker": main_speaker,
        #         # "ref_voice": ref_voice_path,
        #         # "translations": translations,
        #         # "tts_outputs": outputs,
        #         "dubbed_videos": dubbed_videos,
        #     }
        #     st.success("Pipeline complete ✅")
        # st.success("Pipeline complete ✅")

with tab_results:
    if "pipeline_results" in st.session_state:
        res = st.session_state["pipeline_results"]
        st.subheader("Artifacts")

        st.markdown("**Main speaker detected:** " + f"`{res['main_speaker']}`")
        st.markdown("**Reference voice:**")
        st.audio(res["ref_voice"])

        st.markdown("**Transcript (Segments):**")
        st.json(res["transcript"]["segments"])

        st.markdown("**Translations:**")
        st.json(res["translations"])

        st.markdown("**Dubbed audio files:**")
        for lang_name, path in res["tts_outputs"]:
            st.write(lang_name)
            st.audio(path)
            with open(path, "rb") as f:
                st.download_button(
                    label=f"Download {lang_name} audio",
                    data=f.read(),
                    file_name=f"dub_{lang_name}.wav",
                    mime="audio/wav"
                )

        if res["dubbed_videos"]:
            st.markdown("**Dubbed videos:**")
            for lang_name, path in res["dubbed_videos"]:
                st.write(lang_name)
                st.video(path)
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download {lang_name} video",
                        data=f.read(),
                        file_name=f"video_dub_{lang_name}.mp4",
                        mime="video/mp4"
                    )
    else:
        st.info("Run the pipeline in the **Process** tab.")
