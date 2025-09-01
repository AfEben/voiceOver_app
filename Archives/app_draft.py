import streamlit as st

import ffmpeg #from moviepy.editor import VideoFileClip
import whisper
from googletrans import Translator
from TTS.api import TTS
import tempfile
import os

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load TTS model
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

# Translator
translator = Translator()

st.title("🎙️ Multilingual Voice-Over Generator")

# Upload video
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path)

    # Extract audio
    st.info("Extracting audio...")
    #video = VideoFileClip(video_path)
    video = ffmpeg.input('example.mp4').output('output_audio.mp3').run()

    audio_path = video_path.replace(".mp4", ".wav")
    video.audio.write_audiofile(audio_path)

    # Transcribe audio
    st.info("Transcribing audio...")
    result = whisper_model.transcribe(audio_path)
    transcription = result["text"]
    st.success("Transcription complete!")
    st.text_area("Transcription", transcription, height=200)

    # Upload reference voice sample
    st.subheader("Upload a short voice sample of the main character (WAV format)")
    reference_audio = st.file_uploader("Voice sample", type=["wav"])

    # Choose target language
    target_lang = st.selectbox("Translate to:", ["fr", "es", "de", "zh-cn", "sw", "ar", "pt", "hi"])

    if st.button("Generate Translated Voice-Over"):
        # Translate transcription
        st.info("Translating text...")
        translated = translator.translate(transcription, dest=target_lang).text
        st.text_area("Translated Text", translated, height=200)

        # Generate speech
        st.info("Generating speech...")
        output_path = os.path.join(tempfile.gettempdir(), "../output.wav")
        tts_model.tts_to_file(text=translated, speaker_wav=reference_audio, file_path=output_path)

        st.audio(output_path, format="audio/wav")
        st.success("Voice-over generated!")

