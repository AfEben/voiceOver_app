import streamlit as st
from streamlit_player import st_player
from cop_voice_over import download_youtube, extract_audio, speech_to_text
import cop_voice_over

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    yt_url = st.text_input("YT link", "Enter youtube video link")
    audio_output_name = st.text_input("Audio output file name", "audio_1")
#    yt_url = st.text_input("YT link", "Enter youtube video link")
    if st.button("Enter", type="secondary"):
        st_player(yt_url)

    if st.button("Transcribe video", type="primary"):
        path_downloaded_video = download_youtube(yt_url)
        extract_audio(video_path=path_downloaded_video, out_wav=rf"C:\Users\Hp\Desktop\Cours\repos_githubs\voiceOver_app\extracted_audios\{audio_output_name}.wav")
        transcribed_video = speech_to_text(rf"C:\Users\Hp\Desktop\Cours\repos_githubs\voiceOver_app\extracted_audios\{audio_output_name}.wav", path_output_audio=None)

        with col2:
            st.write("Please review the text and adjust if needed")
            #st.text_input("Adjust if needed", transcribed_video)
            #st.markdown(transcribed_video)
            updated_text = st.text_area("Original language", value=transcribed_video)

        with col3:
            st.write("Please review the text and adjust if needed")
            st.info("Loading translator (NLLB-200 distilled 600M)...")
            translator = cop_voice_over.load_translator()

            # Map Whisper language (like 'en') to NLLB code; if unknown, assume eng
            # A simple mapping for common languages:
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
                    translations[lang_name] = translate_text(translator, main_speaker_text, src_lang_nllb, tgt_code)
                except Exception as e:
                    st.error(f"Translation to {lang_name} failed: {e}")

            st.success("Translation complete.")

            translated_text = st.text_area("Translated language", value=transcribed_video)
