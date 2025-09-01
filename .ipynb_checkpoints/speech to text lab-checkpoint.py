from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
doc = """ """
for segment in segments:
    doc = doc + "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text) + '\n'