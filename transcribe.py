import whisper

model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\kanna\Downloads\videoplayback.mp3")
print(result["text"])
