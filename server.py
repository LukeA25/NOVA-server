from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import openai
import tempfile
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/process_audio")
async def process_audio(file: UploadFile):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Step 1: Speech-to-text
    with open(tmp_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    text = transcript.text

    # Step 2: AI reasoning
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":text}]
    )
    reply_text = completion.choices[0].message.content

    # Step 3: Text-to-speech
    speech_file = tmp_path.replace(".wav", "_reply.wav")
    with open(speech_file, "wb") as f:
        response = openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply_text
        )
        f.write(response.read())  # stream audio to file

    return FileResponse(speech_file, media_type="audio/wav")
