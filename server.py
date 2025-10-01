from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI
import tempfile
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

@app.post("/process_audio")
async def process_audio(file: UploadFile):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Step 1: Speech-to-text
    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            language="en"
        )
    text = transcript.text
    print(text);

    # Step 2: AI reasoning
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Nova, a smart, friendly, and conversational AI desk assistant. "
                    "Your tone is calm, helpful, slightly witty like JARVIS from Iron Man, "
                    "but always polite and precise like Alexa or Google Assistant. "
                    "You live on a desk, answer user questions, respond to voice commands, "
                    "and provide useful information or perform tasks."
                )
            },
            {"role": "user", "content": text}
            ]
    )
    reply_text = completion.choices[0].message.content
    print("Replying:")
    print(reply_text)

    # Step 3: Text-to-speech
    speech_file = tmp_path.replace(".wav", "_reply.wav")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=reply_text,
    ) as response:
        response.stream_to_file(speech_file)

    return FileResponse(speech_file, media_type="audio/wav")
