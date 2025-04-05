import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import pdfplumber
import openai

# Set your OpenAI API key as an environment variable: OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Podcast + Audiobook Assistant MVP")

# ----------------------------
# Endpoint: PDF to Audio (TTS)
# ----------------------------
@app.post("/pdf-to-audio")
async def pdf_to_audio(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract text from PDF using pdfplumber
        extracted_text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Convert text to audio using Google Cloud TTS
        from google.cloud import texttospeech
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=extracted_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        tts_response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # Save synthesized audio to a temporary file
        audio_path = tmp_path + ".mp3"
        with open(audio_path, "wb") as out:
            out.write(tts_response.audio_content)
        
        return FileResponse(audio_path, media_type="audio/mpeg", filename="output.mp3")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup the temporary PDF file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --------------------------------------------------
# Endpoint: Audio Transcription & Show Notes Generation
# --------------------------------------------------
@app.post("/transcribe-and-notes")
async def transcribe_and_notes(file: UploadFile = File(...)):
    if file.content_type not in ["audio/mpeg", "audio/wav"]:
        raise HTTPException(status_code=400, detail="File must be an audio file (MP3 or WAV)")
    
    try:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name
        
        # Transcribe audio using OpenAI's Whisper model via the API
        transcript_response = openai.Audio.transcribe("whisper-1", open(audio_path, "rb"))
        transcript_text = transcript_response.get("text", "")
        
        if not transcript_text.strip():
            raise HTTPException(status_code=400, detail="Transcription failed or returned empty text")
        
        # Generate detailed show notes using GPT-4
        prompt = (
            "Using the following podcast transcript, generate detailed show notes "
            "including a summary, key timestamps, and notable quotes:\n\n"
            f"{transcript_text}"
        )
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        show_notes = chat_response.choices[0].message.content
        
        return JSONResponse(content={"transcript": transcript_text, "show_notes": show_notes})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
