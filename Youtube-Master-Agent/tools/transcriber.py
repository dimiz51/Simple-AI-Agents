import aiohttp
import os
import dotenv

dotenv.load_dotenv()

# Endpoint url for Whisper (STT model) from HuggingFaceInferenceAPI
WHISPER_HF_API_URL = (
    "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
)


async def transcribe_audio(audio_file: str) -> str:
    """
    Asynchronously transcribes an MP3 audio file into text using the Hugging Face Inference API.
    """

    # Load HF token
    HF_TOKEN = os.getenv("HF_API_KEY")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Verify audio file ends with .mp3 and remove duplicate '.mp3'
    if not audio_file.endswith(".mp3"):
        return "Audio file must be in MP3 format."

    # Using aiohttp for non-blocking HTTP requests
    async with aiohttp.ClientSession() as session:
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        # Send the audio file for transcription
        async with session.post(
            WHISPER_HF_API_URL, headers=headers, data=audio_data
        ) as response:
            result = await response.json()

    # Return transcription text or error message
    return result.get(
        "text",
        "Transcription failed for {audio_file}. Either no word was detected or there is something wrong with the audio file.",
    )
