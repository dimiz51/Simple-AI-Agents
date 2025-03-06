import asyncio
from pathlib import Path
import tempfile
import uuid
import yt_dlp

async def download_youtube_audio(url: str) -> str:
    """
    Downloads a YouTube video and extracts its audio as an MP3 file, storing it in a temporary directory.
    Returns the path to the downloaded MP3 file and the title of the video.
    """
    temp_dir = Path(tempfile.gettempdir())  # Use system temp directory
    audio_file_id = str(uuid.uuid4())
    output_dir = str(temp_dir / f"{audio_file_id}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    loop = asyncio.get_running_loop()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = await loop.run_in_executor(None, ydl.extract_info, url)
            video_title = info_dict.get("title", "Unknown Title")
        except Exception as e:
            return str(e)

    audio_file = f"{output_dir}.mp3"

    # Pack response
    response = f"Audio file: {audio_file}, Title: {video_title}"
    return response
