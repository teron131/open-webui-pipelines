from functools import lru_cache
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai.chat_models.base import ChatOpenAI
from opencc import OpenCC
from pytubefix import YouTube

from .whisper_fal import whisper_fal_transcribe
from .whisper_hf import whisper_hf_transcribe

load_dotenv()


# File handling functions


def create_cache_dir(title: str) -> Path:
    """Create a cache directory for the given title."""
    cache_dir = Path(f".cache/{title}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_output_path(cache_dir: Path, title: str) -> Path:
    """Get the output path for the given cache directory and title."""
    return Path(cache_dir / title)


def read_file(file_path: Path) -> str:
    """Read a file with multiple encoding attempts."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_file(file_path: Path, content: str) -> None:
    """Write content to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


# Subtitle processing functions


@lru_cache(maxsize=None)
def s2hk(content: str) -> str:
    converter = OpenCC("s2hk")
    return converter.convert(content)


def llm_format_txt(txt_filepath: str, chunk_size: int = 1000) -> None:
    """Format subtitles using LLM."""
    txt_path = Path(txt_filepath).with_suffix(".txt")

    preprocess_subtitles_chain = (
        hub.pull("preprocess_subtitles")
        | ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        | StrOutputParser()
        | RunnableLambda(s2hk)
    )

    subtitles = read_file(txt_path)
    chunked_subtitles = [subtitles[i : i + chunk_size] for i in range(0, len(subtitles), chunk_size)]

    formatted_subtitles = preprocess_subtitles_chain.batch([{"subtitles": chunk} for chunk in chunked_subtitles])
    formatted_subtitles = "".join(formatted_subtitles)

    write_file(txt_path, formatted_subtitles)
    print(f"Formatted TXT: {txt_path}")


# YouTube video processing functions


def download_audio(youtube: YouTube, cache_dir: Path, output_path: Path) -> None:
    """Download audio from YouTube video."""
    mp3_path = output_path.with_suffix(".mp3")
    if mp3_path.exists():
        print(f"Audio file already exists: {mp3_path}")
    else:
        youtube.streams.get_audio_only().download(output_path=str(cache_dir), mp3=True)
        print(f"Downloaded audio: {mp3_path}")


def download_subtitles(youtube: YouTube, output_path: Path) -> bool:
    """Download subtitles from YouTube video."""
    txt_path = output_path.with_suffix(".txt")

    if txt_path.exists():
        print(f"TXT already exists: {txt_path}")
        return True

    preferred_langs = ["zh-HK", "zh-CN", "en"]
    for lang in preferred_langs:
        if lang in youtube.captions:
            youtube.captions[lang].save_captions(filename=txt_path)
            print(f"Downloaded subtitle: {txt_path}")
            if lang == "zh-CN":
                content = s2hk(read_file(txt_path))
                write_file(txt_path, content)
                print(f"Converted subtitle: {txt_path}")
            return True

    print("No suitable subtitles found for download.")
    return False


def response_to_txt(result: Dict, txt_path: str) -> None:
    """
    Process the transcription result into a plain text format and write to a file.

    Args:
        result (Dict): The transcription result from the Whisper model.
        txt_path (str): The path to the output text file.

    Returns:
        None
    """
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        for chunk in result["chunks"]:
            transcript = chunk["text"].strip()
            transcript = s2hk(transcript)
            txt_file.write(f"{transcript}\n")


def process_subtitles(youtube: YouTube, output_path: Path, always_transcribe: bool = False, whisper_model: str = "fal") -> None:
    """Process subtitles: download or transcribe as needed."""
    mp3_path = output_path.with_suffix(".mp3")
    txt_path = output_path.with_suffix(".txt")
    available_subtitles = youtube.captions
    print(f"Available subtitles: {available_subtitles}")

    if txt_path.exists():
        print(f"TXT already exists: {txt_path}")
        return

    if not always_transcribe and download_subtitles(youtube, output_path):
        return

    transcribe_language = "en"
    if "a.en" in available_subtitles:
        transcribe_language = "en"
    elif available_subtitles is None or available_subtitles in ["zh-HK", "zh-CN"]:
        transcribe_language = "zh"

    if whisper_model == "fal":
        response = whisper_fal_transcribe(str(mp3_path), language=transcribe_language)
    elif whisper_model == "hf":
        response = whisper_hf_transcribe(str(mp3_path))
    else:
        raise ValueError(f"Unsupported whisper model: {whisper_model}")

    response_to_txt(response, str(txt_path))
    print(f"Transcribed TXT: {txt_path}")


def process_youtube_video(url: str, always_transcribe: bool = False, whisper_model: str = "fal") -> str:
    """Process a YouTube video: download audio and handle subtitles."""
    try:
        youtube = YouTube(url)
        cache_dir = create_cache_dir(youtube.title)
        output_path = get_output_path(cache_dir, youtube.title)
        txt_path = output_path.with_suffix(".txt")

        download_audio(youtube, cache_dir, output_path)
        process_subtitles(youtube, output_path, always_transcribe, whisper_model)
        llm_format_txt(txt_path)

        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read()

    except Exception as e:
        print(f"Error processing video {url}: {str(e)}")
