import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVEN_KEY"))


def text_to_speech(
    text: str,
    output_path: str | Path = "output.mp3",
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",  # Default: George
    model_id: str = "eleven_multilingual_v2",
) -> tuple:
    """
    Convert text to speech using Eleven Labs API.

    Args:
        text: The text to convert to speech.
        output_path: Path to save the audio file.
        voice_id: Eleven Labs voice ID to use.
        model_id: Eleven Labs model ID to use.

    Returns:
        A tuple containing:
            - audio_generator: The generator yielding audio chunks from the API.
            - output_path: Path to the saved audio file.
    """
    output_path = Path(output_path)

    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
    )

    # Write audio chunks to file
    with open(output_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    return audio_generator, output_path


if __name__ == "__main__":
    # Example usage
    result = text_to_speech("Hello! This is a test of the text to speech system.")
    print(f"Audio saved to: {result}")
