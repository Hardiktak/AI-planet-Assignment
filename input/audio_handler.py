"""
Audio Input Handler — ASR pipeline using OpenAI hosted transcription.
"""

import io
import os
import re
from typing import Dict, Any

ASR_CONFIDENCE_THRESHOLD = 0.6

# Math phrase normalization rules
MATH_PHRASE_MAP = {
    r"square root of (\w+)": r"sqrt(\1)",
    r"(\w+) squared": r"\1^2",
    r"(\w+) cubed": r"\1^3",
    r"(\w+) raised to the power (\w+)": r"\1^\2",
    r"(\w+) raised to (\w+)": r"\1^\2",
    r"integral of": "integrate",
    r"derivative of": "d/dx",
    r"d y by d x": "dy/dx",
    r"dy by dx": "dy/dx",
    r"d by d x": "d/dx",
    r"d by dx": "d/dx",
    r"greater than or equal to": ">=",
    r"less than or equal to": "<=",
    r"greater than": ">",
    r"less than": "<",
    r"not equal to": "!=",
    r"plus or minus": "+-",
    r"minus or plus": "-+",
}


class AudioHandler:
    """Handles audio input with hosted ASR transcription."""

    def __init__(self, client):
        self.client = client
        self.name = "Audio Handler"

    def process_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = filename
        return self.process(file_obj)

    def process(self, audio_file) -> Dict[str, Any]:
        try:
            transcript = self.client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )

            text = transcript.text.strip()

            result = {
                "text": text,
                "confidence": 0.9,  # Hosted API doesn't return logprobs
                "engine": "openai-hosted",
                "model": "gpt-4o-mini-transcribe",
            }

        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "error": f"Transcription failed: {str(e)}",
                "needs_review": True,
                "input_type": "audio",
            }

        # Normalize math phrases
        if result.get("text"):
            result["raw_transcript"] = result["text"]
            result["text"] = self._normalize_math(result["text"])

        result["input_type"] = "audio"
        result["needs_review"] = result["confidence"] < ASR_CONFIDENCE_THRESHOLD
        result["raw_input"] = "audio_stream"

        return result

    def _normalize_math(self, text: str) -> str:
        result = text
        for pattern, replacement in MATH_PHRASE_MAP.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
