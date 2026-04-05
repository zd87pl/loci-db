"""Cloud VLM client for complex scene descriptions.

When YOLO has low confidence or detects fewer objects than expected,
this client sends the frame to GPT-4o mini (or Gemini 2.5 Flash) and
extracts structured object data for LOCI-DB ingestion.

The VLM is prompted to return a JSON list of objects with estimated
normalized positions, so the output is directly compatible with the
spatial_memory.observe() API.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_SCENE_PROMPT = """\
You are a spatial scene analyzer helping a blind person track their belongings.

Analyze this image and identify all visible objects that a person might want to find later.
Focus on everyday items: keys, phone, wallet, glasses, remote, cup, bottle, bag, etc.

Return ONLY a JSON array with this exact structure (no other text):
[
  {
    "label": "object name in lowercase",
    "cx": 0.5,
    "cy": 0.5,
    "width": 0.1,
    "height": 0.1,
    "confidence": 0.9,
    "description": "brief description for voice output"
  }
]

Rules:
- cx, cy, width, height are normalized [0,1] relative to image dimensions
- cx=0 is left edge, cx=1 is right edge
- cy=0 is top edge, cy=1 is bottom edge
- confidence is your certainty that this object is correctly identified [0,1]
- Only include objects you can clearly identify
- Use simple, natural label names (e.g., "water bottle" not "plastic_bottle_0")
- If no relevant objects are visible, return an empty array: []
"""


class VLMClient:
    """Wraps GPT-4o mini (or Gemini 2.5 Flash) for scene understanding.

    Configuration via environment variables:
      VLM_PROVIDER: "openai" (default) | "gemini"
      OPENAI_API_KEY: required for OpenAI
      GOOGLE_API_KEY: required for Gemini
      VLM_MODEL: model override (e.g. "gpt-4o-mini", "gemini-2.5-flash-preview-04-17")
    """

    def __init__(self) -> None:
        self._provider = os.environ.get("VLM_PROVIDER", "openai").lower()
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._google_key = os.environ.get("GOOGLE_API_KEY", "")
        self._model = os.environ.get("VLM_MODEL", "")
        self._client = None   # lazy-init

    def _get_openai_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._openai_key)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    @property
    def is_available(self) -> bool:
        if self._provider == "openai":
            return bool(self._openai_key)
        if self._provider == "gemini":
            return bool(self._google_key)
        return False

    async def describe_scene(self, image_bytes: bytes) -> list[dict]:
        """Analyze an image and return a list of detected objects with positions.

        Returns:
            List of dicts with keys: label, cx, cy, width, height, confidence, description
        """
        if not self.is_available:
            logger.debug("VLM not available (no API key configured)")
            return []

        if self._provider == "openai":
            return await self._describe_openai(image_bytes)
        elif self._provider == "gemini":
            return await self._describe_gemini(image_bytes)
        else:
            logger.warning("Unknown VLM provider: %s", self._provider)
            return []

    async def _describe_openai(self, image_bytes: bytes) -> list[dict]:
        """Use GPT-4o mini to describe the scene."""
        client = self._get_openai_client()
        model = self._model or "gpt-4o-mini"

        b64 = base64.b64encode(image_bytes).decode()
        # Determine image type from magic bytes
        if image_bytes[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        else:
            mime = "image/jpeg"  # default

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _SCENE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or "[]"
            return _parse_vlm_response(raw)
        except Exception as e:
            logger.error("OpenAI VLM call failed: %s", e)
            return []

    async def _describe_gemini(self, image_bytes: bytes) -> list[dict]:
        """Use Gemini 2.5 Flash to describe the scene."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._google_key)
            model_name = self._model or "gemini-2.5-flash-preview-04-17"
            model = genai.GenerativeModel(model_name)

            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))

            response = await model.generate_content_async([_SCENE_PROMPT, img])
            raw = response.text or "[]"
            return _parse_vlm_response(raw)
        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            return []
        except Exception as e:
            logger.error("Gemini VLM call failed: %s", e)
            return []

    async def answer_location_question(
        self,
        question: str,
        objects: list[dict],
        image_bytes: bytes | None = None,
    ) -> str:
        """Answer a natural language location question given known object positions.

        This is used when the spatial memory query needs LLM reasoning to
        compose a natural response (e.g., "your keys are near the left side
        of the table, about 2 meters from where you're standing").

        Args:
            question: The user's spoken question.
            objects: List of ObjectObservation.to_dict() results from LOCI-DB.
            image_bytes: Optional current frame for visual context.

        Returns:
            A natural language answer suitable for TTS output.
        """
        if not self.is_available:
            return _fallback_location_answer(question, objects)

        if self._provider == "openai":
            return await self._answer_openai(question, objects, image_bytes)
        return _fallback_location_answer(question, objects)

    async def _answer_openai(
        self,
        question: str,
        objects: list[dict],
        image_bytes: bytes | None,
    ) -> str:
        client = self._get_openai_client()
        model = self._model or "gpt-4o-mini"

        objects_summary = json.dumps(objects, indent=2) if objects else "No objects found."

        system = (
            "You are a helpful assistant for a blind person. "
            "Answer location questions concisely using the spatial memory data provided. "
            "Describe positions in natural terms (left/right/center, near/far). "
            "cx=0 is far left, cx=1 is far right, cy=0 is top/far, cy=1 is bottom/near. "
            "Keep answers under 2 sentences. Speak naturally, as if to a person."
        )

        user_content: list[dict] = [
            {
                "type": "text",
                "text": f"Question: {question}\n\nSpatial memory data:\n{objects_summary}",
            }
        ]
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content or "I could not find that object."
        except Exception as e:
            logger.error("OpenAI answer call failed: %s", e)
            return _fallback_location_answer(question, objects)


def _parse_vlm_response(raw: str) -> list[dict]:
    """Parse VLM JSON response, tolerating markdown fences and trailing text."""
    # Strip markdown code fences if present
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Find the JSON array
    arr_match = re.search(r"\[[\s\S]*\]", raw)
    if not arr_match:
        return []

    try:
        data = json.loads(arr_match.group(0))
        validated = []
        for obj in data:
            if not isinstance(obj, dict) or "label" not in obj:
                continue
            validated.append({
                "label": str(obj.get("label", "object")).strip().lower(),
                "cx": float(obj.get("cx", 0.5)),
                "cy": float(obj.get("cy", 0.5)),
                "width": float(obj.get("width", 0.1)),
                "height": float(obj.get("height", 0.1)),
                "confidence": float(obj.get("confidence", 0.7)),
                "description": str(obj.get("description", "")),
            })
        return validated
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to parse VLM response: %s — raw: %.200s", e, raw)
        return []


def _fallback_location_answer(question: str, objects: list[dict]) -> str:
    """Rule-based location answer when VLM is not available."""
    if not objects:
        return "I don't have that object in memory. Try scanning the area with the camera."

    obj = objects[0]
    label = obj.get("label", "object")
    cx = obj.get("cx", 0.5)
    cy = obj.get("cy", 0.5)
    age = obj.get("age_seconds", 0)

    # Describe horizontal position
    if cx < 0.33:
        h_pos = "on the left"
    elif cx > 0.67:
        h_pos = "on the right"
    else:
        h_pos = "in the center"

    # Describe vertical/depth position
    if cy < 0.4:
        v_pos = "toward the back"
    elif cy > 0.7:
        v_pos = "near the front"
    else:
        v_pos = ""

    pos_desc = h_pos + (f", {v_pos}" if v_pos else "")

    if age < 60:
        time_desc = f"{int(age)} seconds ago"
    elif age < 3600:
        time_desc = f"{int(age / 60)} minutes ago"
    else:
        time_desc = f"{int(age / 3600)} hours ago"

    return f"Your {label} was last seen {pos_desc}, {time_desc}."
