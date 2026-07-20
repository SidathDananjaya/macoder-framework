# Phase 9: send the finished session report to a LOCAL Ollama model to write a plain-English interpretation. The LLM does not analyse behaviour (the trained models do that) - it only summarises the report, grounded to its numbers and framed as indicators, not proof. Runs offline with no API key, returns structured JSON, and falls back to a rule-based summary if Ollama is unavailable.

import json
import os

import requests


# Local Ollama configuration (override via environment).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

# Short JSON output, so a small token budget and low temperature are enough; thinking is disabled since summarising needs no reasoning.
NUM_PREDICT = 700
TEMPERATURE = 0.3
# Generous timeout: a 9B model on CPU is slow, and the running vision+audio pipeline contends for CPU, so generation can take minutes.
REQUEST_TIMEOUT = 600  # seconds

# Disable the model's reasoning phase (override with OLLAMA_THINK=true); models that don't support the flag are retried without it.
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "false").lower() == "true"

# The interpretation schema the model must return (Ollama structured output).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "observations": {"type": "array", "items": {"type": "string"}},
        "overall_assessment": {"type": "string"},
        "confidence_note": {"type": "string"},
    },
    "required": [
        "headline",
        "observations",
        "overall_assessment",
        "confidence_note",
    ],
}

SYSTEM_PROMPT = (
    "You are a behavioural-analysis assistant for an interview decision-support "
    "tool. You are given a structured report of multimodal signals captured "
    "during an interview (facial emotion, stress, gaze, blink rate, head "
    "movement and a computed deception-risk score over time).\n\n"
    "Write a concise, professional interpretation of the session for a human "
    "reviewer. Rules you must follow:\n"
    "1. Reference ONLY numbers, levels and timestamps that appear in the report. "
    "Never invent figures or events.\n"
    "2. These signals are INDICATORS, not proof, of deception. Frame everything "
    "as correlation, not causation. Elevated stress or gaze aversion can have "
    "innocent explanations (nerves, difficulty, environment).\n"
    "3. This is decision support only - it must never be presented as a verdict "
    "of guilt or a definitive lie-detection result.\n"
    "4. Tie observations to their moments in time where the timeline supports it "
    "(e.g. a stress spike at a given timestamp).\n\n"
    "Return ONLY a JSON object with these fields: 'headline' (a one-line "
    "verdict), 'observations' (3-6 short bullet strings, each citing a figure or "
    "time), 'overall_assessment' (a short flowing paragraph), and "
    "'confidence_note' (the limitations, stating the signals are indicators "
    "rather than proof).\n\n"
    "Be concise: one sentence per observation, and keep the overall assessment "
    "to 3-4 sentences. Do not include any text outside the JSON object."
)


class LLMInterpreter:
    """Turns a Phase-8 report into a locally-generated interview interpretation."""

    def interpret(self, report):
        # Return a structured interpretation of the report; falls back to a rule-based one when the local model can't be used.

        report = report or {}

        # No point calling the model for an empty recording.
        if not report.get("frame_count"):
            return self._fallback(
                report,
                "No session data was recorded, so no interpretation could be "
                "generated. Record a session first.",
            )

        try:
            return self._interpret_with_llm(report)
        except Exception as error:  # noqa: BLE001 - degrade on *any* failure
            return self._fallback(
                report,
                f"Local LLM unavailable ({type(error).__name__}); showing the "
                "rule-based summary instead. Make sure Ollama is running "
                f"(`ollama serve`) and the '{OLLAMA_MODEL}' model is pulled. "
                "Behavioural signals are indicators, not proof, of deception.",
            )

    # LLM path (Ollama).

    def _interpret_with_llm(self, report):
        """Call the local Ollama model and parse the structured interpretation."""

        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "think": OLLAMA_THINK,  # off by default - see _chat for the retry
            "format": RESPONSE_SCHEMA,  # structured output (JSON schema)
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Here is the session report as JSON:\n\n"
                        + json.dumps(report, indent=2)
                    ),
                },
            ],
        }

        data = self._chat(payload)

        # With a JSON schema in `format` and thinking off, the message content is the JSON string.
        content = data["message"]["content"]
        parsed = json.loads(content)

        return {
            "llm_available": True,
            "model": OLLAMA_MODEL,
            "headline": parsed.get("headline", ""),
            "observations": parsed.get("observations", []),
            "overall_assessment": parsed.get("overall_assessment", ""),
            "confidence_note": parsed.get("confidence_note", ""),
        }

    def _chat(self, payload):
        # POST to Ollama's chat endpoint; if the model rejects the `think` flag (400), retry once without it.

        url = f"{OLLAMA_HOST}/api/chat"

        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)

        if (
            response.status_code == 400
            and "think" in payload
            and "think" in response.text.lower()
        ):
            retry = {k: v for k, v in payload.items() if k != "think"}
            response = requests.post(url, json=retry, timeout=REQUEST_TIMEOUT)

        response.raise_for_status()

        return response.json()

    # Fallback path.

    def _fallback(self, report, note):
        """Rule-based interpretation from the Phase-8 report, no LLM needed."""

        summary = report.get("summary", "No summary available.")
        recommendations = report.get("recommendations", [])

        return {
            "llm_available": False,
            "model": None,
            "headline": summary,
            "observations": recommendations,
            "overall_assessment": summary,
            "confidence_note": note,
        }
