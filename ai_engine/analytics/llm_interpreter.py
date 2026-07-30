import json
import os
import requests


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

NUM_PREDICT = 700
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 600
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "false").lower() == "true"

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
    def interpret(self, report):

        report = report or {}

        if not report.get("frame_count"):
            return self._fallback(
                report,
                "No session data was recorded, so no interpretation could be "
                "generated. Record a session first.",
            )

        try:
            return self._interpret_with_llm(report)
        except Exception as error: 
            return self._fallback(
                report,
                f"Local LLM unavailable ({type(error).__name__}); showing the "
                "rule-based summary instead. Make sure Ollama is running "
                f"(`ollama serve`) and the '{OLLAMA_MODEL}' model is pulled. "
                "Behavioural signals are indicators, not proof, of deception.",
            )

    def _interpret_with_llm(self, report):
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "think": OLLAMA_THINK,  
            "format": RESPONSE_SCHEMA,  
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

    def _fallback(self, report, note):
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
