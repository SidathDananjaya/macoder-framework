"""Phase 7 - Session Recording service layer.

Thin async wrappers around the shared :class:`SessionLogger` singleton so the
websocket pipeline and the REST routes operate on the same live recording.
"""

from ai_engine.analytics.session_logger import (
    SessionLogger
)
from ai_engine.analytics.session_report_generator import (
    SessionReportGenerator
)
from ai_engine.analytics.llm_interpreter import (
    LLMInterpreter
)

logger = SessionLogger()
report_generator = SessionReportGenerator()
interpreter = LLMInterpreter()


async def reset_session():
    """Start a fresh recording (called when a live session connects)."""

    logger.reset()


async def log_session(result):
    """Record one processed frame."""

    logger.add_record(result)


async def save_session():
    """Persist the flat CSV export. Returns the file path (or None)."""

    return logger.export_csv()


async def save_session_json():
    """Persist the structured JSON recording. Returns the file path (or None)."""

    return logger.export_json()


async def session_status():
    """Return a snapshot of the current recording."""

    return logger.status()


async def session_report():
    """Phase 8 - generate the interview report for the live recording."""

    return report_generator.generate(logger.snapshot())


def report_from_file(path):
    """Phase 8 - generate the interview report for a stored JSON recording."""

    return report_generator.generate(path)


async def session_interpretation():
    """Phase 9 - LLM interpretation of the live recording."""

    report = report_generator.generate(logger.snapshot())

    return interpreter.interpret(report)


def interpret_from_file(path):
    """Phase 9 - LLM interpretation of a stored JSON recording."""

    report = report_generator.generate(path)

    return interpreter.interpret(report)
