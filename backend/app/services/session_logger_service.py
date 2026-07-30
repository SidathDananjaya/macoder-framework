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
    logger.reset()


async def log_session(result):
    logger.add_record(result)


async def save_session():
    return logger.export_csv()


async def save_session_json():
    return logger.export_json()


async def session_status():
    return logger.status()


async def session_report():
    return report_generator.generate(logger.snapshot())


def report_from_file(path):
    return report_generator.generate(path)


async def session_interpretation():
    report = report_generator.generate(logger.snapshot())

    return interpreter.interpret(report)


def interpret_from_file(path):
    report = report_generator.generate(path)

    return interpreter.interpret(report)
