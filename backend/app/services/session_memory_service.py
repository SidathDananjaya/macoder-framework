from ai_engine.temporal.session_memory import (
    SessionMemory
)

memory = SessionMemory()


async def update_memory(result):
    return memory.update(result)


async def reset_memory():
    memory.reset()
