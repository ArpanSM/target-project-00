import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner

from rag_service import orchestration_agent

async def main():
    result = Runner.run_streamed(orchestration_agent, input="tell me about target?")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())