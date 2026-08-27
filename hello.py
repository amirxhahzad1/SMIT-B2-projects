from agents import Agent, Runner

import asyncio

agent = Agent(
    name="History Tutor",
    instructions="You answer history questions clearly and concisely.",
)

async def main():
    result = await Runner.run(agent, "When did the Roman Empire fall?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())