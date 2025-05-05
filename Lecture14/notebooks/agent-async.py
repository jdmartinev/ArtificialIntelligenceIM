#pip install openai-agents
import asyncio
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

async def main():
    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)

# Ejecutar el programa de forma correcta
asyncio.run(main())

