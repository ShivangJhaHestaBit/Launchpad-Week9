import asyncio
from agents import research_agent, summarizer_agent, answer_agent
async def main() -> None:
    
    result = await research_agent.research_agent.run(task="What is quantam computing.")
    research_text = result.messages[-1].content

    summarized = await summarizer_agent.summarizer_agent.run(task = research_text)
    summary_text = summarized.messages[-1].content

    final_answer = await answer_agent.answer_agent.run(task=summary_text)
    answer = final_answer.messages[-1].content

    print(answer)

asyncio.run(main())
