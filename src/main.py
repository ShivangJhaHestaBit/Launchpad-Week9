'''
Day 1

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

'''

'''
Day 2

import asyncio
from autogen_ext.models.llama_cpp import LlamaCppChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from orchestrator.planner_agent import Planner
from agents.answer_agent import answer_agent

async def main():
    model_client = LlamaCppChatCompletionClient(
        model_path="src/models/qwen2.5-3b-instruct-q4_0.gguf",
        verbose=False,
        temperature =0.7,
        n_ctx=4096,
        max_tokens = 256
    )
    
    planner = Planner(model_client=model_client)
    query = (
        "Explain how retrieval augmented generation (RAG) works, "
        "including ingestion, indexing, and inference."
    )

    final_answer, execution_tree = await planner.run(query)
    cancellation = CancellationToken()
    result = await answer_agent.on_messages(
        [TextMessage(content=final_answer,source="reflector")],
        cancellation
    )
    print("\nFINAL ANSWER\n")
    print(result.chat_message.content)

    print("\nEXECUTION TREE\n")
    for node_id, data in execution_tree.items():
        print(f"Node: {node_id}")
        print(f"  Deps  : {data['deps']}")
        print(f"  Output:\n{data['output']}\n")

if __name__ == "__main__":
    asyncio.run(main())

'''

from orchestrator import run_orchestration,summarize_results
from agents.answer_agent import answer_agent
import asyncio

async def main():
    user_query = "Analyze sales.csv and generate top 5 insights"
    context = await run_orchestration(user_query)
    final_summary = summarize_results(context)
    task = f"You have to reply to user query: {user_query}, based on the context available below: \n{final_summary}"
    result = await answer_agent.run(task=task)
    print("=== Final Agent Outputs ===")
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())