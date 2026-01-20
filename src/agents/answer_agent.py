from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.llama_cpp import LlamaCppChatCompletionClient
from autogen_agentchat.tools import AgentTool

llama_model = LlamaCppChatCompletionClient(
        model_path="src/models/qwen2.5-3b-instruct-q4_0.gguf",
        verbose=False,
        temperature =0.7,
        n_ctx=4096,
        max_tokens = 256
    )
answer_agent = AssistantAgent(
    name = "AnswerAgent",
    model_client=llama_model,
    description="A final-response agent that converts summarized information into a clear, coherent, and user-ready answer. It focuses on correctness, clarity, and direct relevance to the user’s question.",
    system_message=("""
        You are a final-answer AI agent.
        Your role is to generate a clear, accurate, and complete response using the provided summarized information.
        Rules:
        - Use only the given summary as your source of truth
        - Do not add new facts, assumptions, or external knowledge
        - Answer the user’s question directly and clearly
        - If the summary is insufficient, explicitly state the limitation
        Output guidelines:
        - Be concise, well-structured, and easy to understand
        - Use bullet points or short paragraphs when helpful
        - Maintain a neutral and professional tone
        - Avoid unnecessary repetition or filler
        Do not reveal internal reasoning.
        Your goal is to deliver the best possible final answer based on the summary."""
    )
)

answer_agent_tool = AgentTool(answer_agent)