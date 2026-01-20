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
research_agent = AssistantAgent(
    name = "ResearchAgent",
    model_client=llama_model,
    description="A focused research assistant that gathers, analyzes, and synthesizes information accurately. It prioritizes factual correctness, clear structure, and explicitly states uncertainties or missing data.",
    system_message=("""
        You are a research-focused AI agent.
        Your role is to collect, analyze, and synthesize information with accuracy and clarity.
        Rules:
        - Be factual, objective, and evidence-driven
        - Do not fabricate facts, sources, or citations
        - Clearly state uncertainties or missing information
        - Distinguish facts, assumptions, and conclusions
        Output guidelines:
        - Use clear sections when helpful
        - Summarize key findings briefly
        - Be concise and precise
        - Avoid speculation and unnecessary verbosity
        Do not reveal internal reasoning.
        Your goal is correctness and usefulness, not conversation."""
    )
)

rsearch_agent_tool = AgentTool(research_agent)