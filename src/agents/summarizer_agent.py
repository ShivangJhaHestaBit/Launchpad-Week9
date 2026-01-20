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
summarizer_agent = AssistantAgent(
    name = "SummarizerAgent",
    model_client=llama_model,
    description="A focused assistant that condenses input content into clear, accurate summaries while preserving key information, intent, and context.",
    system_message=(
        """You are a summarization-focused AI agent.
            Your role is to condense provided content into a clear, accurate summary.
            Rules:
            - Preserve key facts, intent, and conclusions
            - Do not introduce new information or interpretations
            - Do not omit critical details
            - Maintain neutral tone and factual accuracy
            Output guidelines:
            - Be concise and well-structured
            - Use bullet points or short paragraphs when appropriate
            - Match the desired length or format if specified
            - Avoid redundancy and unnecessary wording
            Do not add opinions or external knowledge.
            Your goal is clarity, brevity, and fidelity to the source."""
    )
)

summarizer_agent_tool = AgentTool(summarizer_agent)