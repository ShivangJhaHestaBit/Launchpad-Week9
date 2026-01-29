import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from nexus_ai.agents.researcher_agent import ResearcherAgent
from nexus_ai.agents.analyst_agent import AnalystAgent
from nexus_ai.agents.critic_agent import CriticAgent
from nexus_ai.agents.optimizer_agent import OptimizerAgent
from nexus_ai.agents.validator_agent import ValidatorAgent
from nexus_ai.agents.reporter_agent import ReporterAgent
from nexus_ai.agents.planner_agent import PlannerAgent
from nexus_ai.agents.coder_agent import CoderAgent
from nexus_ai.agents.orchestrator import OrchestratorAgent
import os
from dotenv import load_dotenv
load_dotenv()

async def main():
    
    key = os.getenv("GROQ_API_KEY")
    model_info = {
        "family": "oss",
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "context_length": 4096,
    }

    model_client = OpenAIChatCompletionClient(
        model="openai/gpt-oss-120b",
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        model_info=model_info,
    )

    planner = PlannerAgent(model_client)
    researcher = ResearcherAgent(model_client)
    analyst = AnalystAgent(model_client)
    coder = CoderAgent(model_client)
    critic = CriticAgent(model_client)
    optimizer = OptimizerAgent(model_client)
    validator = ValidatorAgent(model_client)
    reporter = ReporterAgent(model_client)
    
    agents = {
        "Researcher": researcher,
        "Analyst": analyst,
        "Coder": coder,
        "Critic": critic,
        "Optimizer": optimizer,
        "Validator": validator,
        "Reporter": reporter
    }
    
    orchestrator = OrchestratorAgent(planner, agents)
    
    task = "Design a RAG pipeline for 10k documents."
    
    result = await orchestrator.execute(task)
    
    print("FINAL OUTPUT")
    print("="*70)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
