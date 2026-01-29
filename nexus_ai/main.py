# import asyncio
# from autogen_ext.models.openai import OpenAIChatCompletionClient
# from nexus_ai.agents.researcher_agent import ResearcherAgent
# from nexus_ai.agents.analyst_agent import AnalystAgent
# from nexus_ai.agents.critic_agent import CriticAgent
# from nexus_ai.agents.optimizer_agent import OptimizerAgent
# from nexus_ai.agents.validator_agent import ValidatorAgent
# from nexus_ai.agents.reporter_agent import ReporterAgent
# from nexus_ai.agents.planner_agent import PlannerAgent
# from nexus_ai.agents.coder_agent import CoderAgent
# from nexus_ai.agents.orchestrator import OrchestratorAgent
# import os
# from dotenv import load_dotenv
# load_dotenv()

# async def main():
    
#     key = os.getenv("GROQ_API_KEY")
#     model_info = {
#         "family": "oss",
#         "vision": False,
#         "function_calling": True,
#         "json_output": True,
#         "structured_output": True,
#         "context_length": 4096,
#     }

#     model_client = OpenAIChatCompletionClient(
#         model="openai/gpt-oss-120b",
#         api_key=key,
#         base_url="https://api.groq.com/openai/v1",
#         model_info=model_info,
#     )

#     planner = PlannerAgent(model_client)
#     researcher = ResearcherAgent(model_client)
#     analyst = AnalystAgent(model_client)
#     coder = CoderAgent(model_client)
#     critic = CriticAgent(model_client)
#     optimizer = OptimizerAgent(model_client)
#     validator = ValidatorAgent(model_client)
#     reporter = ReporterAgent(model_client)
    
#     agents = {
#         "Researcher": researcher,
#         "Analyst": analyst,
#         "Coder": coder,
#         "Critic": critic,
#         "Optimizer": optimizer,
#         "Validator": validator,
#         "Reporter": reporter
#     }
    
#     orchestrator = OrchestratorAgent(planner, agents)
    
#     task = "Design a RAG pipeline for 10k documents."
    
#     result = await orchestrator.execute(task)
    
#     print("FINAL OUTPUT")
#     print("="*70)
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(main())


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
from nexus_ai.agents.orchestrator import MemoryEnabledOrchestrator
from nexus_ai.memory.agent_memory import AgentMemorySystem
import os
from dotenv import load_dotenv
load_dotenv()

async def main():
    
    print("Multi-Agent System\n")    
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
        model="openai/gpt-oss-20b",
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        model_info=model_info,
    )
    
    memory_system = AgentMemorySystem(
        session_max_turns=50,
        vector_k=5,
        vector_threshold=0.3,
        db_path="nexus_ai/datastore/agent_long_term.db",
        vector_persist_path="nexus_ai/datastore/agent_vectors.faiss"
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
    
    orchestrator = MemoryEnabledOrchestrator(
        planner_agent=planner,
        agents_dict=agents,
        memory_system=memory_system
    )
    
    tasks = [
        "What did we discuss last time.",
    ]
    
    for task in tasks:
        result = await orchestrator.execute(task, use_memory=True)
        
        print("\n" + "="*70)
        print("FINAL OUTPUT")
        print("="*70)
        print(result)
        print("\n" + "="*70)
        
        stats = await orchestrator.get_memory_stats()
        print(f"\nMemory Stats: {stats}\n")
    
    await memory_system.close()


if __name__ == "__main__":
    asyncio.run(main())
