from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class PlannerAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="planner",
            system_message=
                ("""You are the Planner agent.
                    Responsibilities:
                    - Decompose the user's goal into a clear, ordered execution plan.
                    - Assign each step to ONE of the following agents only:
                    Researcher, Analyst, Coder, Critic, Optimizer, Validator, Reporter
                    - Do NOT solve the task.
                    - Do NOT add explanations.

                    Output format (STRICT JSON):
                    {
                    "steps": [
                            {"agent": "<agent_name>", "task": "<specific task>"}
                        ]
                    }"""
            ),
            model_client=model_client
        )
        
    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content,source="agent")],cancellation
        )
        return response.chat_message.content
