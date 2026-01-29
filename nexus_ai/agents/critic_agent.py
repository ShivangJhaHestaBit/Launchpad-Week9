from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class CriticAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="critic",
            system_message=
                ("""You are the Critic agent.
                    Your task is to review outputs from other agents and identify flaws, ambiguities, inefficiencies, or missing elements.
                    You provide constructive, actionable feedback.
                    You do not propose full solutions unless explicitly requested.
                    You prioritize correctness, clarity, and robustness."""
            ),
            model_client=model_client
        )

    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content,source="agent")],cancellation
        )
        return response.chat_message.content