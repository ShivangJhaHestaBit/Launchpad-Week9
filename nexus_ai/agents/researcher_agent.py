from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class ResearcherAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="researcher",
            system_message=
                ("""You are the Researcher agent.
                    Your role is to gather accurate, relevant, and up-to-date information needed to complete assigned tasks.
                    You focus on external knowledge, documentation, best practices, algorithms, libraries, APIs, and prior art.
                    You do not write code or make final decisions.
                    You present findings clearly, with assumptions and limitations noted."""
            ),
            model_client=model_client
        )

    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content,source="agent")],cancellation
        )
        return response.chat_message.content