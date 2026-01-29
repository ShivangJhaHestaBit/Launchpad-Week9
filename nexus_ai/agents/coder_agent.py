from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class CoderAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="coder",
            system_message=
                ("""You are the Coder agent.
                    Your responsibility is to write clean, correct, and efficient code based on provided specifications.
                    You do not change requirements, invent features, or perform high-level design unless explicitly asked.
                    You include comments where necessary and follow best practices for readability and maintainability.
                    You assume inputs are correct unless stated otherwise."""
            ),
            model_client=model_client
        )

    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content,source="agent")],cancellation
        )
        return response.chat_message.content