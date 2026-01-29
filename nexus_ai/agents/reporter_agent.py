from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class ReporterAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="reporter",
            system_message=
                ("""You are the Reporter agent.
                    Your task is to compile and present final results in a clear, structured, and user-friendly manner.
                    You summarize decisions, outputs, and reasoning without adding new information.
                    You adapt the explanation level to the intended audience."""
            ),
            model_client=model_client
        )

    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content,source="agent")],cancellation
        )
        return response.chat_message.content