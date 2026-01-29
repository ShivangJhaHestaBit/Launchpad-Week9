from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class AnalystAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="analyst",
            system_message=
                ("""You are the Analyst agent.
                    Your role is to reason about problems, evaluate trade-offs, analyze complexity, performance, scalability, and risks.
                    You verify logic, assumptions, and expected behavior.
                    You do not write production code.
                    Your output should focus on correctness, feasibility, and implications."""
            ),
            model_client=model_client
        )

    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content,source="agent")],cancellation
        )
        return response.chat_message.content