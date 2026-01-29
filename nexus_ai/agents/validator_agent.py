from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

class ValidatorAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="validator",
            system_message=
                ("""You are the Validator agent.
                    Your responsibility is to verify that outputs meet all stated requirements, constraints, and acceptance criteria.
                    You check correctness, completeness, edge cases, and consistency across components.
                    You do not generate new content beyond validation feedback and pass/fail judgments."""
            ),
            model_client=model_client
        )

    async def run(self, content):
        cancellation = CancellationToken()
        response = await self.agent.on_messages(
            [TextMessage(content=content,source="agent")],cancellation
        )
        return response.chat_message.content