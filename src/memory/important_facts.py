from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from memory.agent_memory import AgentMemorySystem
from autogen_core.memory import MemoryContent, MemoryMimeType

FACT_EXTRACTION_PROMPT = """You are a fact extraction specialist. Extract important information from conversations.

Extract TWO TYPES:

1. USER FACTS (about the person):
   - profile: name, age, location, occupation
   - preferences: likes, dislikes, favorites
   - interests: hobbies, topics of interest
   - skills: knowledge, abilities, expertise
   - goals: objectives, aspirations

2. CONVERSATION CONTEXT (about the discussion):
   - topic: main subjects discussed
   - learning: concepts explained or learned
   - context: important discussion points

RULES:
1. Return ONLY bullet points
2. Format: "- [category] description"
3. Extract BOTH user facts AND conversation topics
4. Skip greetings and small talk
5. If nothing important: "- No important facts to save"

Example:
User: "I'm Shivang. Can you explain how neural networks work?"
Assistant: "Hello Shivang! Neural networks are computing systems inspired by the brain..."

Output:
- [profile] User's name is Shivang
- [topic] User asked about neural networks
- [learning] Explained neural network basics and architecture"""


IMPORTANCE_SCORES = {
    "profile": 10,
    "goals": 9,
    "preferences": 8,
    "interests": 8,
    "skills": 7,
    "topic": 6,
    "learning": 6,
    "context": 5
}


async def create_fact_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="fact_extractor",
        model_client=model_client,
        system_message=FACT_EXTRACTION_PROMPT
    )


async def extract_and_save_facts(
    fact_agent: AssistantAgent,
    user_message: str,
    assistant_response: str,
    memory_system: AgentMemorySystem
) -> dict:
    
    conversation = f"""User: {user_message}
                        Assistant: {assistant_response}
                        Extract important facts from this conversation."""
    try:
        result = await fact_agent.run(task=conversation)
        facts_text = ""
        if result and hasattr(result, 'messages'):
            for msg in result.messages:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if msg.source != 'user':
                        facts_text = msg.content
                        break
        
        if not facts_text or "no important facts" in facts_text.lower():
            return {"user_facts": 0, "conversation_context": 0}
        
        user_facts = 0
        conversation_facts = 0
        
        for line in facts_text.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('-'):
                continue
            
            fact = line[1:].strip()
            category = "general"
            
            if fact.startswith('['):
                end_idx = fact.find(']')
                if end_idx != -1:
                    category = fact[1:end_idx].strip().lower()
                    fact = fact[end_idx + 1:].strip()
            
            importance = IMPORTANCE_SCORES.get(category, 5)
            
            is_user_fact = category in ["profile", "goals", "preferences", "interests", "skills"]
            memory_type = "semantic" if is_user_fact else "episodic"
            
            await memory_system.long_term.add(
                MemoryContent(
                    content=fact,
                    mime_type=MemoryMimeType.TEXT
                ),
                memory_type=memory_type,
                importance=importance
            )
        
            await memory_system.vector.add(
                MemoryContent(
                    content=fact,
                    mime_type=MemoryMimeType.TEXT,
                    metadata={"category": category, "type": memory_type}
                )
            )
            
            if is_user_fact:
                user_facts += 1
            else:
                conversation_facts += 1
        
        return {"user_facts": user_facts, "conversation_context": conversation_facts}
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return {"user_facts": 0, "conversation_context": 0}
