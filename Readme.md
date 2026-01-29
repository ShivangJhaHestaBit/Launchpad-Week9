# NEXUS AI

A production-ready orchestration system that coordinates specialized AI agents with persistent memory capabilities.

## Features

- **7 Specialized Agents**: Researcher, Analyst, Coder, Critic, Optimizer, Validator, Reporter
- **Intelligent Planning**: Auto-decomposes complex tasks into executable steps
- **Persistent Memory**: Remembers user preferences, past conversations, and learned facts
- **Context-Aware**: Uses semantic search to retrieve relevant information


## Project Structure

```
nexus_ai/
├── agents/
│   ├── planner.py          # Task decomposition
│   ├── researcher.py       # Information gathering
│   ├── analyst.py          # Data analysis
│   ├── coder.py            # Code generation
│   ├── critic.py           # Quality review
│   ├── optimizer.py        # Performance optimization
│   ├── validator.py        # Result validation
│   ├── reporter.py         # Report compilation
│   └── orchestrator.py     # Agent coordination
├── memory/
│   ├── agent_memory.py     # Memory system
│   ├── long_term.py        # SQLite persistence
│   ├── vector_store.py     # FAISS semantic search
│   └── session_memory.py   # Short-term memory
├── main_final.py           # Entry point
└── requirements.txt        # Dependencies
```

## Usage

### Basic Task Execution

```python
from orchestrator_final import MemoryEnabledOrchestrator
from memory.agent_memory import AgentMemorySystem

# Initialize memory
memory = AgentMemorySystem(vector_threshold=0.2)

# Initialize orchestrator
orchestrator = MemoryEnabledOrchestrator(planner, agents, memory)

# Execute task
result = await orchestrator.execute("Your task here")
```

### Save Important Facts

```python
await orchestrator.save_important_fact(
    "User prefers Python for development",
    importance=9
)
```

### Memory Statistics

```python
stats = await orchestrator.get_memory_stats()
print(stats)
```

## How It Works

1. **User submits task** → "Build a web scraper for product prices"
2. **Planner decomposes** → Research → Code → Validate → Report
3. **Memory retrieves** → Relevant past work, user preferences
4. **Agents execute** → Each step with full context
5. **Results saved** → Long-term memory for future use

## Configuration

### Memory Settings

```python
memory = AgentMemorySystem(
    session_max_turns=50,      # Recent conversation history
    vector_k=5,                # Top-k similar memories
    vector_threshold=0.2,      # Similarity threshold (0-1)
    db_path="long_term.db",    # SQLite database path
)
```

## Requirements

- Python 3.12+
- autogen-agentchat
- autogen-ext[openai]
- faiss-cpu
- sentence-transformers

## Memory Layers

1. **Session Memory**: Last 50 conversation turns (RAM)
2. **Vector Store**: Semantic similarity search (FAISS)
3. **Long-term DB**: Persistent facts and learnings (SQLite)

## Example Tasks

```python
# Research and analysis
await orchestrator.execute("Research RAG systems and create a report")

# Code generation
await orchestrator.execute("Build a sentiment analysis script")

# Multi-step workflow
await orchestrator.execute("Create a web app, optimize it, and document")

# Memory-aware
await orchestrator.execute("What did we discuss last time?")
```

