# System Architecture

## Overview

Multi-agent orchestration system with persistent memory, built on AutoGen framework.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR LAYER                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Memory Context Builder                              │   │
│  │  - Retrieves important facts                         │   │
│  │  - Queries vector store                              │   │
│  │  - Fetches recent session                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      PLANNING LAYER                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PlannerAgent                                        │   │
│  │  - Decomposes task into steps                        │   │
│  │  - Assigns steps to agents                           │   │
│  │  - Considers memory context                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                           │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │Researcher│ Analyst  │  Coder   │ Critic   │Optimizer│   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
│  ┌──────────┬──────────┐                                     │
│  │Validator │ Reporter │                                     │
│  └──────────┴──────────┘                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY LAYER                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Session Memory (RAM)                                │   │
│  │  - Last 50 conversation turns                        │   │
│  │  - Temporary context                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Vector Store (FAISS)                                │   │
│  │  - Semantic embeddings                               │   │
│  │  - Similarity search                                 │   │
│  │  - 384-dim vectors                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Long-term DB (SQLite)                               │   │
│  │  - Persistent storage                                │   │
│  │  - Importance scoring                                │   │
│  │  - Type classification                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Orchestrator

**File**: `agents/orchestrator.py`

**Responsibilities**:
- Coordinate workflow execution
- Build memory context for each step
- Manage agent communication
- Save results to memory

**Key Methods**:
```python
execute(user_goal, use_memory=True)
_build_comprehensive_memory_context(query)
_build_agent_context(task, previous_results, ...)
_save_to_memory(content, importance, memory_type)
```

### 2. Planner Agent

**File**: `agents/planner.py`

**Purpose**: Task decomposition

**Input**: User goal + Memory context

**Output**: JSON execution plan
```json
{
  "steps": [
    {"agent": "Researcher", "task": "Research topic X"},
    {"agent": "Coder", "task": "Write code for Y"},
    {"agent": "Reporter", "task": "Compile results"}
  ]
}
```

### 3. Execution Agents

| Agent | Role | Output |
|-------|------|--------|
| **Researcher** | Information gathering | Research findings |
| **Analyst** | Data analysis | Insights, patterns |
| **Coder** | Code generation | Working code |
| **Critic** | Quality review | Issues, improvements |
| **Optimizer** | Performance tuning | Optimizations |
| **Validator** | Correctness check | Validation report |
| **Reporter** | Result compilation | Final report |

### 4. Memory System

**File**: `memory/agent_memory.py`

**Components**:

#### Session Memory
- **Storage**: In-memory list
- **Capacity**: 50 turns (configurable)
- **Use**: Recent conversation context
- **Persistence**: No (clears on restart)

#### Vector Store
- **Backend**: FAISS
- **Model**: all-MiniLM-L6-v2 (384-dim)
- **Similarity**: Cosine (L2 normalized)
- **Use**: Semantic search for relevant memories
- **Persistence**: Yes (saved to disk)

#### Long-term Memory
- **Backend**: SQLite
- **Schema**:
  ```sql
  CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    content TEXT,
    memory_type TEXT,      -- 'semantic' or 'episodic'
    importance INTEGER,     -- 0-10 score
    created_at TIMESTAMP
  )
  ```
- **Use**: Persistent facts and important information
- **Persistence**: Yes

## Data Flow

### Task Execution Flow

```
1. User Input
   ↓
2. Orchestrator receives task
   ↓
3. Query memory for relevant context
   - Long-term: get_important_memories(importance >= 7)
   - Vector: query(user_goal)
   - Session: get_recent(n=3)
   ↓
4. Build memory context (formatted sections)
   ↓
5. Planner creates execution plan
   ↓
6. For each step:
   a. Build agent context (memory + task + previous)
   b. Execute agent
   c. Save result to memory (importance 5-6)
   ↓
7. Compile final result
   ↓
8. Save to long-term (importance 7)
   ↓
9. Return to user
```

### Memory Save Flow

```
Content → MemoryContent object
           ↓
        AgentMemorySystem.add()
           ↓
    ┌──────┴──────┬──────────┐
    ↓             ↓          ↓
Session.add()  Vector.add()  LongTerm.add()
    │             │              │
    ↓             ↓              ↓
  RAM List    FAISS Index    SQLite DB
```

### Memory Retrieval Flow

```
Query String
    ↓
AgentMemorySystem.get_context_for_query()
    ↓
┌───────┴────────┬──────────────┐
↓                ↓              ↓
Session       Vector        Long-term
get_recent()  query()       get_important()
    ↓            ↓              ↓
Combined & Deduplicated
    ↓
Formatted Context
    ↓
Agent Prompt
```

## Memory Context Format

Agents receive context in structured sections:

```
=== IMPORTANT INFORMATION ===
  • User's name is Shivang
  • Shivang prefers Python

=== RELEVANT PAST CONTEXT ===
  • Previously discussed RAG systems
  • Built a web scraper last week

=== RECENT CONVERSATION ===
  • User asked about neural networks
  • Explained backpropagation

=== ORIGINAL GOAL ===
Build a sentiment analysis tool

=== YOUR TASK ===
Research sentiment analysis libraries

=== PREVIOUS STEPS ===
  • Researcher: Found NLTK, TextBlob, VADER...
```
