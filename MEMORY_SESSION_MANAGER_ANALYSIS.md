# Understanding MemorySessionManager - AgentCore Memory SDK

## Overview

This is AWS Bedrock's **official high-level SDK** for AgentCore Memory. It provides a conversation-centric API on top of the low-level boto3 AgentCore operations.

---

## Architecture Comparison

### Your Implementation vs MemorySessionManager

| Aspect | Your Code | MemorySessionManager |
|--------|-----------|---------------------|
| **Purpose** | LangGraph agent with memory | Standalone conversation manager |
| **Memory Layer** | `AgentCoreMemoryStore` (direct) | Wraps boto3 bedrock-agentcore client |
| **Namespace Pattern** | Tuples: `("preferences", actor_id)` | Strings: `"preferences/{actorId}"` |
| **Integration** | LangGraph StateGraph + Checkpointer | LLM callback pattern |
| **Short-term Storage** | AgentCoreMemorySaver (checkpointing) | Events API (conversation turns) |
| **Long-term Storage** | AgentCoreMemoryStore.search() | retrieve_memory_records API |

---

## Key Classes

### 1. MemorySessionManager (Main Class)

**Purpose:** High-level interface for managing conversational sessions and memory operations.

**Responsibilities:**
```python
# Short-term memory (Events)
manager.add_turns(actor_id, session_id, messages)
manager.get_last_k_turns(actor_id, session_id, k=5)
manager.list_events(actor_id, session_id)

# Long-term memory (Memory Records)
manager.search_long_term_memories(query, namespace_prefix, top_k=3)
manager.list_long_term_memory_records(namespace_prefix)

# LLM integration
manager.process_turn_with_llm(actor_id, session_id, user_input, llm_callback)

# Actor/session management
manager.list_actors()
manager.list_actor_sessions(actor_id)
```

### 2. MemorySession (Session Scope)

**Purpose:** Scoped to a specific actor + session, delegates to MemorySessionManager.

```python
session = manager.create_memory_session(actor_id, session_id)

# All operations are scoped to this actor/session
session.add_turns(messages)
session.get_last_k_turns(k=5)
session.process_turn_with_llm(user_input, llm_callback)
```

### 3. Actor (Actor Scope)

**Purpose:** Represents an actor (user), can list their sessions.

```python
actor = session.get_actor()
sessions = actor.list_sessions()
```

---

## Core Concepts

### Short-Term Memory: Events

**Events** are conversation turns stored chronologically.

```python
# Structure
Event {
    eventId: str
    actorId: str
    sessionId: str
    eventTimestamp: datetime
    payload: [
        {
            "conversational": {
                "role": "user",
                "content": {"text": "Hello"}
            }
        },
        {
            "conversational": {
                "role": "assistant",
                "content": {"text": "Hi there!"}
            }
        }
    ]
    branch: Optional[{name, rootEventId}]
    metadata: Optional[Dict]
}
```

**Operations:**
- `add_turns()` - Create event with messages
- `list_events()` - Retrieve events (with pagination)
- `get_last_k_turns()` - Get recent conversation history
- `fork_conversation()` - Branch conversations

### Long-Term Memory: Memory Records

**Memory Records** are semantic memories indexed by namespace.

```python
# Structure
MemoryRecord {
    memoryRecordId: str
    namespace: str
    content: {
        "text": str
    }
    relevanceScore: float  # For search results
    metadata: Optional[Dict]
}
```

**Operations:**
- `search_long_term_memories()` - Semantic search
- `list_long_term_memory_records()` - List all in namespace
- `delete_all_long_term_memories_in_namespace()` - Bulk delete

---

## Key Patterns

### 1. LLM Callback Pattern

```python
def my_llm(user_input: str, memories: List[Dict]) -> str:
    """Your LLM logic here."""
    # Format context from memories
    context = "\n".join([m.get('content', {}).get('text', '') for m in memories])

    # Call your LLM
    response = bedrock.invoke_model(...)
    return response['content']

# Process turn: retrieve → LLM → save
memories, response, event = manager.process_turn_with_llm(
    actor_id="user-123",
    session_id="session-456",
    user_input="What did we discuss?",
    llm_callback=my_llm,
    retrieval_config={
        "support/facts/{sessionId}": RetrievalConfig(top_k=5, relevance_score=0.3)
    }
)
```

**What it does:**
1. Retrieves relevant memories from namespaces
2. Calls your LLM callback with (user_input, memories)
3. Saves the conversation turn as an Event

### 2. Namespace Templates

```python
# Namespace with template variables
namespace = "support/facts/{sessionId}"

# Automatically resolved at runtime
retrieval_config = {
    "support/facts/{sessionId}": RetrievalConfig(top_k=5),
    "user/preferences/{actorId}": RetrievalConfig(top_k=3)
}
```

**Template Variables:**
- `{actorId}` - Resolved to actor_id
- `{sessionId}` - Resolved to session_id
- `{strategyId}` - Resolved to strategy_id (if provided)

### 3. Conversation Branching

```python
# Fork conversation from a specific event
manager.fork_conversation(
    actor_id="user-123",
    session_id="session-456",
    root_event_id="event-789",
    branch_name="alternative-response",
    messages=[
        ConversationalMessage("Let me try a different approach", MessageRole.ASSISTANT)
    ]
)

# List all branches
branches = manager.list_branches(actor_id, session_id)
# Returns: [
#   Branch(name="main", eventCount=10),
#   Branch(name="alternative-response", rootEventId="event-789", eventCount=3)
# ]

# Get events from specific branch
branch_events = manager.list_events(
    actor_id, session_id,
    branch_name="alternative-response",
    include_parent_branches=True  # Include events from parent branches
)
```

### 4. Dynamic Method Forwarding

```python
class MemorySessionManager:
    def __getattr__(self, name: str):
        """Forward unknown methods to boto3 client."""
        if name in self._ALLOWED_DATA_PLANE_METHODS:
            return getattr(self._data_plane_client, name)
        raise AttributeError(...)

# Direct boto3 access
manager.retrieve_memory_records(...)  # Forwarded to boto3 client
manager.batch_create_memory_records(...)
```

---

## How It Relates to Your Code

### Your Implementation (LangGraph + AgentCore)

```python
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

# Short-term: Checkpointer (LangGraph state)
self.checkpointer = AgentCoreMemorySaver(memory_id, region_name)

# Long-term: Direct store access
self.memory_store = AgentCoreMemoryStore(memory_id, region_name)

# Explicit hooks
async def _pre_model_memory(self, state, user_query):
    # Save user message
    await self.memory_store.put(
        ("conversation", actor_id, thread_id),
        str(uuid.uuid4()),
        {"role": "user", "content": user_query}
    )
    # Retrieve memories
    results = await self.memory_store.search(
        ("preferences", actor_id),
        query=user_query,
        limit=3
    )
```

### MemorySessionManager Approach

```python
manager = MemorySessionManager(memory_id, region_name)

# Short-term: Events API
manager.add_turns(
    actor_id, session_id,
    messages=[
        ConversationalMessage(user_query, MessageRole.USER),
        ConversationalMessage(response, MessageRole.ASSISTANT)
    ]
)

# Long-term: Wrapped retrieve API
memories = manager.search_long_term_memories(
    query=user_query,
    namespace_prefix="preferences/{actorId}",  # String template
    top_k=3
)

# LLM callback pattern
memories, response, event = manager.process_turn_with_llm(
    actor_id, session_id, user_query,
    llm_callback=my_llm,
    retrieval_config={...}
)
```

---

## Key Differences

### 1. Namespace Strategy

**Your Code (Better for Direct Control):**
```python
# Tuple-based (type-safe)
conversation_ns = ("conversation", actor_id, thread_id)
preferences_ns = ("preferences", actor_id)

await self.memory_store.put(conversation_ns, key, value)
results = await self.memory_store.search(preferences_ns, query)
```

**MemorySessionManager:**
```python
# String-based templates
namespace = "preferences/{actorId}"
# Resolved at runtime: "preferences/user-123"

memories = manager.search_long_term_memories(
    query,
    namespace_prefix=namespace,  # Template resolved internally
    top_k=3
)
```

### 2. Short-Term Storage

**Your Code:**
- Uses `AgentCoreMemorySaver` (checkpointer)
- Integrates with LangGraph's StateGraph
- Automatic state persistence after each node
- State stored in checkpoints, not Events

**MemorySessionManager:**
- Uses Events API directly
- Stores conversation turns as Events
- No LangGraph integration
- Manual conversation management

### 3. Integration Pattern

**Your Code (LangGraph-native):**
```python
# Workflow node with explicit hooks
async def _process_query_node(self, state):
    # Pre-hook: save + retrieve
    memory_context = await self._pre_model_memory(state, user_query)

    # LLM call
    response = await self.llm.ainvoke(messages)

    # Post-hook: save response
    await self._post_model_memory(state, answer)
    return state
```

**MemorySessionManager (Callback-based):**
```python
# All-in-one callback pattern
def my_llm(user_input, memories):
    # Your LLM logic
    return response

memories, response, event = manager.process_turn_with_llm(
    actor_id, session_id, user_input,
    llm_callback=my_llm,
    retrieval_config={...}
)
```

---

## Should You Use MemorySessionManager?

### Use MemorySessionManager If:
- ✅ Building standalone conversation system (not LangGraph)
- ✅ Want built-in conversation turn management
- ✅ Need conversation branching features
- ✅ Prefer callback pattern over explicit hooks
- ✅ Want automatic Event storage for all turns

### Keep Your Implementation If:
- ✅ Using LangGraph StateGraph workflows
- ✅ Need tight integration with checkpointing
- ✅ Prefer explicit control over memory operations
- ✅ Like tuple-based namespaces
- ✅ Want pre/post hooks pattern
- ✅ Don't need conversation branching

---

## Could You Integrate Both?

**Yes**, you could use MemorySessionManager for specific operations:

```python
class AskBillAgent(BaseAgent):
    def __init__(self, config):
        # Your current setup
        self.checkpointer = AgentCoreMemorySaver(memory_id, region_name)
        self.memory_store = AgentCoreMemoryStore(memory_id, region_name)

        # Add MemorySessionManager for advanced features
        self.session_manager = MemorySessionManager(memory_id, region_name)

    async def _pre_model_memory(self, state, user_query):
        """Your current explicit hook."""
        # Save using your tuple approach
        await self.memory_store.put(
            ("conversation", actor_id, thread_id),
            str(uuid.uuid4()),
            {"role": "user", "content": user_query}
        )

        # Could also use session_manager for advanced features
        # like conversation branching if needed
        # branches = self.session_manager.list_branches(actor_id, session_id)

        # Retrieve using your approach
        results = await self.memory_store.search(...)
        return formatted_context
```

---

## Recommendations

### For Your Use Case (LangGraph Agent):

**Keep your current implementation** because:
1. ✅ **Better LangGraph integration** - Your explicit hooks work perfectly with StateGraph
2. ✅ **Cleaner namespaces** - Tuple-based is more type-safe
3. ✅ **Direct control** - You control exactly when memory operations happen
4. ✅ **Checkpointing** - AgentCoreMemorySaver handles state persistence
5. ✅ **Simpler** - No extra abstraction layer

**MemorySessionManager would add:**
- ⚠️ Extra complexity (another abstraction)
- ⚠️ String template namespaces (less type-safe)
- ⚠️ Callback pattern (less flexible than your hooks)
- ✅ Conversation branching (if you need it)
- ✅ Built-in turn grouping (if you need it)

### When to Use MemorySessionManager:

1. **Conversation branching** - If you need to explore alternative conversation paths
2. **Standalone system** - If not using LangGraph
3. **Quick prototyping** - Built-in LLM callback pattern is convenient
4. **Event-centric** - If you need detailed event tracking vs checkpoints

---

## Summary

**MemorySessionManager** is AWS's high-level SDK for conversation management with AgentCore Memory. It provides:
- Conversation turn management (Events API)
- LLM callback integration
- Conversation branching
- String-template namespaces

**Your implementation** is architecturally superior for LangGraph agents because:
- Explicit pre/post memory hooks (cleaner separation)
- Tuple-based namespaces (more robust)
- Direct AgentCoreMemoryStore integration
- Better checkpointing with AgentCoreMemorySaver

**Bottom line:** Your approach is the right choice for LangGraph agents. MemorySessionManager is better suited for standalone conversation systems that don't use LangGraph's workflow patterns.
