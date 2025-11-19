"""Refactored Ask Bill Agent with Checkpointer-Optimized Workflow.

This version uses:
- **ChatBedrockConverse** - Modern Converse API with extended features
- **AgentCoreMemorySaver** - LangGraph checkpointing for state persistence
- **Minimal nodes** - 4 meaningful nodes optimized for checkpointing
- **Shared LLM instance** - One ChatBedrockConverse for all operations
- **Minimal state** - Only essential fields in AskBillAgentState
- **Clear checkpoint boundaries** - Each node represents expensive/resumable operation
- **LLM-driven logic** - Intent handled by Claude, not rules
- **Future-ready** - Designed for LangGraph checkpointer integration
"""

# ruff: noqa: UP040
from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph_checkpoint_aws import AgentCoreMemorySaver

# Import bill PDF structuring node functions
from ask_bill.agents.ask_bill.bill_pdf_structurer import (
    extract_bill_text,
    structure_bill_data,
)
from ask_bill.agents.ask_bill.prompts import build_system_prompt
from ask_bill.agents.base.agent import BaseAgent
from ask_bill.core.circuit_breaker import CircuitBreaker
from ask_bill.core.exceptions.agent import AgentInvocationError, WorkflowError
from ask_bill.core.memory.hooks import AskBillMemoryHooks
from ask_bill.core.models.state import AskBillAgentState, WorkflowAction, WorkflowNode
from ask_bill.core.validation import InputValidator, ValidationError

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ask_bill.core.config.bedrock import AskBillConfig


logger = logging.getLogger(__name__)


# LLM content may be a raw string or a list of parts (strings or dicts)
StrOrListStrOrDict: TypeAlias = str | list[str | dict[str, Any]]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _as_text(value: StrOrListStrOrDict) -> str:
    """Convert `str | list[str | dict[str, Any]]` into a deterministic string.

    Args:
        value: Content from LLM (string or list of strings/dicts)

    Returns:
        Normalized text string
    """
    if isinstance(value, str):
        return value

    return " ".join(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True) for item in value)


def _message_signature(message: BaseMessage) -> tuple[str, str | None, str | None]:
    """Generate unique signature for message deduplication.

    Args:
        message: LangChain message object

    Returns:
        Tuple of (class_name, content_repr, tool_calls_repr)
    """
    content = getattr(message, "content", None)
    content_repr: str | None

    if isinstance(content, list | dict):
        try:
            content_repr = json.dumps(content, sort_keys=True)
        except TypeError:
            content_repr = repr(content)
    else:
        content_repr = repr(content) if content is not None else None

    tool_calls = getattr(message, "tool_calls", None)
    tool_repr: str | None = None
    if tool_calls is not None:
        try:
            tool_repr = json.dumps(tool_calls, sort_keys=True)
        except TypeError:
            tool_repr = repr(tool_calls)

    return (message.__class__.__name__, content_repr, tool_repr)


# ============================================================================
# MEMORY MANAGER
# ============================================================================


class MemoryManager:
    """Manages memory operations for the AskBill agent."""

    def __init__(self, memory_hooks: AskBillMemoryHooks | None):
        """Initialize memory manager.

        Args:
            memory_hooks: Optional memory hooks instance
        """
        self.memory_hooks = memory_hooks
        self._sessions: dict[str, Any] = {}

    def is_enabled(self) -> bool:
        """Check if memory is enabled."""
        return self.memory_hooks is not None

    def get_session(self, user_id: str, session_id: str) -> Any:  # noqa: ANN401
        """Get or create a memory session."""
        if not self.memory_hooks:
            raise RuntimeError("Memory not configured")

        key = f"{user_id}:{session_id}"
        if key not in self._sessions:
            self._sessions[key] = self.memory_hooks.start_session(user_id, session_id)
            logger.info(f"Created new session: {key}")

        return self._sessions[key]

    def get_short_term_context(
        self,
        user_id: str,
        session_id: str,
        k: int = 5,
    ) -> list[BaseMessage]:
        """Retrieve short-term conversation history."""
        if not self.memory_hooks:
            return []

        try:
            session = self.get_session(user_id, session_id)
            recent_turns = self.memory_hooks.get_last_turns(session, k=k)
            messages = self._parse_turns_to_messages(recent_turns)
            logger.info(f"Retrieved {len(messages)} context messages")
            return messages
        except Exception as e:
            logger.warning(f"Failed to retrieve short-term context: {e}")
            return []

    async def get_long_term_context(
        self,
        user_id: str,
        session_id: str,
        query: str | None,
    ) -> str | None:
        """Retrieve long-term memory context for prompt enrichment."""
        if not query or not self.memory_hooks:
            return None

        try:
            session = self.get_session(user_id, session_id)

            # Resolve namespaces
            prefs_namespace = self.memory_hooks.resolve_namespace("USER_PREFERENCES", actor_id=user_id, session_id=session_id)
            facts_namespace = self.memory_hooks.resolve_namespace("SEMANTIC_FACTS", actor_id=user_id, session_id=session_id)
            summaries_namespace = self.memory_hooks.resolve_namespace("SESSION_SUMMARIES", actor_id=user_id, session_id=session_id)

            # Retrieve from each namespace
            preferences = self.memory_hooks.retrieve_long_term(session, query, prefs_namespace, max_results=3) if prefs_namespace else []
            facts = self.memory_hooks.retrieve_long_term(session, query, facts_namespace, max_results=5) if facts_namespace else []
            summaries = self.memory_hooks.retrieve_long_term(session, query, summaries_namespace, max_results=3) if summaries_namespace else []

            if preferences or facts or summaries:
                logger.info(
                    "Retrieved long-term context counts: preferences=%d facts=%d summaries=%d",
                    len(preferences),
                    len(facts),
                    len(summaries),
                )
                return self._format_memory_context(preferences, facts, summaries)

            return None

        except Exception as e:
            logger.warning(f"Failed to retrieve long-term context: {e}")
            return None

    def add_user_message(self, user_id: str, session_id: str, message: str) -> None:
        """Store user message in memory."""
        if not self.memory_hooks:
            return

        try:
            session = self.get_session(user_id, session_id)
            self.memory_hooks.add_user_message(session, message)
        except Exception as e:
            logger.warning(f"Failed to store user message: {e}")

    def add_assistant_message(self, user_id: str, session_id: str, message: str) -> None:
        """Store assistant message in memory."""
        if not self.memory_hooks:
            return

        try:
            session = self.get_session(user_id, session_id)
            self.memory_hooks.add_assistant_message(session, message)
        except Exception as e:
            logger.warning(f"Failed to store assistant message: {e}")

    @staticmethod
    def _format_memory_context(
        preferences: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        summaries: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format long-term memory into context string."""
        parts = []

        if preferences:
            prefs_str = "; ".join([MemoryManager._record_snippet(p, 100) for p in preferences[:3]])
            parts.append(f"User preferences: {prefs_str}")

        if facts:
            facts_str = "; ".join([MemoryManager._record_snippet(f, 100) for f in facts[:5]])
            parts.append(f"Recent bill facts: {facts_str}")

        if summaries:
            summaries_str = "; ".join([MemoryManager._record_snippet(s, 100) for s in summaries[:3]])
            parts.append(f"Previous sessions: {summaries_str}")

        return " | ".join(parts) if parts else ""

    @staticmethod
    def _record_snippet(record: dict[str, Any], max_length: int) -> str:
        """Return a safe, truncated text snippet from a memory record."""
        content = record.get("content")
        snippet_source: str

        if isinstance(content, dict):
            text = content.get("text")
            snippet_source = text if isinstance(text, str) else json.dumps(content, ensure_ascii=False)
        elif isinstance(content, str):
            snippet_source = content
        else:
            snippet_source = json.dumps(record, ensure_ascii=False)

        snippet = snippet_source.strip()
        if len(snippet) > max_length:
            snippet = f"{snippet[:max_length].rstrip()}..."
        return snippet

    def _parse_turns_to_messages(self, turns: list[Any]) -> list[BaseMessage]:
        """Parse conversation turns into LangChain messages."""

        def _extract_message_data(item: Any) -> tuple[str | None, str | None]:  # noqa: ANN401
            """Normalize a raw turn item into (role, content)."""
            data: dict[str, Any] | None = None

            if isinstance(item, dict):
                data = item
            elif hasattr(item, "to_dict"):
                try:
                    candidate = item.to_dict()
                    if isinstance(candidate, dict):
                        data = candidate
                except Exception:  # pragma: no cover
                    data = None

            if data is None:
                potential_role = getattr(item, "role", getattr(item, "message_role", getattr(item, "sender", None)))
                potential_content = getattr(item, "content", getattr(item, "text", getattr(item, "message", None)))
                data = {
                    "role": potential_role,
                    "content": potential_content,
                }

            raw_role = data.get("role") or data.get("message_role") or data.get("sender")
            raw_content = data.get("content") or data.get("text") or data.get("message") or data.get("value") or data.get("body")

            role = str(raw_role).lower() if raw_role is not None else None

            if raw_content is None:
                content = None
            elif isinstance(raw_content, dict | list):
                try:
                    content = json.dumps(raw_content)
                except TypeError:  # pragma: no cover
                    content = str(raw_content)
            else:
                content = str(raw_content)

            return role, content

        messages: list[BaseMessage] = []

        def _consume(item: Any) -> None:  # noqa: ANN401
            if item is None:
                return

            if hasattr(item, "messages"):
                potential_messages = item.messages
                if isinstance(potential_messages, list | tuple):
                    _consume(potential_messages)
                    return

            if isinstance(item, dict) and "messages" in item and "role" not in item:
                potential_messages = item.get("messages")
                if isinstance(potential_messages, list | tuple):
                    _consume(potential_messages)
                    return

            if isinstance(item, list | tuple):
                for sub_item in item:
                    _consume(sub_item)
                return

            role, content = _extract_message_data(item)
            if not role or not content:
                return

            if role in {"user", "human"}:
                messages.append(HumanMessage(content=content))
            elif role in {"assistant", "ai"}:
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        for turn in turns:
            _consume(turn)

        return messages


# ============================================================================
# ASK BILL AGENT
# ============================================================================


class AskBillAgent(BaseAgent):
    """Ask Bill AI Agent with checkpointer-optimized workflow.

    Architecture:
    - 4 meaningful nodes optimized for checkpointing
    - ChatBedrockConverse LLM with modern Converse API
    - AgentCoreMemorySaver for state persistence
    - Minimal AskBillAgentState (only essential fields)
    - Clear checkpoint boundaries for pause/resume
    - LLM-driven logic (no hard-coded rules)

    Workflow nodes:
    1. process_query - Main LLM reasoning (checkpoint: conversation state)
    2. extract_bill_text - PDF download & extraction (checkpoint: extracted text)
    3. structure_bill_data - LLM structuring (checkpoint: structured JSON)
    4. process_query - Final response generation (checkpoint: complete response)
    """

    def __init__(self, config: AskBillConfig) -> None:
        """Initialize the agent with configuration."""
        super().__init__(config=config)
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Initialize circuit breaker for Bedrock API calls
        self.bedrock_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=30.0,
            expected_exception=Exception,
        )

        # Initialize SHARED LLM - ChatBedrockConverse (modern Converse API)
        self.llm = ChatBedrockConverse(
            model=config.model_id,  # Can use 'model' or 'model_id' (alias)
            region_name=config.region.value,
            temperature=config.temperature,
            max_tokens=int(config.max_tokens),
        )

        # Initialize memory manager
        memory_hooks = None
        if config.memory_config.memory_id:
            memory_hooks = AskBillMemoryHooks(
                memory_id=config.memory_config.memory_id,
                region=config.region.value,
                memory_config=config.memory_config,
            )
            logger.info(f"Memory hooks initialized: memory_id={config.memory_config.memory_id}")
        else:
            logger.warning("Memory ID not configured - memory disabled")

        self.memory = MemoryManager(memory_hooks)

        # Initialize AgentCoreMemorySaver checkpointer
        self.checkpointer = None
        if config.memory_config.memory_id and config.memory_config.checkpointing_enabled:
            self.checkpointer = AgentCoreMemorySaver(
                memory_id=config.memory_config.memory_id,
                region_name=config.region.value,
            )
            logger.info(f"AgentCoreMemorySaver initialized: memory_id={config.memory_config.memory_id}")
        else:
            logger.warning("Checkpointing not enabled - workflow state will not persist")

        # Input validation
        self.validator = InputValidator(
            max_message_length=5000,
            min_message_length=1,
            allow_empty_message=False,
        )

        # Build unified workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> CompiledStateGraph:
        """Build the checkpointer-optimized LangGraph workflow.

        Workflow:
        process_query → (needs bill?) → extract_bill_text →
        structure_bill_data → process_query → END

        Checkpoint boundaries:
        1. After process_query: Conversation state
        2. After extract_bill_text: Extracted text (avoids re-download)
        3. After structure_bill_data: Structured JSON (avoids re-LLM call)
        4. Final process_query: Complete response

        All nodes use AskBillAgentState and share the same LLM.
        """
        workflow = StateGraph(AskBillAgentState)

        # Add workflow nodes
        workflow.add_node(WorkflowNode.PROCESS_QUERY, self._process_query_node)
        workflow.add_node(
            WorkflowNode.EXTRACT_BILL_TEXT,
            lambda state: extract_bill_text(state, config=self.config),
        )
        workflow.add_node(
            WorkflowNode.STRUCTURE_BILL_DATA,
            self._structure_bill_data_wrapper,
        )

        # Set entry point
        workflow.set_entry_point(WorkflowNode.PROCESS_QUERY)

        # Define routing logic
        workflow.add_conditional_edges(
            WorkflowNode.PROCESS_QUERY,
            self._route_from_process_query,
            {
                WorkflowAction.EXTRACT_BILL: WorkflowNode.EXTRACT_BILL_TEXT,
                WorkflowAction.FINISH: END,
            },
        )

        # Bill structuring pipeline (sequential)
        workflow.add_edge(WorkflowNode.EXTRACT_BILL_TEXT, WorkflowNode.STRUCTURE_BILL_DATA)
        workflow.add_edge(WorkflowNode.STRUCTURE_BILL_DATA, WorkflowNode.PROCESS_QUERY)

        # Compile with checkpointer if available
        if self.checkpointer:
            logger.info("Compiling workflow with AgentCoreMemorySaver checkpointer")
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            logger.info("Compiling workflow without checkpointer")
            return workflow.compile()

    # ========================================================================
    # WORKFLOW NODE WRAPPERS
    # ========================================================================

    async def _structure_bill_data_wrapper(self, state: AskBillAgentState) -> dict[str, Any]:
        """Wrapper to properly await the async structure_bill_data function."""
        return await structure_bill_data(state, llm=self.llm, config=self.config)

    # ========================================================================
    # MAIN AGENT NODE
    # ========================================================================

    async def _process_query_node(self, state: AskBillAgentState) -> AskBillAgentState:
        """Main query processing node with LLM reasoning.

        This node:
        1. Checks if bill extraction is needed
        2. Calls LLM to generate response based on bill data
        3. Handles errors gracefully
        """
        try:
            # Check if bill structuring is needed BEFORE responding
            if state.file_path and not state.bill_json and not state.has_error():
                logger.info(f"Bill extraction needed for: {state.file_path[:80]}...")
                return state  # Will be routed to extract_bill_text

            # Extract state data (including any error information)
            logger.info("Processing query with ChatBedrockConverse LLM")
            user_query = state.current_query or ""
            bill_json = state.bill_json or {}
            error_info = state.error if state.has_error() else None

            if error_info:
                logger.error(f"Error in workflow, passing to LLM: {error_info}")

            # Retrieve memory context if enabled
            context_messages: list[BaseMessage] = []
            memory_context: str | None = None

            if self.memory.is_enabled():
                if self.config.memory_config.short_term_enabled:
                    context_messages = self.memory.get_short_term_context(state.user_id, state.session_id, k=5)

                if self.config.memory_config.long_term_enabled:
                    memory_context = await self.memory.get_long_term_context(state.user_id, state.session_id, user_query)

            # Build system prompt
            system_prompt = build_system_prompt(memory_context=memory_context or "")

            # Construct conversation with deduplication
            conversation_messages: list[BaseMessage] = [
                SystemMessage(content=system_prompt),
            ]

            # Add bill data or error information
            if error_info:
                conversation_messages.append(
                    SystemMessage(
                        content=f"ERROR: Bill processing failed:\n{error_info}\n\nPlease help the user understand what went wrong and suggest next steps."
                    )
                )
            else:
                conversation_messages.append(HumanMessage(content=f"Bill Data:\n{json.dumps(bill_json, ensure_ascii=False, indent=2)}"))

            # Add memory context messages
            conversation_messages.extend([m for m in context_messages if not isinstance(m, SystemMessage)])

            # Deduplicate and append state messages
            seen = {_message_signature(msg) for msg in conversation_messages}
            for message in state.messages:
                signature = _message_signature(message)
                if signature not in seen:
                    conversation_messages.append(message)
                    seen.add(signature)

            # Invoke LLM with circuit breaker protection
            logger.info(f"Calling ChatBedrockConverse for query: {user_query[:100]}")
            llm_response = await self._call_llm_with_circuit_protection(conversation_messages)

            # Extract response content
            raw_content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            answer = _as_text(cast("StrOrListStrOrDict", raw_content))

            # Log token usage if available (ChatBedrockConverse provides usage_metadata)
            if hasattr(llm_response, "usage_metadata") and llm_response.usage_metadata:
                logger.info(
                    f"Token usage - input: {llm_response.usage_metadata.get('input_tokens', 0)}, "
                    f"output: {llm_response.usage_metadata.get('output_tokens', 0)}, "
                    f"total: {llm_response.usage_metadata.get('total_tokens', 0)}"
                )

            logger.info(f"LLM response: {answer[:100]}...")

            # Update state
            state.messages.append(AIMessage(content=answer))
            state.response = answer

            return state

        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            raise WorkflowError(f"Query processing failed: {e}", state.session_id) from e

    def _route_from_process_query(self, state: AskBillAgentState) -> str:
        """Determine routing from process_query node."""
        # If we have a file path but no bill data and no error, extract the bill
        if state.file_path and not state.bill_json and not state.has_error():
            logger.info("Routing to extract_bill_text")
            return WorkflowAction.EXTRACT_BILL.value

        # Otherwise, we're done
        logger.info("Routing to finish")
        return WorkflowAction.FINISH.value

    async def _call_llm_with_circuit_protection(self, messages: list[BaseMessage]) -> BaseMessage:
        """Call AWS Bedrock LLM with circuit breaker fault tolerance.

        Wraps the LLM call with a circuit breaker to prevent cascading failures.
        After 3 consecutive failures, the circuit opens for 30 seconds.

        Args:
            messages: Conversation messages to send to LLM

        Returns:
            AI response message from Bedrock

        Raises:
            Exception: If LLM call fails or circuit is open
        """

        @self.bedrock_breaker
        async def _protected_bedrock_call() -> BaseMessage:
            return await self.llm.ainvoke(messages)

        return await _protected_bedrock_call()  # type: ignore[no-any-return]

    async def process_user_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process a user's billing question through the complete agent workflow.

        This is the main entry point for handling user requests. It:
        1. Validates and extracts inputs
        2. Stores user message in memory
        3. Executes the LangGraph workflow (with checkpointing if enabled)
        4. Stores assistant response in memory
        5. Returns formatted response

        Args:
            payload: Request containing:
                - message: User's question
                - user_id: User identifier (default: "default_user")
                - session_id: Session identifier (auto-generated if not provided)
                - file_path: Path to bill PDF (optional)

        Returns:
            Response dict containing:
                - session_id: Session identifier
                - user_id: User identifier
                - response: AI assistant's answer
                - bill_json: Structured bill data (if available)

        Raises:
            AgentInvocationError: If validation or workflow execution fails
        """
        session_id = payload.get("session_id") or str(uuid.uuid4())  # For error handling

        try:
            # Extract and validate inputs
            user_id, session_id, message, file_path = self._extract_and_validate_inputs(payload)

            logger.info(f"Processing request for session {session_id}, user {user_id}")

            # Execute workflow with checkpointing config
            initial_state = AskBillAgentState(
                messages=[HumanMessage(content=message)],
                session_id=session_id,
                user_id=user_id,
                current_query=message,
                file_path=file_path,
            )

            # If checkpointer is enabled, pass config with thread_id for checkpointing
            if self.checkpointer:
                config = {
                    "configurable": {
                        "thread_id": session_id,  # Use session_id as thread_id
                        "checkpoint_ns": user_id,  # Namespace by user
                    }
                }
                logger.info(f"Invoking workflow WITH checkpointing: thread_id={session_id}, checkpoint_ns={user_id}")
                result = await self.workflow.ainvoke(initial_state, config=config)
            else:
                logger.info("Invoking workflow WITHOUT checkpointing")
                result = await self.workflow.ainvoke(initial_state)

            logger.info(f"Workflow result type: {type(result).__name__}, has file_path: {bool(file_path)}")

            # Extract response - with checkpointing, result is always a dict
            if isinstance(result, dict):
                messages = result.get("messages", [])
                bill_json = result.get("bill_json")
            else:
                # Fallback for when checkpointing is disabled
                messages = result.messages
                bill_json = result.bill_json

            response_content = _as_text(cast("StrOrListStrOrDict", messages[-1].content)) if messages else "No response"

            # Store both user and assistant messages in memory after workflow completes
            self.memory.add_user_message(user_id, session_id, message)
            self.memory.add_assistant_message(user_id, session_id, response_content)

            return {
                "session_id": session_id,
                "user_id": user_id,
                "response": response_content,
                "bill_json": bill_json,
            }

        except ValidationError as ve:
            logger.warning(f"Input validation failed: {ve}")
            raise AgentInvocationError(f"Invalid input: {ve}", session_id) from ve
        except AgentInvocationError:
            raise
        except Exception as e:
            logger.error(f"Request processing failed: {e}", exc_info=True)
            raise AgentInvocationError(f"Request processing failed: {e}", session_id) from e

    def _extract_and_validate_inputs(self, payload: dict[str, Any]) -> tuple[str, str, str, str | None]:
        """Extract and validate all inputs from request payload.

        Args:
            payload: Request payload dictionary

        Returns:
            Tuple of (user_id, session_id, message, file_path)

        Raises:
            ValidationError: If any input validation fails
        """
        # Extract inputs
        session_id = payload.get("session_id") or str(uuid.uuid4())
        user_id = payload.get("user_id", "default_user")
        message = payload.get("message") or payload.get("current_query", "")
        file_path = payload.get("file_path")

        # Validate
        validated_message = self.validator.validate_message(message)
        validated_user_id = self.validator.validate_user_id(user_id)
        validated_session_id = self.validator.validate_session_id(session_id) if session_id else session_id

        return validated_user_id, validated_session_id, validated_message, file_path

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Public API entry point - delegates to process_user_request.

        Maintained for backward compatibility with existing callers.

        Args:
            payload: Request payload (see process_user_request for details)

        Returns:
            Response dict (see process_user_request for details)

        Raises:
            AgentInvocationError: If request processing fails
        """
        return await self.process_user_request(payload)
