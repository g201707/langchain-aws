"""Ask Bill Agent – Best of Both Worlds Version.

Combines:
- User's superior architecture (explicit memory hooks, tuple namespaces)
- Claude's modern API (ChatBedrockConverse)
- Enhanced observability (token tracking)

Key Features:
- Explicit pre/post memory hooks for clean separation
- Tuple-based namespaces for AgentCoreMemoryStore
- Conversation history persisted in long-term store
- ChatBedrockConverse for modern Bedrock Converse API
- Full token usage tracking
- 4-node workflow optimized for checkpointing
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

from ask_bill.agents.ask_bill.bill_pdf_structurer import (
    extract_bill_text,
    structure_bill_data,
)
from ask_bill.agents.ask_bill.prompts import build_system_prompt
from ask_bill.agents.base.agent import BaseAgent
from ask_bill.core.circuit_breaker import CircuitBreaker
from ask_bill.core.exceptions.agent import AgentInvocationError, WorkflowError
from ask_bill.core.models.state import AskBillAgentState, WorkflowAction, WorkflowNode
from ask_bill.core.validation import InputValidator, ValidationError

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ask_bill.core.config.bedrock import AskBillConfig


logger = logging.getLogger(__name__)

# Type alias for LLM content
StrOrListStrOrDict: TypeAlias = str | list[str | dict[str, Any]]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _as_text(value: StrOrListStrOrDict) -> str:
    """Normalize LLM content into a single string."""
    if isinstance(value, str):
        return value
    return " ".join(
        item
        if isinstance(item, str)
        else json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for item in value
    )


def _message_signature(message: BaseMessage) -> tuple[str, str | None, str | None]:
    """Generate unique signature for message deduplication."""
    content = getattr(message, "content", None)
    if isinstance(content, (list, dict)):
        try:
            content_repr: str | None = json.dumps(content, sort_keys=True)
        except TypeError:
            content_repr = repr(content)
    else:
        content_repr = repr(content) if content is not None else None

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is not None:
        try:
            tool_repr: str | None = json.dumps(tool_calls, sort_keys=True)
        except TypeError:
            tool_repr = repr(tool_calls)
    else:
        tool_repr = None

    return (message.__class__.__name__, content_repr, tool_repr)


# ============================================================================
# ASK BILL AGENT - BEST OF BOTH WORLDS
# ============================================================================


class AskBillAgent(BaseAgent):
    """Ask Bill AI Agent - Production-Ready with Modern API and Clean Architecture.

    Architecture:
    -------------
    - 4-node workflow: PROCESS_QUERY → EXTRACT_BILL_TEXT → STRUCTURE_BILL_DATA → END
    - ChatBedrockConverse: Modern Bedrock Converse API
    - AgentCoreMemorySaver: Short-term checkpointing
    - AgentCoreMemoryStore: Long-term semantic memory with tuple namespaces
    - Explicit pre/post memory hooks for clean separation

    Memory Strategy:
    ----------------
    Tuple-based namespaces for clarity and type safety:
    - ("conversation", actor_id, thread_id) - Full conversation history
    - ("preferences", actor_id) - User preferences
    - ("semantic_facts", actor_id) - Bill-related facts
    - ("session_summaries", actor_id) - Session summaries

    Workflow:
    ---------
    1. PROCESS_QUERY: Check if bill extraction needed
       - Pre-hook: Save user message + retrieve long-term context
       - Call LLM with memory context
       - Post-hook: Save AI response
    2. EXTRACT_BILL_TEXT: Download PDF and extract text (if needed)
    3. STRUCTURE_BILL_DATA: LLM-based structuring (if needed)
    4. PROCESS_QUERY: Generate final response with structured data
    """

    def __init__(self, config: AskBillConfig) -> None:
        """Initialize the agent with configuration.

        Args:
            config: AskBillConfig with model settings and memory configuration
        """
        super().__init__(config=config)
        self.logger = logging.getLogger(__name__)
        self.config = config

        # =====================================================================
        # Circuit Breaker
        # =====================================================================
        self.bedrock_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=30.0,
            expected_exception=Exception,
        )

        # =====================================================================
        # ChatBedrockConverse (Modern Converse API)
        # =====================================================================
        self.llm = ChatBedrockConverse(
            model=config.model_id,
            region_name=config.region.value,
            temperature=float(getattr(config, "temperature", 0.7)),
            max_tokens=int(getattr(config, "max_tokens", 2048)),
        )
        logger.info(f"Initialized ChatBedrockConverse: model={config.model_id}")

        # =====================================================================
        # AgentCore Memory Integrations
        # =====================================================================
        memory_id = (
            getattr(config, "memory_config", None).memory_id
            if getattr(config, "memory_config", None)
            else None
        )
        region_name = config.region.value

        self.checkpointer: AgentCoreMemorySaver | None = None
        self.memory_store: AgentCoreMemoryStore | None = None

        if memory_id:
            # Short-term / state persistence
            self.checkpointer = AgentCoreMemorySaver(memory_id, region_name=region_name)
            # Long-term semantic memory
            self.memory_store = AgentCoreMemoryStore(memory_id, region_name=region_name)
            logger.info(
                "AgentCore Memory enabled: memory_id=%s region=%s",
                memory_id,
                region_name,
            )
        else:
            logger.warning("AgentCore Memory not configured (memory_id is empty)")

        # =====================================================================
        # Input Validation
        # =====================================================================
        self.validator = InputValidator(
            max_message_length=5000,
            min_message_length=1,
            allow_empty_message=False,
        )

        # =====================================================================
        # Build Workflow
        # =====================================================================
        self.workflow = self._build_workflow()

    # =========================================================================
    # WORKFLOW BUILD
    # =========================================================================

    def _build_workflow(self) -> CompiledStateGraph:
        """Build the checkpointer-optimized LangGraph workflow.

        Returns:
            Compiled StateGraph with optional checkpointer
        """
        workflow = StateGraph(AskBillAgentState)

        # Main processing node
        workflow.add_node(WorkflowNode.PROCESS_QUERY, self._process_query_node)

        # Bill structuring pipeline
        workflow.add_node(
            WorkflowNode.EXTRACT_BILL_TEXT,
            lambda state: extract_bill_text(state, config=self.config),
        )
        workflow.add_node(
            WorkflowNode.STRUCTURE_BILL_DATA,
            self._structure_bill_data_wrapper,
        )

        # Entry point
        workflow.set_entry_point(WorkflowNode.PROCESS_QUERY)

        # Routing from PROCESS_QUERY
        workflow.add_conditional_edges(
            WorkflowNode.PROCESS_QUERY,
            self._route_from_process_query,
            {
                WorkflowAction.EXTRACT_BILL: WorkflowNode.EXTRACT_BILL_TEXT,
                WorkflowAction.FINISH: END,
            },
        )

        # Structuring pipeline edges
        workflow.add_edge(WorkflowNode.EXTRACT_BILL_TEXT, WorkflowNode.STRUCTURE_BILL_DATA)
        workflow.add_edge(WorkflowNode.STRUCTURE_BILL_DATA, WorkflowNode.PROCESS_QUERY)

        # Compile with checkpointer if available
        if self.checkpointer:
            logger.info("Compiling workflow WITH AgentCoreMemorySaver checkpointer")
            return workflow.compile(checkpointer=self.checkpointer)

        logger.info("Compiling workflow WITHOUT checkpointer")
        return workflow.compile()

    async def _structure_bill_data_wrapper(
        self, state: AskBillAgentState
    ) -> dict[str, Any]:
        """Async wrapper for bill structuring node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with bill_json or error
        """
        return await structure_bill_data(state, llm=self.llm, config=self.config)

    # =========================================================================
    # MEMORY HOOKS (Pre/Post LLM)
    # =========================================================================

    async def _pre_model_memory(
        self,
        state: AskBillAgentState,
        user_query: str,
    ) -> str | None:
        """Pre-LLM hook: Save user message + retrieve long-term memory context.

        This hook:
        1. Saves the user's message to AgentCoreMemoryStore
        2. Retrieves relevant long-term memories (preferences, facts, summaries)
        3. Returns formatted context string to inject into system prompt

        Args:
            state: Current workflow state
            user_query: User's current query

        Returns:
            Formatted memory context string or None if no relevant memory
        """
        if not self.memory_store:
            return None

        actor_id = state.user_id or "anonymous"
        thread_id = state.session_id or "default"

        # Namespace for raw conversation messages
        conversation_ns = ("conversation", actor_id, thread_id)

        # Save user message to store
        try:
            await self.memory_store.put(
                conversation_ns,
                str(uuid.uuid4()),
                {"role": "user", "content": user_query},
            )
            logger.debug(f"Saved user message to namespace: {conversation_ns}")
        except Exception as e:
            logger.warning("Failed to store user message in AgentCore Memory: %s", e)

        # Retrieve long-term memories from semantic namespaces
        try:
            preferences_ns = ("preferences", actor_id)
            facts_ns = ("semantic_facts", actor_id)
            summaries_ns = ("session_summaries", actor_id)

            preferences = await self._memory_search(preferences_ns, user_query, limit=3)
            facts = await self._memory_search(facts_ns, user_query, limit=5)
            summaries = await self._memory_search(summaries_ns, user_query, limit=3)

            # Format context
            parts: list[str] = []

            if preferences:
                prefs_text = "; ".join(preferences)
                parts.append(f"User preferences: {prefs_text}")
            if facts:
                facts_text = "; ".join(facts)
                parts.append(f"Recent bill facts: {facts_text}")
            if summaries:
                summaries_text = "; ".join(summaries)
                parts.append(f"Previous session summaries: {summaries_text}")

            if parts:
                logger.info(
                    f"Retrieved long-term memory: prefs={len(preferences)}, "
                    f"facts={len(facts)}, summaries={len(summaries)}"
                )
                return " | ".join(parts)

            return None

        except Exception as e:
            logger.warning("Failed to retrieve long-term memory: %s", e)
            return None

    async def _post_model_memory(
        self,
        state: AskBillAgentState,
        answer: str,
    ) -> None:
        """Post-LLM hook: Save AI response to AgentCoreMemoryStore.

        Args:
            state: Current workflow state
            answer: AI's response to save
        """
        if not self.memory_store:
            return

        actor_id = state.user_id or "anonymous"
        thread_id = state.session_id or "default"
        conversation_ns = ("conversation", actor_id, thread_id)

        try:
            await self.memory_store.put(
                conversation_ns,
                str(uuid.uuid4()),
                {"role": "assistant", "content": answer},
            )
            logger.debug(f"Saved assistant message to namespace: {conversation_ns}")
        except Exception as e:
            logger.warning("Failed to store assistant message: %s", e)

    async def _memory_search(
        self,
        namespace: tuple[str, ...],
        query: str,
        limit: int,
    ) -> list[str]:
        """Search AgentCore Memory and return text snippets.

        Args:
            namespace: Tuple-based namespace (e.g., ("preferences", "user_123"))
            query: Search query
            limit: Maximum results to return

        Returns:
            List of text snippets from matching records
        """
        if not self.memory_store:
            return []

        try:
            results = await self.memory_store.search(namespace, query=query, limit=limit)
            snippets: list[str] = []

            for item in results:
                value = item.get("value", {})
                content = (
                    value.get("content")
                    or value.get("message")
                    or value.get("text")
                    or ""
                )

                if isinstance(content, dict):
                    content = content.get("text") or json.dumps(
                        content, ensure_ascii=False
                    )
                if not isinstance(content, str):
                    content = str(content)

                content = content.strip()
                if content:
                    # Truncate to prevent context explosion
                    if len(content) > 160:
                        content = content[:157].rstrip() + "..."
                    snippets.append(content)

            return snippets

        except Exception as e:
            logger.warning(f"Memory search failed for namespace {namespace}: %s", e)
            return []

    # =========================================================================
    # MAIN NODE: PROCESS_QUERY
    # =========================================================================

    async def _process_query_node(
        self, state: AskBillAgentState
    ) -> AskBillAgentState:
        """Main query processing node with LLM reasoning.

        This node:
        1. Checks if bill extraction is needed (routing decision)
        2. Calls pre-memory hook to save user message and retrieve context
        3. Builds conversation with memory context
        4. Calls LLM with circuit breaker protection
        5. Logs token usage (ChatBedrockConverse provides detailed metadata)
        6. Calls post-memory hook to save AI response
        7. Updates state with response

        Args:
            state: Current workflow state

        Returns:
            Updated state with response or routing decision

        Raises:
            WorkflowError: If query processing fails
        """
        try:
            # =================================================================
            # Step 1: Check if bill extraction is required
            # =================================================================
            if state.file_path and not state.bill_json and not state.has_error():
                logger.info("Bill extraction required for file_path=%s", state.file_path)
                return state  # Routing function will send to EXTRACT_BILL_TEXT

            # =================================================================
            # Step 2: Extract state data
            # =================================================================
            user_query = state.current_query or ""
            bill_json = state.bill_json or {}
            error_info = state.error if state.has_error() else None

            # =================================================================
            # Step 3: Pre-model memory hook
            # =================================================================
            memory_context = await self._pre_model_memory(state, user_query)

            # =================================================================
            # Step 4: Build conversation
            # =================================================================
            system_prompt = build_system_prompt(memory_context=memory_context or "")
            conversation_messages: list[BaseMessage] = [
                SystemMessage(content=system_prompt)
            ]

            # Add bill data or error context
            if error_info:
                conversation_messages.append(
                    SystemMessage(
                        content=(
                            "Bill processing failed with the following error. "
                            "Explain clearly to the customer what went wrong and what they should do next.\n\n"
                            f"{error_info}"
                        )
                    )
                )
            elif bill_json:
                conversation_messages.append(
                    HumanMessage(
                        content=(
                            f"Here is the structured bill JSON:\n"
                            f"{json.dumps(bill_json, ensure_ascii=False, indent=2)}\n\n"
                            f"User Question: {user_query}"
                        )
                    )
                )
            else:
                conversation_messages.append(HumanMessage(content=user_query))

            # Deduplicate and add existing messages from state
            seen = {_message_signature(msg) for msg in conversation_messages}
            for msg in state.messages:
                sig = _message_signature(msg)
                if sig not in seen:
                    conversation_messages.append(msg)
                    seen.add(sig)

            # =================================================================
            # Step 5: Call LLM with circuit breaker
            # =================================================================
            logger.info("Calling ChatBedrockConverse: query=%s", user_query[:100])
            llm_response = await self._call_llm_with_circuit_protection(
                conversation_messages
            )

            # =================================================================
            # Step 6: Extract response and log token usage
            # =================================================================
            raw_content = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )
            answer = _as_text(cast(StrOrListStrOrDict, raw_content))

            # Token usage tracking (ChatBedrockConverse provides full metadata)
            if hasattr(llm_response, "usage_metadata") and llm_response.usage_metadata:
                usage = llm_response.usage_metadata
                logger.info(
                    "Token usage - input: %s, output: %s, total: %s",
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("total_tokens", 0),
                )

            logger.info("LLM response: %s", answer[:100])

            # =================================================================
            # Step 7: Post-model memory hook
            # =================================================================
            await self._post_model_memory(state, answer)

            # =================================================================
            # Step 8: Update state
            # =================================================================
            state.messages.append(AIMessage(content=answer))
            state.response = answer

            return state

        except Exception as e:
            logger.error("Query processing error: %s", e, exc_info=True)
            raise WorkflowError(
                f"Query processing failed: {e}", state.session_id
            ) from e

    def _route_from_process_query(self, state: AskBillAgentState) -> str:
        """Route from PROCESS_QUERY node to either bill extraction or finish.

        Args:
            state: Current workflow state

        Returns:
            WorkflowAction value (EXTRACT_BILL or FINISH)
        """
        if state.file_path and not state.bill_json and not state.has_error():
            logger.info("Routing: PROCESS_QUERY → EXTRACT_BILL_TEXT")
            return WorkflowAction.EXTRACT_BILL.value

        logger.info("Routing: PROCESS_QUERY → FINISH")
        return WorkflowAction.FINISH.value

    # =========================================================================
    # LLM CALL WITH CIRCUIT BREAKER
    # =========================================================================

    async def _call_llm_with_circuit_protection(
        self, messages: list[BaseMessage]
    ) -> BaseMessage:
        """Call AWS Bedrock LLM with circuit breaker protection.

        Args:
            messages: Conversation messages to send to LLM

        Returns:
            AI response message from Bedrock

        Raises:
            Exception: If LLM call fails or circuit is open
        """

        @self.bedrock_breaker
        async def _protected_call() -> BaseMessage:
            return await self.llm.ainvoke(messages)

        return await _protected_call()  # type: ignore[no-any-return]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def _extract_and_validate_inputs(
        self, payload: dict[str, Any]
    ) -> tuple[str, str, str, str | None]:
        """Extract and validate all inputs from the request payload.

        Args:
            payload: Request payload dictionary

        Returns:
            Tuple of (user_id, session_id, message, file_path)

        Raises:
            ValidationError: If any input validation fails
        """
        session_id = payload.get("session_id") or str(uuid.uuid4())
        user_id = payload.get("user_id", "default_user")
        message = payload.get("message") or payload.get("current_query", "")
        file_path = payload.get("file_path")

        validated_message = self.validator.validate_message(message)
        validated_user_id = self.validator.validate_user_id(user_id)
        validated_session_id = (
            self.validator.validate_session_id(session_id) if session_id else session_id
        )

        return validated_user_id, validated_session_id, validated_message, file_path

    async def process_user_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process a user request end-to-end.

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
        session_id = payload.get("session_id") or str(
            uuid.uuid4()
        )  # For error reporting

        try:
            # =================================================================
            # Step 1: Validate inputs
            # =================================================================
            user_id, session_id, message, file_path = self._extract_and_validate_inputs(
                payload
            )
            logger.info(
                "Processing request: session=%s user=%s has_file=%s",
                session_id,
                user_id,
                bool(file_path),
            )

            # =================================================================
            # Step 2: Build initial state
            # =================================================================
            initial_state = AskBillAgentState(
                messages=[HumanMessage(content=message)],
                session_id=session_id,
                user_id=user_id,
                current_query=message,
                file_path=file_path,
            )

            # =================================================================
            # Step 3: Invoke workflow
            # =================================================================
            result = await self.workflow.ainvoke(initial_state)
            logger.info("Workflow result type: %s", type(result).__name__)

            # =================================================================
            # Step 4: Extract response
            # =================================================================
            if isinstance(result, dict):
                messages = result.get("messages", [])
                bill_json = result.get("bill_json")
            else:
                messages = result.messages
                bill_json = result.bill_json

            response_text = (
                _as_text(cast(StrOrListStrOrDict, messages[-1].content))
                if messages
                else "No response"
            )

            return {
                "session_id": session_id,
                "user_id": user_id,
                "response": response_text,
                "bill_json": bill_json,
            }

        except ValidationError as ve:
            logger.warning("Input validation failed: %s", ve)
            raise AgentInvocationError(f"Invalid input: {ve}", session_id) from ve
        except AgentInvocationError:
            raise
        except Exception as e:
            logger.error("Request processing failed: %s", e, exc_info=True)
            raise AgentInvocationError(
                f"Request processing failed: {e}", session_id
            ) from e

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Backward-compatible public entrypoint.

        Delegates to process_user_request for actual processing.

        Args:
            payload: Request payload (see process_user_request for details)

        Returns:
            Response dict (see process_user_request for details)

        Raises:
            AgentInvocationError: If request processing fails
        """
        return await self.process_user_request(payload)
