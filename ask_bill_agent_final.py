"""Ask Bill Agent – Production-Ready, Checkpointer-Optimized Version.

Key Features
============
- ChatBedrockConverse: Modern Bedrock Converse API with enhanced features
- AgentCoreMemorySaver: Automatic checkpointing for state persistence
- AgentCoreMemoryStore: Long-term memory for user preferences and facts
- 4-node workflow optimized for clear checkpoint boundaries
- LLM-driven reasoning (no hard-coded business logic)
- Circuit breaker for fault tolerance
- Comprehensive error handling

Workflow Nodes
==============
1. PROCESS_QUERY       – Main LLM reasoning (pre & post structuring)
2. EXTRACT_BILL_TEXT   – PDF download and text extraction (expensive I/O)
3. STRUCTURE_BILL_DATA – LLM-based structuring of extracted text
4. END                 – Terminal state via WorkflowAction.FINISH

Checkpoint Boundaries
=====================
- After PROCESS_QUERY: Conversation state saved
- After EXTRACT_BILL_TEXT: Extracted text saved (avoids re-download)
- After STRUCTURE_BILL_DATA: Structured JSON saved (avoids re-LLM call)
- After final PROCESS_QUERY: Complete response saved
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

# Import bill PDF structuring utilities
from ask_bill.agents.ask_bill.bill_pdf_structurer import (
    extract_bill_text,
    structure_bill_data,
)
from ask_bill.agents.ask_bill.prompts import build_system_prompt
from ask_bill.agents.base.agent import BaseAgent
from ask_bill.core.circuit_breaker import CircuitBreaker
from ask_bill.core.exceptions.agent import AgentInvocationError, WorkflowError
from ask_bill.core.validation import InputValidator, ValidationError

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ask_bill.core.config.bedrock import AskBillConfig
    from ask_bill.core.models.state import AskBillAgentState, WorkflowAction, WorkflowNode


logger = logging.getLogger(__name__)

# Type alias for LLM content (string or structured list)
StrOrListStrOrDict: TypeAlias = str | list[str | dict[str, Any]]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _as_text(value: StrOrListStrOrDict) -> str:
    """Convert LLM content into a normalized text string.

    Args:
        value: Content from LLM (string or list of strings/dicts)

    Returns:
        Normalized text string
    """
    if isinstance(value, str):
        return value

    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        else:
            # Serialize dicts/objects to stable JSON
            parts.append(
                json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            )
    return " ".join(parts)


def _message_signature(message: BaseMessage) -> tuple[str, str | None, str | None]:
    """Generate unique signature for message deduplication.

    Args:
        message: LangChain message object

    Returns:
        Tuple of (class_name, content_repr, tool_calls_repr)
    """
    content = getattr(message, "content", None)
    content_repr: str | None

    if isinstance(content, (list, dict)):
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
# ASK BILL AGENT
# ============================================================================


class AskBillAgent(BaseAgent):
    """Production-ready Ask Bill AI Agent.

    Architecture
    ------------
    - 4 meaningful nodes optimized for checkpointing
    - ChatBedrockConverse for modern Bedrock Converse API
    - AgentCoreMemorySaver for automatic state persistence
    - AgentCoreMemoryStore for long-term user memory
    - Minimal state design for efficient checkpointing
    - LLM-driven logic (no hard-coded rules)
    - Circuit breaker for fault tolerance

    Workflow
    --------
    1. PROCESS_QUERY: Check if bill extraction needed
       - Has file_path but no bill_json → route to EXTRACT_BILL_TEXT
       - Has bill_json or error → generate response
       - No file_path → answer directly

    2. EXTRACT_BILL_TEXT: Download PDF and extract text
       - Success → route to STRUCTURE_BILL_DATA
       - Error → route back to PROCESS_QUERY with error info

    3. STRUCTURE_BILL_DATA: Use LLM to structure text into JSON
       - Success → route back to PROCESS_QUERY with bill_json
       - Error → route back to PROCESS_QUERY with error info

    4. PROCESS_QUERY (final): Generate user-facing response
       - Include bill_json or error context
       - Retrieve long-term memory if available
       - Return final answer
    """

    def __init__(self, config: AskBillConfig) -> None:
        """Initialize the agent with configuration.

        Args:
            config: AskBillConfig with model settings and memory configuration
        """
        super().__init__(config=config)
        self.config = config
        self.logger = logging.getLogger(__name__)

        # =====================================================================
        # Circuit Breaker for Bedrock API calls
        # =====================================================================
        self.bedrock_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=30.0,
            expected_exception=Exception,
        )

        # =====================================================================
        # Shared LLM Instance (ChatBedrockConverse)
        # =====================================================================
        self.llm = ChatBedrockConverse(
            model=config.model_id,
            region_name=config.region.value,
            temperature=float(config.temperature),
            max_tokens=int(config.max_tokens),
        )
        logger.info(f"Initialized ChatBedrockConverse: model={config.model_id}")

        # =====================================================================
        # AgentCore Memory Integrations
        # =====================================================================
        self.memory_id: str | None = getattr(config.memory_config, "memory_id", None)
        self.checkpointer: AgentCoreMemorySaver | None = None
        self.store: AgentCoreMemoryStore | None = None

        if self.memory_id:
            # Short-term memory (checkpointing)
            self.checkpointer = AgentCoreMemorySaver(
                memory_id=self.memory_id,
                region_name=config.region.value,
            )
            # Long-term memory (semantic storage)
            self.store = AgentCoreMemoryStore(
                memory_id=self.memory_id,
                region_name=config.region.value,
            )
            logger.info(
                f"AgentCore Memory enabled: memory_id={self.memory_id}, "
                f"checkpointing={self.checkpointer is not None}, "
                f"store={self.store is not None}"
            )
        else:
            logger.warning(
                "AgentCore Memory ID not configured – checkpointing and long-term memory disabled"
            )

        # =====================================================================
        # Input Validation
        # =====================================================================
        self.validator = InputValidator(
            max_message_length=5000,
            min_message_length=1,
            allow_empty_message=False,
        )

        # =====================================================================
        # Build LangGraph Workflow
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
        from ask_bill.core.models.state import AskBillAgentState, WorkflowAction, WorkflowNode

        graph = StateGraph(AskBillAgentState)

        # Add nodes
        graph.add_node(WorkflowNode.PROCESS_QUERY, self._process_query_node)
        graph.add_node(
            WorkflowNode.EXTRACT_BILL_TEXT,
            lambda state: extract_bill_text(state, config=self.config),
        )
        graph.add_node(
            WorkflowNode.STRUCTURE_BILL_DATA,
            self._structure_bill_data_node,
        )

        # Set entry point
        graph.set_entry_point(WorkflowNode.PROCESS_QUERY)

        # Routing from PROCESS_QUERY
        graph.add_conditional_edges(
            WorkflowNode.PROCESS_QUERY,
            self._route_from_process_query,
            {
                WorkflowAction.EXTRACT_BILL: WorkflowNode.EXTRACT_BILL_TEXT,
                WorkflowAction.FINISH: END,
            },
        )

        # Sequential bill processing pipeline
        graph.add_edge(WorkflowNode.EXTRACT_BILL_TEXT, WorkflowNode.STRUCTURE_BILL_DATA)
        graph.add_edge(WorkflowNode.STRUCTURE_BILL_DATA, WorkflowNode.PROCESS_QUERY)

        # Compile with checkpointer if available
        if self.checkpointer:
            logger.info("Compiling workflow WITH AgentCoreMemorySaver checkpointer")
            return graph.compile(checkpointer=self.checkpointer)
        else:
            logger.info("Compiling workflow WITHOUT checkpointer")
            return graph.compile()

    # =========================================================================
    # WORKFLOW NODE IMPLEMENTATIONS
    # =========================================================================

    async def _structure_bill_data_node(self, state: AskBillAgentState) -> dict[str, Any]:
        """Async wrapper for structure_bill_data function.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with bill_json or error
        """
        return await structure_bill_data(state, llm=self.llm, config=self.config)

    async def _process_query_node(self, state: AskBillAgentState) -> AskBillAgentState:
        """Main query processing node with LLM reasoning.

        This node handles:
        1. Routing to bill extraction if needed
        2. Retrieving long-term memory context
        3. Calling LLM to generate response
        4. Handling errors gracefully

        Args:
            state: Current workflow state

        Returns:
            Updated state with response or routing decision
        """
        try:
            # =================================================================
            # Step 1: Check if bill extraction/structuring is needed
            # =================================================================
            if state.file_path and not state.bill_json and not state.has_error():
                logger.info(f"Bill extraction needed for: {state.file_path[:80]}...")
                return state  # Will be routed to EXTRACT_BILL_TEXT

            # =================================================================
            # Step 2: Extract state data
            # =================================================================
            user_query: str = state.current_query or ""
            bill_json: dict[str, Any] = state.bill_json or {}
            error_info: str | None = state.error if state.has_error() else None

            if error_info:
                logger.error(f"Workflow error detected: {error_info}")

            # =================================================================
            # Step 3: Retrieve long-term memory context
            # =================================================================
            memory_context = await self._get_long_term_memory_context(
                actor_id=state.user_id,
                session_id=state.session_id,
                query=user_query,
            )

            # =================================================================
            # Step 4: Build system prompt
            # =================================================================
            system_prompt = build_system_prompt(memory_context=memory_context or "")

            # =================================================================
            # Step 5: Construct conversation messages
            # =================================================================
            conversation: list[BaseMessage] = [SystemMessage(content=system_prompt)]

            # Add bill data or error information
            if error_info:
                conversation.append(
                    SystemMessage(
                        content=(
                            f"ERROR: Bill processing failed:\n{error_info}\n\n"
                            "Please help the user understand what went wrong and suggest next steps."
                        )
                    )
                )
            elif bill_json:
                conversation.append(
                    HumanMessage(
                        content=(
                            f"Bill Data:\n{json.dumps(bill_json, ensure_ascii=False, indent=2)}\n\n"
                            f"User Question: {user_query}"
                        )
                    )
                )
            else:
                # No bill data available
                conversation.append(
                    HumanMessage(content=f"User Question: {user_query}")
                )

            # Deduplicate and append existing messages from state
            seen_signatures = {_message_signature(msg) for msg in conversation}
            for message in state.messages:
                sig = _message_signature(message)
                if sig not in seen_signatures:
                    conversation.append(message)
                    seen_signatures.add(sig)

            # =================================================================
            # Step 6: Call LLM with circuit breaker protection
            # =================================================================
            logger.info(f"Calling ChatBedrockConverse for query: {user_query[:100]}")
            llm_response = await self._call_llm_with_circuit_protection(conversation)

            # =================================================================
            # Step 7: Extract and log response
            # =================================================================
            raw_content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            answer = _as_text(cast(StrOrListStrOrDict, raw_content))

            # Log token usage if available
            if hasattr(llm_response, "usage_metadata") and llm_response.usage_metadata:
                usage = llm_response.usage_metadata
                logger.info(
                    f"Token usage - input: {usage.get('input_tokens', 0)}, "
                    f"output: {usage.get('output_tokens', 0)}, "
                    f"total: {usage.get('total_tokens', 0)}"
                )

            logger.info(f"LLM response: {answer[:100]}...")

            # =================================================================
            # Step 8: Update state with response
            # =================================================================
            state.messages.append(AIMessage(content=answer))
            state.response = answer

            return state

        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            raise WorkflowError(f"Query processing failed: {e}", state.session_id) from e

    def _route_from_process_query(self, state: AskBillAgentState) -> str:
        """Determine routing from process_query node.

        Args:
            state: Current workflow state

        Returns:
            WorkflowAction value (EXTRACT_BILL or FINISH)
        """
        from ask_bill.core.models.state import WorkflowAction

        # If we have a file path but no bill data and no error, extract the bill
        if state.file_path and not state.bill_json and not state.has_error():
            logger.info("Routing to EXTRACT_BILL_TEXT")
            return WorkflowAction.EXTRACT_BILL.value

        # Otherwise, we're done
        logger.info("Routing to FINISH")
        return WorkflowAction.FINISH.value

    # =========================================================================
    # LONG-TERM MEMORY INTEGRATION
    # =========================================================================

    async def _get_long_term_memory_context(
        self,
        actor_id: str,
        session_id: str,
        query: str,
    ) -> str | None:
        """Retrieve long-term memory context from AgentCoreMemoryStore.

        Retrieves relevant context from:
        - User preferences
        - Semantic facts (bill-related information)
        - Session summaries

        Args:
            actor_id: User identifier
            session_id: Session identifier
            query: User's current query

        Returns:
            Formatted context string or None if no relevant memory found
        """
        if not self.store or not query:
            return None

        try:
            # Define memory namespaces
            namespaces = {
                "preferences": f"user:{actor_id}:preferences",
                "facts": f"user:{actor_id}:facts",
                "summaries": f"session:{session_id}:summaries",
            }

            # Retrieve from each namespace
            memories: dict[str, list[dict[str, Any]]] = {}
            for key, namespace in namespaces.items():
                try:
                    # Use store.search() for semantic retrieval
                    results = await self.store.asearch(
                        query=query,
                        namespace=namespace,
                        limit=3 if key == "preferences" else 5,
                    )
                    memories[key] = results if results else []
                except Exception as e:
                    logger.warning(f"Failed to retrieve {key} from namespace {namespace}: {e}")
                    memories[key] = []

            # Format context
            if any(memories.values()):
                logger.info(
                    f"Retrieved long-term memory: preferences={len(memories['preferences'])}, "
                    f"facts={len(memories['facts'])}, summaries={len(memories['summaries'])}"
                )
                return self._format_memory_context(
                    preferences=memories["preferences"],
                    facts=memories["facts"],
                    summaries=memories["summaries"],
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to retrieve long-term memory context: {e}")
            return None

    @staticmethod
    def _format_memory_context(
        preferences: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
    ) -> str:
        """Format long-term memory into context string.

        Args:
            preferences: User preference records
            facts: Bill-related fact records
            summaries: Session summary records

        Returns:
            Formatted context string
        """
        parts: list[str] = []

        if preferences:
            pref_texts = [AskBillAgent._extract_text_from_record(p) for p in preferences[:3]]
            parts.append(f"User preferences: {'; '.join(pref_texts)}")

        if facts:
            fact_texts = [AskBillAgent._extract_text_from_record(f) for f in facts[:5]]
            parts.append(f"Recent bill facts: {'; '.join(fact_texts)}")

        if summaries:
            summary_texts = [AskBillAgent._extract_text_from_record(s) for s in summaries[:3]]
            parts.append(f"Previous sessions: {'; '.join(summary_texts)}")

        return " | ".join(parts) if parts else ""

    @staticmethod
    def _extract_text_from_record(record: dict[str, Any], max_length: int = 100) -> str:
        """Extract and truncate text from a memory record.

        Args:
            record: Memory record dictionary
            max_length: Maximum text length

        Returns:
            Truncated text snippet
        """
        content = record.get("content") or record.get("text") or record.get("value")

        if isinstance(content, dict):
            text = content.get("text") or json.dumps(content, ensure_ascii=False)
        elif isinstance(content, str):
            text = content
        else:
            text = json.dumps(record, ensure_ascii=False)

        snippet = text.strip()
        if len(snippet) > max_length:
            snippet = f"{snippet[:max_length].rstrip()}..."

        return snippet

    # =========================================================================
    # LLM CALL WITH CIRCUIT BREAKER
    # =========================================================================

    async def _call_llm_with_circuit_protection(
        self, messages: list[BaseMessage]
    ) -> BaseMessage:
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

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process a user's billing question through the complete agent workflow.

        This is the main entry point for handling user requests. It:
        1. Validates and extracts inputs
        2. Executes the LangGraph workflow (with checkpointing if enabled)
        3. Returns formatted response

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
        from ask_bill.core.models.state import AskBillAgentState

        session_id = payload.get("session_id") or str(uuid.uuid4())

        try:
            # =================================================================
            # Step 1: Extract and validate inputs
            # =================================================================
            user_id, session_id, message, file_path = self._extract_and_validate_inputs(payload)

            logger.info(f"Processing request: session={session_id}, user={user_id}")

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
            # Step 3: Invoke workflow with checkpointing config
            # =================================================================
            if self.checkpointer:
                config = {
                    "configurable": {
                        "thread_id": session_id,
                        "checkpoint_ns": user_id,
                    }
                }
                logger.info(
                    f"Invoking workflow WITH checkpointing: thread_id={session_id}, "
                    f"checkpoint_ns={user_id}"
                )
                result = await self.workflow.ainvoke(initial_state, config=config)
            else:
                logger.info("Invoking workflow WITHOUT checkpointing")
                result = await self.workflow.ainvoke(initial_state)

            # =================================================================
            # Step 4: Extract response from result
            # =================================================================
            if isinstance(result, dict):
                # Checkpointing enabled - result is a dict
                messages = result.get("messages", [])
                bill_json = result.get("bill_json")
            else:
                # Checkpointing disabled - result is AskBillAgentState
                messages = result.messages
                bill_json = result.bill_json

            response_content = (
                _as_text(cast(StrOrListStrOrDict, messages[-1].content))
                if messages
                else "No response"
            )

            logger.info(f"Workflow completed: response_length={len(response_content)}")

            # =================================================================
            # Step 5: Return formatted response
            # =================================================================
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

    def _extract_and_validate_inputs(
        self, payload: dict[str, Any]
    ) -> tuple[str, str, str, str | None]:
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
        validated_session_id = (
            self.validator.validate_session_id(session_id) if session_id else session_id
        )

        return validated_user_id, validated_session_id, validated_message, file_path
