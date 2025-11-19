"""Example usage of the updated Ask Bill Agent with ChatBedrockConverse and AgentCoreMemorySaver."""

import asyncio
import logging
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Example configuration structure (adjust to match your actual config)
class MemoryConfig:
    """Memory configuration for Ask Bill Agent."""

    def __init__(
        self,
        memory_id: str,
        checkpointing_enabled: bool = True,
        short_term_enabled: bool = True,
        long_term_enabled: bool = True,
    ):
        self.memory_id = memory_id
        self.checkpointing_enabled = checkpointing_enabled
        self.short_term_enabled = short_term_enabled
        self.long_term_enabled = long_term_enabled


class AWSRegion:
    """AWS Region enum."""

    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"


class AskBillConfig:
    """Configuration for Ask Bill Agent."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-west-2",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_id: str | None = None,
    ):
        self.model_id = model_id
        self.region = AWSRegion()
        self.region.value = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_config = MemoryConfig(
            memory_id=memory_id or "your-bedrock-memory-id",
            checkpointing_enabled=True,
            short_term_enabled=True,
            long_term_enabled=True,
        )


async def example_basic_usage():
    """Example 1: Basic usage without bill processing."""
    from ask_bill_agent_updated import AskBillAgent

    # Initialize agent with configuration
    config = AskBillConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-west-2",
        temperature=0.7,
        max_tokens=4096,
        memory_id="your-bedrock-memory-id",  # Replace with actual memory ID
    )

    agent = AskBillAgent(config)

    # Process a simple query
    payload = {
        "message": "What are some common items on a utility bill?",
        "user_id": "user_123",
        "session_id": "session_001",
    }

    response = await agent.invoke(payload)

    print("\n=== Basic Query Response ===")
    print(f"Session ID: {response['session_id']}")
    print(f"User ID: {response['user_id']}")
    print(f"Response: {response['response']}")
    print(f"Bill JSON: {response['bill_json']}")


async def example_with_bill_processing():
    """Example 2: Query with bill PDF processing."""
    from ask_bill_agent_updated import AskBillAgent

    config = AskBillConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-west-2",
        memory_id="your-bedrock-memory-id",
    )

    agent = AskBillAgent(config)

    # Process query with bill PDF
    payload = {
        "message": "How much is my electricity bill this month?",
        "user_id": "user_123",
        "session_id": "session_002",
        "file_path": "s3://my-bucket/bills/electric_bill_202501.pdf",
    }

    response = await agent.invoke(payload)

    print("\n=== Bill Processing Response ===")
    print(f"Session ID: {response['session_id']}")
    print(f"User ID: {response['user_id']}")
    print(f"Response: {response['response']}")
    print(f"Bill JSON: {response['bill_json']}")


async def example_multi_turn_conversation():
    """Example 3: Multi-turn conversation with memory."""
    from ask_bill_agent_updated import AskBillAgent

    config = AskBillConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-west-2",
        memory_id="your-bedrock-memory-id",
    )

    agent = AskBillAgent(config)

    session_id = "session_003"
    user_id = "user_123"

    # Turn 1: Ask about bill
    print("\n=== Turn 1 ===")
    response1 = await agent.invoke(
        {
            "message": "What's my current electricity usage?",
            "user_id": user_id,
            "session_id": session_id,
            "file_path": "s3://my-bucket/bills/electric_bill.pdf",
        }
    )
    print(f"Assistant: {response1['response']}")

    # Turn 2: Follow-up question (uses short-term memory)
    print("\n=== Turn 2 ===")
    response2 = await agent.invoke(
        {
            "message": "How does that compare to last month?",
            "user_id": user_id,
            "session_id": session_id,
        }
    )
    print(f"Assistant: {response2['response']}")

    # Turn 3: Another follow-up
    print("\n=== Turn 3 ===")
    response3 = await agent.invoke(
        {
            "message": "What can I do to reduce my bill?",
            "user_id": user_id,
            "session_id": session_id,
        }
    )
    print(f"Assistant: {response3['response']}")


async def example_checkpoint_resume():
    """Example 4: Resume from checkpoint after interruption."""
    from ask_bill_agent_updated import AskBillAgent

    config = AskBillConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-west-2",
        memory_id="your-bedrock-memory-id",
    )

    agent = AskBillAgent(config)

    session_id = "session_004"
    user_id = "user_123"

    # Initial request (may be interrupted)
    print("\n=== Initial Request ===")
    try:
        response = await agent.invoke(
            {
                "message": "Analyze my bill and suggest savings",
                "user_id": user_id,
                "session_id": session_id,
                "file_path": "s3://my-bucket/bills/electric_bill.pdf",
            }
        )
        print(f"Assistant: {response['response']}")
    except Exception as e:
        print(f"Request interrupted: {e}")

    # Resume with same session_id (checkpointer will restore state)
    print("\n=== Resumed Request ===")
    response_resumed = await agent.invoke(
        {
            "message": "Continue with the analysis",
            "user_id": user_id,
            "session_id": session_id,  # Same session ID
        }
    )
    print(f"Assistant: {response_resumed['response']}")


async def example_streaming_usage():
    """Example 5: Streaming responses (future enhancement)."""
    print("\n=== Streaming Example ===")
    print("Note: Streaming support can be added by modifying the agent to use astream():")
    print("""
    async def stream_response(self, payload: dict[str, Any]):
        # Build messages...
        async for chunk in self.llm.astream(conversation_messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
    """)


async def example_token_tracking():
    """Example 6: Token usage tracking."""
    from ask_bill_agent_updated import AskBillAgent

    config = AskBillConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-west-2",
        memory_id="your-bedrock-memory-id",
    )

    agent = AskBillAgent(config)

    # Enable DEBUG logging to see token usage
    logging.getLogger().setLevel(logging.DEBUG)

    print("\n=== Token Tracking Example ===")
    response = await agent.invoke(
        {
            "message": "What's the due date on my bill?",
            "user_id": "user_123",
            "session_id": "session_005",
            "file_path": "s3://my-bucket/bills/electric_bill.pdf",
        }
    )

    print("Check logs above for token usage information:")
    print("  - Input tokens")
    print("  - Output tokens")
    print("  - Total tokens")


async def example_error_handling():
    """Example 7: Error handling and circuit breaker."""
    from ask_bill_agent_updated import AskBillAgent

    config = AskBillConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-west-2",
        memory_id="your-bedrock-memory-id",
    )

    agent = AskBillAgent(config)

    print("\n=== Error Handling Example ===")
    try:
        # Invalid file path
        response = await agent.invoke(
            {
                "message": "What's my bill total?",
                "user_id": "user_123",
                "session_id": "session_006",
                "file_path": "invalid://path/to/bill.pdf",
            }
        )
        print(f"Response: {response['response']}")
    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {e}")

    print("\nCircuit breaker will open after 3 consecutive failures")
    print("and remain open for 30 seconds before retrying.")


def main():
    """Run all examples."""
    print("=" * 80)
    print("Ask Bill Agent - ChatBedrockConverse & AgentCoreMemorySaver Examples")
    print("=" * 80)

    # Choose which example to run
    examples = {
        "1": ("Basic Usage", example_basic_usage),
        "2": ("Bill Processing", example_with_bill_processing),
        "3": ("Multi-turn Conversation", example_multi_turn_conversation),
        "4": ("Checkpoint Resume", example_checkpoint_resume),
        "5": ("Streaming (future)", example_streaming_usage),
        "6": ("Token Tracking", example_token_tracking),
        "7": ("Error Handling", example_error_handling),
    }

    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("\nNote: Update 'your-bedrock-memory-id' with your actual Bedrock AgentCore Memory ID")
    print("      before running these examples.\n")

    # Run example 1 by default (or modify to run all)
    example_name, example_func = examples["1"]
    print(f"\nRunning Example: {example_name}\n")

    try:
        asyncio.run(example_func())
    except Exception as e:
        print(f"\nExample failed: {type(e).__name__}: {e}")
        print("\nMake sure to:")
        print("  1. Set valid memory_id in config")
        print("  2. Configure AWS credentials")
        print("  3. Have required dependencies installed")


if __name__ == "__main__":
    main()
