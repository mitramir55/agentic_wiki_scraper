from unittest.mock import AsyncMock, MagicMock
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from app.core.config import settings

class MockChatOpenAI(ChatOpenAI):
    """Mock ChatOpenAI for testing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ainvoke = AsyncMock()
        self.ainvoke.return_value = AIMessage(
            content='{"topic": "artificial intelligence"}'
        )

def get_mock_llm():
    """Get a mock LLM for testing."""
    return MockChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY
    ) 