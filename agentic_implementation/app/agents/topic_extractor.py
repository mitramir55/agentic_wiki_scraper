from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import settings

class TopicExtraction(BaseModel):
    topic: str = Field(description="The main topic extracted from the query")
    confidence: float = Field(description="Confidence score of the extraction (0-1)")
    is_ambiguous: bool = Field(description="Whether the topic is ambiguous and needs disambiguation")

class TopicExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=TopicExtraction)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a topic extraction expert. Your task is to:
            1. Extract the main topic from the user's query
            2. Determine if the topic is ambiguous
            3. If the query is ambiguous, set the confidence score to LESS THAN 0.7 — even if one interpretation seems dominant./
            As an example, if the query contains a single name like "meryl" or "washington" or "python" or any other name that can refer to multiple people orthings,/
            make sure to set the confidence score to less than 0.7. Also if the topic is not clearly defined, set the confidence score to less than 0.7.
             These should have low confidence scores, such as 0.4 or 0.5, regardless of how commonly known one interpretation may be.
             
            {format_instructions}"""),
            ("user", "{query}")
        ])

    async def extract_topic(self, query: str) -> TopicExtraction:
        """Extract the main topic from a user query."""
        chain = self.prompt | self.llm | self.parser
        
        result = await chain.ainvoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result 