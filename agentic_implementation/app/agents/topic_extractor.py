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
            3. Provide a confidence score. This score should be a number between 0 and 1.
            If the user is asking for a topic like a single name like "Meryl" or "Washington" or "Python" or any other name that can refer to multiple things, set the confidence score to less than 0.7.
             
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