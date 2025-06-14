from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import settings

class TopicExtraction(BaseModel):
    topic: str = Field(description="The main topic extracted from the query")

class TopicExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=TopicExtraction)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a topic extraction expert. Your task is to extract the main topic from the user's query.
            Focus on identifying the core subject or concept that the user is interested in.
            If the query is unclear or could refer to multiple topics, extract the most likely topic based on the context.
            
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