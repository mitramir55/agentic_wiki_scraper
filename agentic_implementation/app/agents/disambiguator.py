from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class DisambiguationOption(BaseModel):
    topic: str = Field(description="The disambiguated topic")

class DisambiguationResult(BaseModel):
    options: List[DisambiguationOption] = Field(description="List of possible topics.")
    conversation_prompt: str = Field(description="The question to ask the user for clarification")

class TopicSelection(BaseModel):
    disambiguated_topic: str = Field(description="The topic selected by the user or extracted from their input")

class Disambiguator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Create parser with explicit format instructions
        self.parser = PydanticOutputParser(pydantic_object=DisambiguationResult)
        self.selection_parser = PydanticOutputParser(pydantic_object=TopicSelection)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a topic disambiguation expert. Your task is to:
            1. Generate a list of possible topics that match the ambiguous query
            2. Provide a specific question to ask the user

            Provide a clear, specific question that helps narrow down the user's intent.

            {format_instructions}"""),
            ("user", """Ambiguous topic: {topic}
Context: {context}""")
        ])

        self.selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a topic selection expert. Your task is to:
            1. Analyze the user's input in relation to the provided options
            2. Either select the most relevant option or extract a new topic from the user's input
            3. Determine if we need to generate more options based on the user's input

            If the user's input clearly matches one of the options, select that option.
            If the user's input suggests a different topic, extract that topic.

            {format_instructions}"""),
            ("user", """Options:
            {options}

            User input: {user_input}""")
        ])

    async def get_disambiguation_options(self, topic: str, context: str = "") -> DisambiguationResult:
        """Generate disambiguation options for an ambiguous topic."""
        chain = self.prompt | self.llm | self.parser
        
        format_instructions = self.parser.get_format_instructions()

        result = await chain.ainvoke({
            "topic": topic,
            "context": context,
            "format_instructions": format_instructions
        })
        
        logger.info(f"Result of the disambiguation for options: {result}")
        return result

    async def select_option(self, disambiguation_result: Optional[DisambiguationResult], user_input: str) -> TopicSelection:
        """Select the best matching option or extract a new topic from user input."""
        chain = self.selection_prompt | self.llm | self.selection_parser
        
        if disambiguation_result is None:
            # If no previous options, just extract the topic from user input
            options_text = "No previous options. Please extract the topic from the user input."
        else:
            options_text = "\n".join([
                f"{i}. {opt.topic}"
                for i, opt in enumerate(disambiguation_result.options)
            ])
        
        selection_format = self.selection_parser.get_format_instructions()
        
        result = await chain.ainvoke({
            "options": options_text,
            "user_input": user_input,
            "format_instructions": selection_format
        })
        
        logger.info(f"Topic selection result: {result}")
        return result 