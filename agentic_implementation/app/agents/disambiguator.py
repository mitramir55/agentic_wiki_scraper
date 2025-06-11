from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from app.core.config import settings

class DisambiguationOption(BaseModel):
    topic: str = Field(description="The disambiguated topic")
    description: str = Field(description="Brief description of this topic")

class DisambiguationResult(BaseModel):
    options: List[DisambiguationOption] = Field(description="List of possible topics")
    conversation_prompt: str = Field(description="The question to ask the user for clarification")

class Disambiguator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=DisambiguationResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a topic disambiguation expert. Your task is to:
            1. Generate a list of possible topics that match the ambiguous query
            2. Provide a brief description for each option
            3. Provide a specific question to ask the user

            Examples:
            - For "Meryl", you should list different people named Meryl (e.g., Meryl Streep, Meryl Davis)
            - For "Washington", you should list different meanings (e.g., George Washington, Washington state, Washington DC)
            - For "Python", you should list different meanings (e.g., Python programming language, Python snake)

            Provide a clear, specific question that helps narrow down the user's intent.
            
            {format_instructions}"""),
            ("user", "Ambiguous topic: {topic}\nContext: {context}")
        ])

    async def get_disambiguation_options(self, topic: str, context: str = "") -> DisambiguationResult:
        """Generate disambiguation options for an ambiguous topic."""
        chain = self.prompt | self.llm | self.parser
        
        result = await chain.ainvoke({
            "topic": topic,
            "context": context,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result

    async def select_option(self, options: DisambiguationResult, user_input: str) -> Optional[DisambiguationOption]:
        """Select the best matching option based on user input."""
        selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """Select the best matching option from the list based on the user's input.
            Return only the index of the selected option (0-based)."""),
            ("user", "Options:\n{options}\nUser input: {user_input}")
        ])
        
        options_text = "\n".join([
            f"{i}. {opt.topic} - {opt.description}"
            for i, opt in enumerate(options.options)
        ])
        
        chain = selection_prompt | self.llm
        
        result = await chain.ainvoke({
            "options": options_text,
            "user_input": user_input
        })
        
        try:
            selected_index = int(result.content.strip())
            if 0 <= selected_index < len(options.options):
                return options.options[selected_index]
        except (ValueError, IndexError):
            pass
        
        return None 