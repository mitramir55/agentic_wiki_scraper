from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from app.core.config import settings

class DisambiguationOption(BaseModel):
    topic: str = Field(description="The disambiguated topic")
    description: str = Field(description="Brief description of this topic")
    confidence: float = Field(description="Confidence score for this option (0-1)")

class DisambiguationResult(BaseModel):
    options: List[DisambiguationOption] = Field(description="List of possible topics")
    selected_option: Optional[DisambiguationOption] = Field(description="The selected topic option")
    needs_conversation: bool = Field(description="Whether the agent needs to have a conversation with the user")
    conversation_prompt: Optional[str] = Field(description="The question to ask the user for clarification")

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
            3. Assign a confidence score to each option
            4. Determine if a conversation with the user is needed
            5. If conversation is needed, provide a specific question to ask the user
            
            If the options are very different from each other or if the user's intent is unclear,
            set needs_conversation to true and provide a specific question to ask.
            
            {format_instructions}"""),
            ("user", "Ambiguous topic: {topic}\nContext: {context}")
        ])

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant trying to understand the user's intent.
            Based on the user's response to your question, select the most appropriate topic
            from the given options or ask a follow-up question if needed.
            
            Format your response as:
            Selected Option: [index of the option, or -1 if need more clarification]
            Follow-up Question: [your question, or empty if no more questions needed]"""),
            ("user", """Original topic: {topic}
            Options:
            {options}
            Your question: {question}
            User's response: {user_response}""")
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

    async def handle_conversation(self, 
                                topic: str, 
                                options: List[DisambiguationOption], 
                                question: str, 
                                user_response: str) -> Dict:
        """Handle the user's response in the conversation."""
        options_text = "\n".join([
            f"{i}. {opt.topic} - {opt.description} (Confidence: {opt.confidence})"
            for i, opt in enumerate(options)
        ])
        
        chain = self.conversation_prompt | self.llm
        
        result = await chain.ainvoke({
            "topic": topic,
            "options": options_text,
            "question": question,
            "user_response": user_response
        })
        
        # Parse the response
        lines = result.content.strip().split("\n")
        selected_index = -1
        follow_up_question = ""
        
        for line in lines:
            if line.startswith("Selected Option:"):
                try:
                    selected_index = int(line.split(":")[1].strip())
                except:
                    selected_index = -1
            elif line.startswith("Follow-up Question:"):
                follow_up_question = line.split(":")[1].strip()
        
        if selected_index >= 0 and selected_index < len(options):
            return {
                "selected_option": options[selected_index],
                "needs_conversation": False,
                "conversation_prompt": None
            }
        else:
            return {
                "selected_option": None,
                "needs_conversation": True,
                "conversation_prompt": follow_up_question
            }

    async def select_option(self, options: DisambiguationResult, user_input: str) -> DisambiguationOption:
        """Select the best matching option based on user input."""
        if options.needs_conversation:
            # If conversation is needed, handle it
            result = await self.handle_conversation(
                options.options[0].topic,  # Use the first option's topic as reference
                options.options,
                options.conversation_prompt,
                user_input
            )
            
            if result["selected_option"]:
                return result["selected_option"]
            else:
                # If still need conversation, return None to indicate more interaction needed
                return None
        else:
            # If no conversation needed, proceed with direct selection
            selection_prompt = ChatPromptTemplate.from_messages([
                ("system", """Select the best matching option from the list based on the user's input.
                Return only the index of the selected option (0-based)."""),
                ("user", "Options:\n{options}\nUser input: {user_input}")
            ])
            
            options_text = "\n".join([
                f"{i}. {opt.topic} - {opt.description} (Confidence: {opt.confidence})"
                for i, opt in enumerate(options.options)
            ])
            
            chain = selection_prompt | self.llm
            
            result = await chain.ainvoke({
                "options": options_text,
                "user_input": user_input
            })
            
            try:
                selected_index = int(result.content.strip())
                return options.options[selected_index]
            except (ValueError, IndexError):
                # If parsing fails, return the highest confidence option
                return max(options.options, key=lambda x: x.confidence) 