import wikipedia
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from app.core.config import settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import logging

logger = logging.getLogger(__name__)

class WikipediaSearchResult(BaseModel):
    title: str = Field(description="The title of the Wikipedia article")
    summary: str = Field(description="A concise summary of the article")
    url: str = Field(description="The URL of the Wikipedia article")

class WikipediaSearcher:
    def __init__(self):
        wikipedia.set_lang(settings.WIKIPEDIA_LANGUAGE)
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=WikipediaSearchResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Wikipedia search expert. Your task is to:
            1. Search for the most relevant Wikipedia article for the given topic
            2. Provide a concise summary of the article
            3. Include the article URL
            
            You must return a JSON object with the following fields:
            - title: The title of the Wikipedia article
            - summary: A concise summary of the article
            - url: The URL of the Wikipedia article
            
            {format_instructions}"""),
            ("user", "Topic: {topic}")
        ])

    async def search(self, topic: str) -> WikipediaSearchResult:
        """Search Wikipedia for information about a topic."""
        try:
            # First try direct Wikipedia search
            try:
                page = wikipedia.page(topic, auto_suggest=False)
                return WikipediaSearchResult(
                    title=page.title,
                    summary=page.summary,
                    url=page.url
                )
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation page, use the first option
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return WikipediaSearchResult(
                    title=page.title,
                    summary=page.summary,
                    url=page.url
                )
            except wikipedia.exceptions.PageError:
                # If page not found, try fuzzy search
                results = await self._fuzzy_search(topic)
                if results:
                    return results[0]  # Return the first result
                raise Exception(f"No Wikipedia article found for topic: {topic}")
        except Exception as e:
            # If all else fails, use LLM to generate a response
            try:
                chain = self.prompt | self.llm | self.parser
                result = await chain.ainvoke({
                    "topic": topic,
                    "format_instructions": self.parser.get_format_instructions()
                })
                return result
            except Exception as llm_error:
                raise Exception(f"Failed to search Wikipedia: {str(e)}. LLM fallback also failed: {str(llm_error)}")

    async def _fuzzy_search(self, topic: str) -> List[WikipediaSearchResult]:
        """Perform a fuzzy search when exact search fails."""
        try:
            # Try searching with different variations
            variations = [
                topic,
                topic.replace(" ", "_"),
                topic.lower(),
                topic.title()
            ]
            
            results = []
            for variation in variations:
                try:
                    search_results = wikipedia.search(variation, results=settings.WIKIPEDIA_MAX_RESULTS)
                    for title in search_results:
                        try:
                            page = wikipedia.page(title, auto_suggest=False)
                            results.append(WikipediaSearchResult(
                                title=page.title,
                                url=page.url,
                                summary=page.summary
                            ))
                        except:
                            continue
                except:
                    continue
            
            return results
        except Exception as e:
            raise Exception(f"Fuzzy search failed: {str(e)}")

    async def get_full_content(self, url: str) -> Optional[str]:
        """Retrieve the full content of a Wikipedia page."""
        try:
            # Extract title from URL
            title = url.split("/")[-1]
            page = wikipedia.page(title, auto_suggest=False)
            # Return just the content as a string, not a dictionary
            return str(page.content)
        except Exception as e:
            logger.error(f"Error getting full content: {str(e)}")
            return None 