from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class Summary(BaseModel):
    summary: str = Field(description="A concise summary of the content")

class Summarizer:
    def __init__(self):
        logger.info("Initializing Summarizer agent")
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize text splitter with smaller chunks
        logger.info("Configuring text splitter with chunk_size=2000, chunk_overlap=100")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Reduced from 4000 to 2000
            chunk_overlap=100,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        
        # Create map prompt template
        self.map_prompt = PromptTemplate(
            template="""Write a concise summary of the following text. The summary should:
            1. Be at most 100 words
            2. Capture the main points and key information
            3. Be clear and easy to understand
            4. Maintain the most important details
            5. Be written in a neutral, informative tone

            Text to summarize:
            {text}

            Summary:""",
            input_variables=["text"]
        )
        
        # Create combine prompt template
        self.combine_prompt = PromptTemplate(
            template="""Write a concise summary of the following text. The summary should:
            1. Make sure the summary is at most 300 words. 
            2. Capture the main points and key information
            3. Be clear and easy to understand
            4. Maintain the most important details
            5. Be written in a neutral, informative tone

            Text to summarize:
            {text}

            Summary:""",
            input_variables=["text"]
        )
        
        logger.info("Summarizer initialization complete")

    async def summarize(self, content: str) -> Summary:
        """Generate a concise summary of the given content."""
        logger.info(f"Starting summarization of content (length: {len(content)} characters)")
        logger.info(f"Content preview: {content[:100]}...")
        
        try:
            # Split content into documents
            logger.info("Splitting content into chunks...")
            docs = [Document(page_content=content)]
            split_docs = self.text_splitter.split_documents(docs)
            logger.info(f"Content split into {len(split_docs)} chunks")
            
            # Log chunk sizes
            for i, doc in enumerate(split_docs):
                logger.info(f"Chunk {i+1} size: {len(doc.page_content)} characters")
            
            # when we have too many chunks map reduce can help
            logger.info("Using map_reduce chain")
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                verbose=True,  # Enable verbose logging
                map_prompt=self.map_prompt,
                combine_prompt=self.combine_prompt,
                return_intermediate_steps=True,  # Get intermediate summaries
                combine_document_variable_name="text"  # Specify the variable name for combine prompt
            )
            
            # Use ainvoke instead of arun for map_reduce chain
            result = await chain.ainvoke({"input_documents": split_docs})
            
            
            # Extract the summary text from the result
            if isinstance(result, dict):
                if 'output_text' in result:
                    summary_text = result['output_text']
                elif 'text' in result:
                    summary_text = result['text']
                else:
                    # If we can't find the summary in the expected keys, try to get it from the first key
                    summary_text = next(iter(result.values()))
            else:
                summary_text = str(result)
                
            logger.info(f"Generated summary (length: {len(summary_text)} characters)")
            
            # Ensure summary is not longer than 100 words
            words = summary_text.split()
            if len(words) > 300:
                logger.info(f"Truncating summary from {len(words)} to 300 words")
                summary_text = ' '.join(words[:300]) + '...'
            
            logger.info("Summarization completed successfully")
            return Summary(summary=summary_text)
            
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            if "context_length_exceeded" in str(e):
                logger.info("Context length exceeded, retrying with smaller chunks")
                # If we hit context length error, try with even smaller chunks
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Further reduce chunk size
                    chunk_overlap=50,  # Reduce overlap
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?"]
                )
                logger.info("Retrying summarization with smaller chunks")
                # Retry with smaller chunks
                return await self.summarize(content)
            raise 