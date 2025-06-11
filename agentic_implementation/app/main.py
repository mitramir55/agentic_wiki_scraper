from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from pydantic import BaseModel
import uvicorn
import os
import logging
import uuid

from app.core.config import settings
from app.db.database import get_db, engine
from app.db import models
from app.agents.topic_extractor import TopicExtractor, TopicExtraction
from app.agents.disambiguator import Disambiguator, DisambiguationResult
from app.agents.wikipedia_search import WikipediaSearcher, WikipediaSearchResult
from app.agents.summarizer import Summarizer, Summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.drop_all(bind=engine)  # Drop existing tables
models.Base.metadata.create_all(bind=engine)  # Create new tables

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class DisambiguationRequest(BaseModel):
    query_id: int
    selected_option: str

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize agents
topic_extractor = TopicExtractor()
disambiguator = Disambiguator()
wikipedia_searcher = WikipediaSearcher()
summarizer = Summarizer()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/v1/process")
async def process_query(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Process a user query through the agent pipeline."""
    request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking
    logger.info(f"[{request_id}] Starting new request processing")
    logger.info(f"[{request_id}] Query: {request.query}")
    
    try:
        # 1. Extract topic
        logger.info(f"[{request_id}] Starting topic extraction...")
        topic_extraction = await topic_extractor.extract_topic(request.query)
        logger.info(f"[{request_id}] Topic extracted: {topic_extraction.topic} (confidence: {topic_extraction.confidence})")
        
        # Store query in database
        logger.info(f"[{request_id}] Storing query in database...")
        db_query = models.Query(
            original_query=request.query,
            extracted_topic=topic_extraction.topic,
            confidence=topic_extraction.confidence
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        logger.info(f"[{request_id}] Query stored with ID: {db_query.id}")
        
        # 2. Check if we need disambiguation based on topic extraction confidence
        if topic_extraction.confidence < 0.7:
            logger.info(f"[{request_id}] Low confidence ({topic_extraction.confidence}), getting disambiguation options...")
            # If confidence is low, get disambiguation options
            disambiguation = await disambiguator.get_disambiguation_options(
                topic_extraction.topic,
                context=request.query
            )
            logger.info(f"[{request_id}] Generated {len(disambiguation.options)} disambiguation options")
            return {
                "status": "needs_disambiguation",
                "options": disambiguation.options,
                "query_id": db_query.id,
                "conversation_prompt": disambiguation.conversation_prompt,
                "agent_info": {
                    "name": "Disambiguator",
                    "status": "processing",
                    "current_operation": "generating_options",
                    "request_id": request_id
                }
            }
        
        # 3. If confidence is high enough, search Wikipedia
        logger.info(f"[{request_id}] Searching Wikipedia for topic: {topic_extraction.topic}")
        search_results = await wikipedia_searcher.search(topic_extraction.topic)
        
        if not search_results:
            logger.error(f"[{request_id}] No Wikipedia results found for topic: {topic_extraction.topic}")
            raise HTTPException(status_code=404, detail="No results found")
        
        logger.info(f"[{request_id}] Found Wikipedia article: {search_results.title}")
        logger.info(f"[{request_id}] Article URL: {search_results.url}")
        
        # 4. Get content and summarize
        logger.info(f"[{request_id}] Getting full content from Wikipedia...")
        content = await wikipedia_searcher.get_full_content(search_results.url)
        
        if not content:
            logger.error(f"[{request_id}] Could not retrieve content from URL: {search_results.url}")
            raise HTTPException(status_code=404, detail="Could not retrieve content")
            
        # Log content length and first 100 characters
        content_length = len(content)
        logger.info(f"[{request_id}] Retrieved content length: {content_length} characters")
        logger.info(f"[{request_id}] Content preview: {content[:100]}...")
        
        logger.info(f"[{request_id}] Starting content summarization...")
        try:
            summary = await summarizer.summarize(content)
            logger.info(f"[{request_id}] Summary generated successfully")
            logger.info(f"[{request_id}] Summary length: {len(summary.summary)} characters")
            logger.info(f"[{request_id}] Summary preview: {summary.summary[:100]}...")
        except Exception as summary_error:
            logger.error(f"[{request_id}] Error during summarization: {str(summary_error)}")
            logger.error(f"[{request_id}] Error type: {type(summary_error).__name__}")
            raise
        
        # Store result in database
        logger.info(f"[{request_id}] Storing search result in database...")
        db_result = models.SearchResult(
            query_id=db_query.id,
            wikipedia_url=search_results.url,
            title=search_results.title,
            content=content,
            summary=summary.summary
        )
        db.add(db_result)
        db.commit()
        logger.info(f"[{request_id}] Search result stored successfully")
        
        return {
            "status": "success",
            "title": search_results.title,
            "url": search_results.url,
            "summary": summary.summary,
            "agent_info": {
                "name": "Summarizer",
                "status": "completed",
                "current_operation": "summarization",
                "content_length": content_length,
                "request_id": request_id
            }
        }
        
    except Exception as e:
        # Get the current agent info
        current_agent = "Unknown"
        current_operation = "Unknown"
        error_details = str(e)
        error_type = type(e).__name__
        
        if "topic_extractor" in error_details:
            current_agent = "TopicExtractor"
            current_operation = "topic_extraction"
        elif "disambiguator" in error_details:
            current_agent = "Disambiguator"
            current_operation = "disambiguation"
        elif "wikipedia_searcher" in error_details:
            current_agent = "WikipediaSearcher"
            current_operation = "search"
        elif "summarizer" in error_details:
            current_agent = "Summarizer"
            current_operation = "summarization"
            
        logger.error(f"[{request_id}] Error in {current_agent} during {current_operation}")
        logger.error(f"[{request_id}] Error type: {error_type}")
        logger.error(f"[{request_id}] Error details: {error_details}")
        
        # Add more detailed error information
        error_info = {
            "error": str(e),
            "error_type": error_type,
            "agent": current_agent,
            "operation": current_operation,
            "request_id": request_id,
            "type": "context_length_exceeded" if "context_length_exceeded" in error_details else "unknown",
            "content_length": content_length if 'content_length' in locals() else None
        }
        
        raise HTTPException(
            status_code=400,
            detail=error_info,
            headers={
                "X-Agent-Info": current_agent,
                "X-Operation": current_operation,
                "X-Request-ID": request_id
            }
        )

@app.post("/api/v1/disambiguate")
async def handle_disambiguation(
    request: DisambiguationRequest,
    db: Session = Depends(get_db)
):
    """Handle user's disambiguation selection or conversation response."""
    try:
        # Get the original query
        db_query = db.query(models.Query).filter(models.Query.id == request.query_id).first()
        if not db_query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Get the disambiguation options
        disambiguation = await disambiguator.get_disambiguation_options(
            db_query.extracted_topic,
            context=db_query.original_query
        )
        
        # Handle the user's response
        selected = await disambiguator.select_option(disambiguation, request.selected_option)
        
        if selected is None:
            # If still need conversation, return the next question
            return {
                "status": "needs_conversation",
                "conversation_prompt": disambiguation.conversation_prompt,
                "query_id": request.query_id,
                "agent_info": {
                    "name": "Disambiguator",
                    "status": "processing",
                    "current_operation": "conversation"
                }
            }
        
        # Search Wikipedia with the selected option
        search_results = await wikipedia_searcher.search(selected.topic)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No results found")
        
        # Get content and summarize
        logger.info("Getting full content...")
        content = await wikipedia_searcher.get_full_content(search_results.url)
        
        if not content:
            raise HTTPException(status_code=404, detail="Could not retrieve content")
        
        logger.info("Summarizing content...")
        summary = await summarizer.summarize(content)
        
        # Store result in database
        db_result = models.SearchResult(
            query_id=db_query.id,
            wikipedia_url=search_results.url,
            title=search_results.title,
            content=content,
            summary=summary.summary
        )
        db.add(db_result)
        db.commit()
        
        return {
            "status": "success",
            "title": search_results.title,
            "url": search_results.url,
            "summary": summary.summary,
            "agent_info": {
                "name": "Summarizer",
                "status": "completed",
                "current_operation": "summarization"
            }
        }
        
    except Exception as e:
        # Get the current agent info
        current_agent = "Unknown"
        current_operation = "Unknown"
        
        if "disambiguator" in str(e):
            current_agent = "Disambiguator"
            current_operation = "disambiguation"
        elif "wikipedia_searcher" in str(e):
            current_agent = "WikipediaSearcher"
            current_operation = "search"
        elif "summarizer" in str(e):
            current_agent = "Summarizer"
            current_operation = "summarization"
            
        logger.error(f"Error in {current_agent} during {current_operation}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
            headers={
                "X-Agent-Info": current_agent,
                "X-Operation": current_operation
            }
        )

@app.get("/api/v1/queries")
async def get_queries(db: Session = Depends(get_db)):
    """Get all saved queries."""
    queries = db.query(models.Query).all()
    return queries

@app.get("/api/v1/results")
async def get_results(db: Session = Depends(get_db)):
    """Get all saved search results."""
    results = db.query(models.SearchResult).all()
    return results

@app.get("/api/v1/query/{query_id}")
async def get_query(query_id: int, db: Session = Depends(get_db)):
    """Get a specific query and its results."""
    query = db.query(models.Query).filter(models.Query.id == query_id).first()
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    results = db.query(models.SearchResult).filter(models.SearchResult.query_id == query_id).all()
    
    return {
        "query": query,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 