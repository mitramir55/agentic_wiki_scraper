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
    user_selected_option: str

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
        logger.info(f"[{request_id}] Topic extracted: {topic_extraction.topic}")
        
        # Store query in database
        logger.info(f"[{request_id}] Storing query in database...")
        db_query = models.Query(
            original_query=request.query,
            extracted_topic=topic_extraction.topic
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        logger.info(f"[{request_id}] Query stored with ID: {db_query.id}")
        
        # 2. Search Wikipedia
        logger.info(f"[{request_id}] Searching Wikipedia for topic: {topic_extraction.topic}")
        search_results = await wikipedia_searcher.search(topic_extraction.topic)
        
        if not search_results:
            logger.info(f"[{request_id}] No Wikipedia results found, prompting for clearer information")
            return {
                "status": "needs_clarification",
                "message": "I couldn't find any Wikipedia articles matching your query. Could you please provide more specific information about what you're looking for?",
                "query_id": db_query.id,
                "original_query": request.query,
                "extracted_topic": topic_extraction.topic,
                "agent_info": {
                    "name": "WikipediaSearcher",
                    "status": "no_results",
                    "current_operation": "search",
                    "request_id": request_id
                }
            }
        
        # Get multiple search results (at least 3)
        search_results_list = []
        if isinstance(search_results, list):
            search_results_list = search_results[:3]  # Take first 3 results
        else:
            search_results_list = [search_results]  # Single result case
        
        logger.info(f"[{request_id}] Found {len(search_results_list)} Wikipedia articles")
        
        # Return search results for user confirmation
        return {
            "status": "needs_confirmation",
            "search_results": [
                {
                    "title": result.title,
                    "url": result.url
                } for result in search_results_list
            ],
            "query_id": db_query.id,
            "original_query": request.query,
            "extracted_topic": topic_extraction.topic,
            "agent_info": {
                "name": "WikipediaSearcher",
                "status": "search_complete",
                "current_operation": "search",
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
        elif "wikipedia_searcher" in error_details:
            current_agent = "WikipediaSearcher"
            current_operation = "search"
            
        logger.error(f"[{request_id}] Error in {current_agent} during {current_operation}")
        logger.error(f"[{request_id}] Error type: {error_type}")
        logger.error(f"[{request_id}] Error details: {error_details}")
        
        raise HTTPException(
            status_code=400,
            detail=str(e),
            headers={
                "X-Agent-Info": current_agent,
                "X-Operation": current_operation,
                "X-Request-ID": request_id
            }
        )

@app.post("/api/v1/confirm")
async def confirm_search_result(
    request: DisambiguationRequest,
    db: Session = Depends(get_db)
):
    """Handle user's confirmation of search result or refinement request."""
    try:
        # Get the original query
        db_query = db.query(models.Query).filter(models.Query.id == request.query_id).first()
        if not db_query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # If user wants to refine the search
        if not request.user_selected_option.startswith('http'):
            # Combine the original query with the user's new input
            combined_query = f"{db_query.original_query} {request.user_selected_option}"
            
            # Update both the original query and extracted topic
            db_query.original_query = combined_query
            
            # Extract topic from the combined query
            topic_extraction = await topic_extractor.extract_topic(combined_query)
            db_query.extracted_topic = topic_extraction.topic
            db.commit()
            
            # Do another search
            search_results = await wikipedia_searcher.search(topic_extraction.topic)
            
            if not search_results:
                logger.info(f"No Wikipedia results found for refined query, prompting for clearer information")
                return {
                    "status": "needs_clarification",
                    "message": "I still couldn't find any Wikipedia articles matching your query. Could you please try rephrasing or providing more specific details about what you're looking for?",
                    "query_id": db_query.id,
                    "original_query": combined_query,
                    "extracted_topic": topic_extraction.topic,
                    "agent_info": {
                        "name": "WikipediaSearcher",
                        "status": "no_results",
                        "current_operation": "search"
                    }
                }
            
            # Get multiple search results (at least 3)
            search_results_list = []
            if isinstance(search_results, list):
                search_results_list = search_results[:3]  # Take first 3 results
            else:
                search_results_list = [search_results]  # Single result case
            
            return {
                "status": "needs_confirmation",
                "search_results": [
                    {
                        "title": result.title,
                        "url": result.url
                    } for result in search_results_list
                ],
                "query_id": db_query.id,
                "original_query": combined_query,
                "extracted_topic": topic_extraction.topic,
                "agent_info": {
                    "name": "WikipediaSearcher",
                    "status": "search_complete",
                    "current_operation": "search"
                }
            }
        
        # If user selected a URL, proceed with scraping and summarization
        search_results = await wikipedia_searcher.search(db_query.extracted_topic)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No results found")
        
        # Get content and summarize
        content = await wikipedia_searcher.get_full_content(request.user_selected_option)
        
        if not content:
            raise HTTPException(status_code=404, detail="Could not retrieve content")
        
        summary = await summarizer.summarize(content)
        
        # Store the result
        db_result = models.SearchResult(
            query_id=db_query.id,
            wikipedia_url=request.user_selected_option,
            title=search_results.title,
            content=content,
            summary=summary.summary
        )
        db.add(db_result)
        db.commit()
        
        return {
            "status": "success",
            "title": search_results.title,
            "url": request.user_selected_option,
            "summary": summary.summary,
            "selected_topic": db_query.extracted_topic,
            "agent_info": {
                "name": "Summarizer",
                "status": "completed",
                "current_operation": "summarization"
            }
        }
        
    except Exception as e:
        current_agent = "Unknown"
        current_operation = "Unknown"
        
        if "topic_extractor" in str(e):
            current_agent = "TopicExtractor"
            current_operation = "topic_extraction"
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

@app.get("/api/v1/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint that verifies the application's health status.
    Checks database connectivity and returns the status of core services.
    """
    try:
        # Test database connection
        db.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "services": {
                "database": "connected",
                "api": "operational"
            },
            "version": settings.VERSION
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 