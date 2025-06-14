import pytest
from fastapi.testclient import TestClient
from app.core.config import settings

@pytest.mark.asyncio
async def test_home_page(client: TestClient):
    """Test the home page endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_health_check(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    if response.status_code != 200:
        error_data = response.json()
        print(f"Error response: {error_data}")
        print(f"Headers: {response.headers}")
        assert False, f"Health check failed with status {response.status_code}: {error_data.get('detail', 'No error detail')}"
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data
    assert data["services"]["database"] == "connected"
    assert data["services"]["api"] == "operational"
    assert "version" in data

@pytest.mark.asyncio
async def test_process_query(client: TestClient):
    """Test the query processing endpoint."""
    response = client.post(
        "/api/v1/process",
        json={"query": "Tell me about artificial intelligence"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "query_id" in data
    assert "original_query" in data
    assert "extracted_topic" in data

@pytest.mark.asyncio
async def test_confirm_search_result(client: TestClient):
    """Test the search result confirmation endpoint."""
    # First create a query
    process_response = client.post(
        "/api/v1/process",
        json={"query": "Tell me about artificial intelligence"}
    )
    assert process_response.status_code == 200, "Failed to process query"
    process_data = process_response.json()
    query_id = process_data["query_id"]
    
    # Then confirm a search result
    response = client.post(
        "/api/v1/confirm",
        json={
            "query_id": query_id,
            "user_selected_option": "https://en.wikipedia.org/wiki/Artificial_intelligence"
        }
    )
    
    # Add detailed error logging
    if response.status_code != 200:
        error_data = response.json()
        print(f"Error response: {error_data}")
        print(f"Headers: {response.headers}")
        assert False, f"Confirm endpoint failed with status {response.status_code}: {error_data.get('detail', 'No error detail')}"
    
    data = response.json()
    assert data["status"] == "success"
    assert "title" in data
    assert "url" in data
    assert "summary" in data
    assert "selected_topic" in data
    assert "agent_info" in data
    assert data["agent_info"]["name"] == "Summarizer"
    assert data["agent_info"]["status"] == "completed"

@pytest.mark.asyncio
async def test_get_queries(client: TestClient):
    """Test getting all queries."""
    response = client.get("/api/v1/queries")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_get_query_by_id(client: TestClient):
    """Test getting a specific query by ID."""
    # First create a query
    process_response = client.post(
        "/api/v1/process",
        json={"query": "Tell me about artificial intelligence"}
    )
    assert process_response.status_code == 200
    query_id = process_response.json()["query_id"]
    
    # Then get the query
    response = client.get(f"/api/v1/query/{query_id}")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert data["query"]["id"] == query_id
    assert "original_query" in data["query"]
    assert "extracted_topic" in data["query"] 