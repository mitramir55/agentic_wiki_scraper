# Agentic Web Content Analysis System

A sophisticated multi-agent system for intelligent web content extraction and analysis. This system uses a series of specialized agents to process user queries, extract relevant content, and provide meaningful summaries.

## Features

- **Multi-Agent Architecture**: Implements a chain of specialized agents for different tasks
- **Intelligent Topic Extraction**: Uses AI to understand and extract topics from user queries
- **Smart Disambiguation**: Handles ambiguous queries through interactive chat
- **Wikipedia Integration**: Seamless Wikipedia content search and extraction
- **Content Analysis**: Advanced content scraping and summarization
- **Modern Web Interface**: Clean and responsive UI with real-time updates

## Agent Workflow

1. **Topic Extraction Agent**: Analyzes user queries to identify main topics
2. **Disambiguation Agent**: Resolves ambiguous queries through chat interaction
3. **Wikipedia Search Agent**: Searches Wikipedia for relevant content
4. **Alternative Search Agent**: Provides fallback search using wikipedia package
5. **Content Scraping Agent**: Extracts content from identified pages
6. **Summarization Agent**: Creates concise summaries of scraped content
7. **Result Delivery Agent**: Presents results in a user-friendly format

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: LangChain, OpenAI GPT-3.5-turbo
- **Database**: PostgreSQL
- **Web Scraping**: BeautifulSoup4, aiohttp
- **Wikipedia Integration**: wikipedia package

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- PostgreSQL database

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd agentic_implementation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   OPENAI_API_KEY=your_api_key_here
   DATABASE_URL=postgresql://user:password@localhost:5432/dbname
   ```

5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Project Structure

```
app/
├── agents/
│   ├── topic_extractor.py
│   ├── disambiguator.py
│   ├── wikipedia_search.py
│   ├── content_scraper.py
│   ├── summarizer.py
│   └── result_delivery.py
├── core/
│   ├── config.py
│   └── security.py
├── db/
│   ├── database.py
│   └── models.py
├── static/
├── templates/
└── main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. 