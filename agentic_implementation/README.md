# Wikipedia Topic Explorer

A powerful application that uses AI to search, analyze, and summarize Wikipedia articles. Built with FastAPI, LangChain, and PostgreSQL.

## Features

- **AI-Powered Topic Extraction**: Automatically identifies the main topic from user queries
- **Smart Disambiguation**: Handles ambiguous topics with interactive clarification
- **Efficient Summarization**: Uses map-reduce chain for processing large articles
- **Smart Search**: Returns the best matching Wikipedia article for your topic
- **Persistent Storage**: Saves queries and results in PostgreSQL database
- **Real-time Processing**: Asynchronous processing with detailed logging
- **Modern UI**: Clean, responsive interface with real-time updates

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd wikipedia-topic-explorer
```

2. Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
WIKIPEDIA_LANGUAGE=en
```

3. Start the application using Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Enter a topic or question in the search box
3. The system will:
   - Extract the main topic
   - Search Wikipedia
   - Generate a concise summary
   - Display the results

### Example Queries

- "Tell me about artificial intelligence"
- "What is quantum computing?"
- "Explain the history of the internet"

## API Endpoints

- `POST /api/v1/process`: Process a new query
- `POST /api/v1/disambiguate`: Handle disambiguation selection
- `GET /api/v1/queries`: Get all saved queries
- `GET /api/v1/results`: Get all saved results
- `GET /api/v1/query/{query_id}`: Get a specific query and its results

## Architecture

The application uses a multi-agent architecture:

1. **Topic Extractor**: Identifies the main topic from user queries
2. **Disambiguator**: Handles ambiguous topics with interactive clarification
3. **Wikipedia Searcher**: Retrieves relevant Wikipedia articles
4. **Summarizer**: Generates concise summaries using map-reduce chain

## Error Handling

The system includes comprehensive error handling:
- Context length management
- Automatic retries with smaller chunks
- Detailed error logging
- User-friendly error messages

## Recent Updates

- Improved summarization with map-reduce chain
- Enhanced error handling and logging
- Added API endpoints for query history
- Optimized chunk sizes for better performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 