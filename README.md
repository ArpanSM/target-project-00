# Target Product Search AI

Smart product search system combining Elasticsearch and AI for natural language queries.

## Features

- **AI-Powered Search**: Understands natural language queries
- **Advanced Filters**: Price ranges, brands, categories, and update dates
- **API Access**: REST endpoint for integration

## Setup

1. **Install dependencies**:
```bash
pip install poetry
poetry install
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Load data**:
```bash
poetry run python src/ingest.py
```

4. **Start API**:
```bash
poetry run uvicorn src.api:app --reload
```

## Usage

**Search via API**:
```bash
curl -X POST "http://localhost:8000/search" \
-H "Content-Type: application/json" \
-d '{"query": "Find diapers under $50 updated this month"}'
```

**Example Response**:
```json
{
  "results": [
    {
      "title": "Pampers Diapers Model 304",
      "price": 32.54,
      "description": "Easy-to-clean surfaces, safety certified...",
      "last_updated": "2025-05-05"
    }
  ]
}
```

## Configuration

Edit `.env` file:
