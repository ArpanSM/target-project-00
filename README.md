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


## Miro Link

Link: [Miro Board](https://miro.com/app/board/uXjVI57Q6Z8=/?share_link_id=773437379455)

## Testing

Run the pytest suite:

```bash
python -m pytest test/test_rag.py -v
```

## Project Decisions

1. Since the text descriptions were repetitive across the 50 products, we used an LLM to evaluate the most frequent top sentence against the product name and metadata, keeping only the valid ones.
2. The text quantity per product was too small to chunk based on each product description.
3. Metadata fields were retained as filters during ingestion.
4. The agent defines two tools: one for semantic search only, and one combining semantic search with metadata filters.
5. If the semantic + metadata tool returns no results, a semantic-only search is executed as a fallback.
6. LangSmith is used for tracing the agent's execution path.
