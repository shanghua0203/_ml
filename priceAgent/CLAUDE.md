# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a local AI shopping comparison assistant with three-tier agent architecture:

```
main.py (CLI entry) → ComparisonAgent (orchestrator) → [ScrapeAgent, PriceGuardAgent]
```

- **ComparisonAgent**: Orchestrates the full comparison flow (search → scrape → analyze → recommend)
- **ScrapeAgent**: Uses Crawl4AI for async dynamic page scraping with retry logic
- **PriceGuardAgent**: LLM-based price validation and fraud detection

## Key Patterns

### Async/I/O Management
- All web scraping and LLM calls use `async/await`
- ComparisonAgent implements Async Context Manager pattern for proper Crawl4AI resource cleanup
- Semaphore(3) limits concurrent scraping to prevent OOM

### LLM Integration
- Only `langchain_ollama.ChatOllama` (local Llama 3) allowed
- All prompts return JSON with strict schema validation
- Price extraction returns `(price, confidence)` tuple
- Confidence < 50% treated as extraction failure

### Data Validation (Pydantic V2)
- Use `model.model_dump()` not `dict(model)`
- Use `@field_validator` / `@model_validator` not `@validator`
- FraudDetectionResult uses `@model_validator(mode="after")` to sync `has_suspicious_text`

## Database
- `src/utils/db.py`: Async `aiosqlite` wrapper for shopping_history and search_logs
- `src/core/cache.py`: Search result caching (24-hour expiry) using SQLite

## Common Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run comparison
python src/main.py --search "iphone 15"
python src/main.py -s "MacBook Pro" --json

# View history/stats
python src/main.py --history
python src/main.py --stats

# Test single scrape
python src/main.py --scrape https://example.com
```

## Project-Specific Rules

1. **Language**: All responses and code comments must use Traditional Chinese
2. **Configuration**: All settings read from `src/core/config.py` - no hardcoding
3. **Error handling**: Always wrap LLM calls and scraping in try-except with fallbacks
4. **Price validation**: 1000-5,000,000 NT$ is reasonable range
