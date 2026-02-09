# Minsy Backend

Backend service for Minsy - AI-powered quantitative trading strategy platform.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- TA-Lib C library (for technical analysis)

### Install TA-Lib (macOS)

```bash
brew install ta-lib
```

### Create Virtual Environment & Install Dependencies

```bash
cd backend
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### Environment Variables

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

## Running the Server

### Development

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check

## Project Structure

```
backend/
├── api/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── routes/            # API route handlers
│   ├── deps.py            # Dependencies
│   └── middleware.py      # Custom middleware
├── agents/                 # AI agents
│   ├── mcp/               # MCP integration
│   └── skills/            # Agent skills
├── engine/                 # Trading engine
│   ├── data/              # Data providers
│   ├── strategy/          # Strategy definitions
│   ├── backtest/          # Backtesting engine
│   ├── stresstest/        # Stress testing
│   └── performance/       # Performance analytics
├── trade/                  # Trading execution (future)
├── tests/                  # Test suite
├── pyproject.toml         # Project configuration
├── Dockerfile             # Docker configuration
└── .env                   # Environment variables
```

## Testing

```bash
pytest
```

## Linting

```bash
ruff check .
ruff format .
```
