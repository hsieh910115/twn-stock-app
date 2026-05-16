# Web Frontend

This folder is reserved for the future React or Next.js frontend.

The first migration step is the FastAPI backend in `app/api/main.py`. Once the API shape is stable, the frontend can call endpoints like:

- `GET /api/analyze?ticker=2330`
- `GET /api/watchlist?symbols=2330,2454`
- `GET /api/backtest?ticker=2330`
- `GET /api/ai/momentum`
- `GET /api/ai/potential`
