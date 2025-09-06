# Web UI (React) + Node API

This folder contains a small React (Vite) UI with tabs to switch between:
- LangGraph Agent (general assistant with calculator + FAQ)
- Retail Agent (sales, inventory, pricing tools)

The React app calls a Node API server which handles tool use and OpenAI calls.

## Prerequisites
- Node.js 18+
- `.env` at repo root with `OPENAI_API_KEY=...` and optional `LLM_MODEL=gpt-4o-mini`

## Start API server
```
cd web/server
npm install
npm start
# API: http://localhost:3001
```

## Start React client
```
cd web/client
npm install
npm run dev
# UI: http://localhost:5173 (proxies /api to 3001)
```

## Endpoints
- POST `/api/agent/chat` → messages: [{role, content}] → returns {message}
- POST `/api/retail/chat` → same format

Both routes run a tool loop using OpenAI function calling.

