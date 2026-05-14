# DynaHMRC Demo

**Decentralized Heterogeneous Multi-Robot Collaboration with LLMs**

Paper reproduction of "DynaHMRC: Decentralized Heterogeneous Multi-Robot Collaboration for Dynamic Tasks with Large Language Models" with WebUI + Simulation visualization.

## Architecture

```
                    ┌──────────────────────┐
                    │   React Frontend      │
                    │  (Vite + TypeScript)  │
                    └──────────┬───────────┘
                               │ WebSocket
                    ┌──────────▼───────────┐
                    │   Node.js Backend     │
                    │ (Express + ws)        │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌──────────┐       ┌──────────────┐    ┌──────────────┐
   │DeepSeek  │       │ DynaHMRC     │    │ Simulation   │
   │API/LLM   │       │ Orchestrator │    │ Environment  │
   └──────────┘       └──────────────┘    └──────────────┘
```

## Quick Start

```bash
# 1. Set DeepSeek API Key (optional, mock mode if not set)
export DEEPSEEK_API_KEY="your-key-here"

# 2. Start backend (port 3001)
cd server && npx tsx src/index.ts &

# 3. Start frontend (port 5173)
cd frontend && npx vite &

# Or run both:
npm run dev
```

## Four-Stage Collaboration

| Stage | Description |
|-------|-------------|
| Self-Description | Each robot describes its capabilities |
| Task Allocation & Bidding | Robots propose plans and campaign for leadership |
| Leader Election | Team votes for a leader |
| Execution & Reflection | Closed-loop execution with periodic reflection |

## Robot Types

| Robot | Type | Capabilities |
|-------|------|-------------|
| Alice | Mobile Manipulation | Navigate, Open, Pick, Place |
| Bob | Fixed Arm | Pick, Place (limited range) |
| David | Mobile Robot | Navigate only |
| Lucy | Drone | Aerial navigation, Pick, Place |

## Tasks

- **Pack Objects**: Pack bowl, fork, soap, apple into tray
- **Sort Solids**: Sort colored shapes onto matching panels
- **Make Sandwich**: Stack ingredients in correct order

## Tech Stack

- **Frontend**: React + TypeScript + Vite
- **Backend**: Node.js + Express + WebSocket
- **LLM**: DeepSeek API (with mock fallback)
- **Simulation**: Custom 2D canvas-based environment
