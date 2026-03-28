# LOCI Demo — Warehouse Robot Memory

An interactive demo showing how LOCI's spatiotemporal vector database works,
using a simulated warehouse robot that builds episodic memory as it patrols.

## What This Is

A full-stack web app that simulates an autonomous robot navigating a 2D
warehouse floor. The robot stores everything it "sees" as episodic memory
in LOCI, and you can interactively query that memory across three modes:

1. **Where & When** — Draw a bounding box + time range to query spatial+temporal memories
2. **What's Similar?** — Click a point to find the most similar memories nearby
3. **Predict & Surprise** — See what the robot expects vs. what it actually knows (novelty detection)

## Run Locally

```bash
# From the repo root:
pip install -e ".[dev]"
pip install fastapi uvicorn[standard]

# Start the server:
uvicorn demo.app.main:app --reload

# Open http://localhost:8000
```

## Run with Docker

```bash
# From the repo root:
docker build -f demo/Dockerfile -t loci-demo .
docker run -p 8000:8000 loci-demo

# Open http://localhost:8000
```

## Deploy on Railway

```bash
# Install Railway CLI, then:
railway up
```

Or connect the GitHub repo and set the root directory to the repo root.
Railway will auto-detect the Dockerfile at `demo/Dockerfile`.

## Architecture

```
Browser (vanilla JS)  <-->  FastAPI backend  <-->  LOCI (LocalLociClient)
```

- **Zero infrastructure** — uses `LocalLociClient` (in-memory, no Qdrant needed)
- **Single container** — Python 3.11 + FastAPI + LOCI, no external services
- **Real-time** — WebSocket streams robot position and observations at 2 Hz
- **128-dim embeddings** — deterministic mock vectors from position + visible objects

## How the Demo Works

- The warehouse is a 20x20 grid with shelves, charging stations, loading docks, and obstacles
- The robot follows a predefined patrol route through the aisles
- Every 500ms tick, it observes objects within a 3-cell radius and stores a WorldState in LOCI
- You can place anomalies to see novelty scores spike when the robot encounters them
