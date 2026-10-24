# LOCI MCP Server

`loci-mcp` exposes LOCI as a **spatial memory for AI agents** over the
[Model Context Protocol](https://modelcontextprotocol.io). Any MCP client
(Claude Desktop, Claude Code, or your own agent) gets five tools to store
embeddings with *where* and *when* they were observed, recall them by
similarity, place, or time, score how novel a new observation is, and replay
trajectories.

## Install

```bash
pip install "loci-stdb[mcp]"
```

The `mcp` extra pulls in the official MCP Python SDK (`mcp>=1.2,<2`).

## Run

```bash
loci-mcp            # stdio transport; this is what MCP clients launch
loci-mcp --help     # print usage and exit (nothing starts)
python -m loci.mcp  # equivalent to loci-mcp
```

The server speaks MCP on stdin/stdout, so it is normally launched *by* the
MCP client (see the config snippets below), not by hand.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `LOCI_MCP_MODE` | `local` | Backend: `local` (in-memory, zero infra), `qdrant` (persistent, needs a Qdrant server), or `cloud` (LOCI Cloud API). |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant URL, used in `qdrant` mode. |
| `LOCI_CLOUD_URL` | — | LOCI Cloud base URL. **Required** in `cloud` mode. |
| `LOCI_API_KEY` | — | API key. **Required** in `cloud` mode; also passed to Qdrant in `qdrant` mode if set. |
| `LOCI_VECTOR_SIZE` | `512` | Embedding dimensionality. Every `vector` argument must have exactly this many components. |
| `LOCI_EPOCH_SIZE_MS` | `5000` | Logical epoch width in milliseconds (consolidation granularity + Hilbert t-normalisation). |
| `LOCI_DISTANCE` | `cosine` | Distance metric: `cosine`, `dot`, or `euclidean`. |
| `LOCI_SCENE_ID` | `default` | Scene used when a tool call omits `scene_id`. |

Configuration is validated at startup: `loci-mcp` exits with a clear message
on stderr (exit code 2) if, say, `cloud` mode is missing its URL or key.

> **Persistence caveat — read this before trusting `local` mode with
> anything important.** `local` mode keeps all memories in RAM inside the
> server process. When the MCP client restarts the server (which Claude
> Desktop does on every app restart), **everything is forgotten**. It is
> perfect for experimentation and per-session scratch memory. For durable
> memory across sessions, run Qdrant (`docker run -p 6333:6333
> qdrant/qdrant`) and set `LOCI_MCP_MODE=qdrant`.

## Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "loci-memory": {
      "command": "loci-mcp",
      "env": {
        "LOCI_MCP_MODE": "local",
        "LOCI_VECTOR_SIZE": "512"
      }
    }
  }
}
```

## Claude Code

```bash
claude mcp add loci-memory --env LOCI_MCP_MODE=local -- loci-mcp
```

or in `.mcp.json`:

```json
{
  "mcpServers": {
    "loci-memory": {
      "command": "loci-mcp",
      "env": {
        "LOCI_MCP_MODE": "local",
        "LOCI_VECTOR_SIZE": "512"
      }
    }
  }
}
```

For a durable setup, switch the env block to
`{"LOCI_MCP_MODE": "qdrant", "QDRANT_URL": "http://localhost:6333"}`.

## Tools

All tools return friendly `{"error": "..."}` payloads on invalid input —
never tracebacks. Positions are normalised coordinates in `[0, 1]`;
timestamps are Unix milliseconds.

### `remember(vector, x, y, z, timestamp_ms=None, scene_id=None, metadata=None)`

Store an observation: an embedding plus where and when it was seen.
`timestamp_ms` defaults to now; `scene_id` defaults to `LOCI_SCENE_ID`;
`metadata` is any JSON object, stored verbatim and returned on recall.

```json
{"vector": [0.12, "...", 0.87], "x": 0.4, "y": 0.1, "z": 0.0,
 "metadata": {"label": "red door", "source": "cam0"}}
```

Returns `{"id": "<uuid>", "epoch": 341}` — keep the `id` if you want to call
`trajectory` later.

### `recall(vector=None, x=None, y=None, z=None, radius=0.1, time_start_ms=None, time_end_ms=None, scene_id=None, limit=5)`

Retrieve memories by **similarity** (pass `vector`), **place** (pass all of
`x`, `y`, `z`; `radius` bounds the box), **time** (pass `time_start_ms`
and/or `time_end_ms`; a missing start means 0, a missing end means *now*),
or any combination. At least one criterion is required.

Returns a list of `{id, x, y, z, timestamp_ms, scene_id, score, metadata}`.

Place-only / time-only behavior (no `vector`): the underlying store is a
vector database, so the server searches with a constant probe vector,
over-fetches, and re-sorts matches by **recency** (newest first) with
`score` set to `null`. This is honest recall of "what did I see there/then",
but when far more matches exist than the internal over-fetch window
(5× `limit`, minimum 50), the subset seen may be arbitrary — pass a `vector`
for fully ranked recall over large memories.

In `cloud` mode, `scene_id` is filtered client-side after an over-fetch
(the cloud API has no payload filters), so matches beyond the over-fetch
window may be missed; `local` and `qdrant` modes filter in the database.

### `novelty(vector, x=None, y=None, z=None)`

"Have I seen something like this before?" Returns

```json
{"novelty": 0.03, "best_cosine": 0.97, "nearest": ["... up to 3 results ..."]}
```

`novelty` is `1 - best cosine similarity` in `[0, 1]`: ~0 means an
essentially identical memory exists, ~1 means nothing similar was ever
remembered (empty memory scores 1.0). Pass all of `x, y, z` to ask "is this
novel *here*?" — the comparison is then restricted to memories within
radius 0.3 of that position.

### `trajectory(state_id, steps_back=20, steps_forward=20)`

Replay the scene around a remembered state: the time-ordered chain of states
in the same `scene_id`, up to `steps_back` before and `steps_forward` after
the anchor. Returns a chronological list of `{id, x, y, z, timestamp_ms}`.
Not available in `cloud` mode.

### `memory_stats()`

Describe the memory: `{mode, vector_size, distance, epoch_size_ms,
default_scene_id, total_states, oldest_timestamp_ms, newest_timestamp_ms}`.
`total_states` is exact in `local` mode and `"unknown"` where counting is not
cheap. The timestamp bounds span raw and consolidated memories and are `null`
when the memory is empty or the backend cannot report them cheaply.

### No `forget` tool

The LOCI client API has no safe targeted-deletion primitive (retention
purging is an internal, cutoff-based, policy-driven mechanism), so the server
ships no `forget` tool rather than a misleading stub.

## Agent recipes

- **Session scratch memory**: `remember` each salient observation with a
  coarse position encoding (e.g. map document/page/section to `x/y/z`);
  `recall` by vector before answering to ground responses in what was seen.
- **Explore vs exploit**: call `novelty` on each new observation; if
  `novelty > 0.7`, slow down and gather more detail before acting; if
  `< 0.2`, rely on the `nearest` memories instead of re-analysing.
- **"What happened around that?"**: after a `recall` hit, feed its `id` to
  `trajectory` to reconstruct the events before and after it.
- **Episodes**: give each task/run its own `scene_id` so trajectories do not
  interleave, then use `recall(scene_id=...)` to scope retrieval to one
  episode.
- **Time-boxed review**: `recall(time_start_ms=..., time_end_ms=...)` with
  no vector returns the newest-first log of that window — a quick episodic
  digest.
