# World Model Integration Guide

## V-JEPA 2 (Meta FAIR)

V-JEPA 2 produces tubelet embeddings: each tubelet covers 2 frames × 16×16 pixels and produces a 1408-dimensional vector.

### Quick Start

```python
import numpy as np
from loci import LocalLociClient
from loci.adapters.vjepa2 import VJEPA2Adapter

# Initialize
client = LocalLociClient(vector_size=1408)
adapter = VJEPA2Adapter()

# Convert a single tubelet
embedding = np.random.randn(1408).astype(np.float32)  # from V-JEPA 2
ws = adapter.tubelet_to_world_state(
    tubelet_embedding=embedding,
    patch_position=(0, 5, 10),       # (time_idx, h_idx, w_idx)
    grid_shape=(8, 14, 14),           # (T, H, W) patch grid dimensions
    timestamp_ms=1000,
    scene_id="kitchen_patrol",
)
client.insert(ws)

# Batch convert an entire clip
clip_output = np.random.randn(8, 14, 14, 1408)  # (T, H, W, D)
states = adapter.batch_clip_to_states(
    clip_embeddings=clip_output,
    start_timestamp_ms=0,
    scene_id="kitchen_patrol",
    frame_interval_ms=33,  # ~30 fps
)
client.insert_batch(states)
```

### Position Mapping

The adapter maps patch grid positions to normalized [0, 1] coordinates:
- `x = w_idx / (W - 1)`
- `y = h_idx / (H - 1)`
- `z = t_idx / (T - 1)`

This maps the 2D+time patch grid into the 3D spatial coordinate system. For actual robot applications, you would typically use the robot's SLAM position for (x, y, z) and treat the embedding as the visual representation at that location.

## DreamerV3 (Hafner et al.)

DreamerV3 uses a Recurrent State-Space Model (RSSM) with:
- Deterministic state `h_t`: GRU hidden state (typically 512 or 1024 dims)
- Stochastic state `z_t`: Categorical distribution (32 categories × 32 classes = 1024 dims)

### Quick Start

```python
import numpy as np
from loci import LocalLociClient
from loci.adapters.dreamer import DreamerV3Adapter

# Initialize with combined vector size
client = LocalLociClient(vector_size=512 + 1024)  # h_t + z_t
adapter = DreamerV3Adapter()

# Convert RSSM state to WorldState
h_t = np.random.randn(512).astype(np.float32)   # from GRU
z_t = np.random.randn(1024).astype(np.float32)  # from categorical

ws = adapter.rssm_to_world_state(
    h_t=h_t,
    z_t=z_t,
    position=(0.3, 0.5, 0.1),  # robot position from SLAM
    timestamp_ms=5000,
    scene_id="sim_episode_42",
)
client.insert(ws)
```

### Confidence Estimation

If no explicit confidence is provided, the adapter estimates it from the stochastic component `z_t`. Higher absolute values in `z_t` indicate more certain categorical choices, yielding higher confidence scores.

### Using Predictions with LOCI

DreamerV3's imagination rollouts pair naturally with LOCI's predict-then-retrieve:

```python
from loci.retrieval.predict import PredictThenRetrieve

def dreamer_predictor(context: list[float]) -> list[float]:
    # Your DreamerV3 imagination step here
    return predicted_embedding

ptr = PredictThenRetrieve(client)
result = ptr.retrieve(
    context_vector=current_state.vector,
    predictor_fn=dreamer_predictor,
    future_horizon_ms=2000,
    current_position=(0.3, 0.5, 0.1),
)
print(f"Novelty: {result.prediction_novelty:.2f}")
```

## Generic Models (numpy / torch)

For any model that produces numpy arrays or PyTorch tensors:

```python
import numpy as np
from loci import LocalLociClient
from loci.adapters.generic import GenericAdapter

client = LocalLociClient(vector_size=256)
adapter = GenericAdapter(expected_dim=256)  # optional dim validation

# From numpy
embedding = np.random.randn(256).astype(np.float32)
ws = adapter.from_numpy(
    embedding=embedding,
    position=(0.5, 0.5, 0.0),
    timestamp_ms=1000,
    scene_id="experiment_1",
    scale_level="frame",
)
client.insert(ws)

# From torch (requires torch installed)
import torch
tensor = torch.randn(256)
ws = adapter.from_torch(
    embedding=tensor,
    position=(0.5, 0.5, 0.0),
    timestamp_ms=2000,
    scene_id="experiment_1",
)
client.insert(ws)
```

## Orange Pi 5 Robot Demo

For a physical robot running on Orange Pi 5:

1. Install LOCI: `pip install loci-db`
2. Run Qdrant locally: `docker run -p 6333:6333 qdrant/qdrant`
3. Use the generic adapter with your robot's sensor pipeline:

```python
from loci import LociClient
from loci.adapters.generic import GenericAdapter

client = LociClient("http://localhost:6333", vector_size=512)
adapter = GenericAdapter()

# In your robot's control loop:
while running:
    embedding = get_visual_embedding()    # your vision model
    x, y, z = get_slam_position()         # your SLAM system
    ts = get_timestamp_ms()

    ws = adapter.from_numpy(
        embedding=embedding,
        position=(x, y, z),
        timestamp_ms=ts,
        scene_id=f"patrol_{session_id}",
    )
    client.insert(ws)

    # Predict-then-retrieve for anticipatory navigation
    result = client.predict_and_retrieve(
        context_vector=ws.vector,
        predictor_fn=your_predictor,
        future_horizon_ms=5000,
        current_position=(x, y, z),
    )
    if result.prediction_novelty > 0.8:
        slow_down_and_explore()
```
