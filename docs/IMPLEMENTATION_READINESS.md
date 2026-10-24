# LOCI Implementation Readiness

This is a delivery and evidence plan, not a claim that the following gates
already pass. Prioritize one working customer workflow before adding backends,
transports, billing, or broader world-model features.

## 1. Select One Pilot and an Owner

Provisional workflow: a robotics engineer selects an incident observation,
retrieves similar historical episodes within a relevant place/time context,
reviews the source clips and trajectories, and exports a regression dataset.
Confirm this with an existing user or potential partner; their actual workflow
may justify a different first pilot.

- Name the daily user, budget owner, engineering owner, and decision date.
- Secure one representative dataset and a fixed set of investigation tasks.
- Record the current process, time per investigation, case quality, and cost.
- Agree scope: one embedding family/version, coordinate convention, ingest
  source, deployment, and retrieval/export path with human review.
- Record acceptance thresholds before implementation, including useful-case
  precision/recall, investigation time, latency, operating cost, and reliability.
- Agree pilot commercial terms and the conditions for a paid continuation;
  interest or a demo alone is not an acceptance signal.

## 2. Make Each Retrieved Observation Usable

Define a versioned ingestion contract and validate it at the boundary:

| Contract | Required decision and evidence |
|---|---|
| Coordinate frame | Frame ID, axes, units, bounds, normalization transform/version, and cross-session transforms |
| Position meaning | Physical XYZ versus image/tubelet coordinates; missing or uncertain pose handling |
| Time | Clock/source, event time versus ingest time, synchronization error, and late arrivals |
| Embedding | Model/checkpoint, preprocessing, version, dimension, metric, and normalization |
| Compatibility | Reject or isolate incompatible frames/embeddings; document re-embedding and migration |
| Provenance | Dataset/episode/device IDs, source clip/frame span, artifact URI/checksum, and access policy |
| Record identity | Stable ingest identity, duplicate/retry policy, and provenance preserved through retrieval/export |

Store needed provenance in metadata or an application-owned manifest and verify
round trips through the selected client/API path. A retrieved vector without
an accessible, correct source observation does not complete the pilot workflow.

## 3. Establish Correctness and Operating Limits

Create an acceptance corpus with independently labeled eligible results and
exact rankings. Verify metrics, spatial/time boundaries, confidence and scene
filters, result limits, and historical cutoffs. Specify empty-result,
incompatible-embedding, and malformed-data behavior.

- Exercise backend timeouts/unavailability, partial batch writes, ambiguous
  acknowledgments, retries, process interruption, and dependency recovery.
- Query backend errors propagate after retries instead of yielding partial
  results; a `loci_data`/`loci_summary` collection deleted out from under a
  client yields an empty result and resets that client's readiness cache.
  Verify this on live Qdrant and audit the trajectory/causal-context helpers,
  which still fall back to the anchor on scroll errors; never treat an
  incomplete search as confirmed novelty.
- Verify data after restart and backup/restore. Record write acknowledgment
  semantics and agreed recovery/data-loss targets.
- Check concurrent collection creation, same-scene writes, duplicate retries,
  and trajectory linking. `insert_batch` links each scene's first state to the
  latest stored predecessor, but concurrent or out-of-order writers can still
  leave inconsistent `prev`/`next` links, and `get_trajectory()` time-sorts the
  scene rather than following them. Document ordering and isolation.
- Validate cloud namespace separation and metadata fidelity if cloud is used.
- The layout is fixed at two collections per tenant (`loci_data` +
  `loci_summary`); bound the raw window, consolidation/retention work,
  connection pools, and memory at the pilot's retention and ingest rate, and
  measure ingest/query contention.
- Define retention/deletion and source-artifact cleanup. Automatic tier
  migration remains future work until implemented and tested.
- Add request/ingest/error/latency and resource telemetry plus a recovery runbook.
  Dependency-audit CI installs the app/cloud dependencies and a `docker-smoke`
  job exercises `/health`, `/insert`, `/query` against a Qdrant container; the
  library's Qdrant integration tests use the embedded `:memory:` engine, so
  live-server client CI and native parity checks remain additional work.

### Upgrade Existing Deployments

Deployments created on the legacy one-collection-per-epoch layout must run
`loci migrate-layout` once (dry-run first) to copy into the bounded
`{prefix}loci_data` / `{prefix}loci_summary` layout; see `CHANGELOG.md`.

Newly created `loci_data` and `loci_summary` collections get FLOAT payload
indexes for `x`, `y`, `z`, and `confidence`, which the exact range conditions
use before top-k; `loci migrate-layout` creates them on the target collections
it creates but leaves an existing target untouched. Collections created before
these indexes existed still return correct results because Qdrant evaluates
Range conditions on unindexed payload fields by scanning; add the indexes
before pilot benchmarking. Run
this idempotent script against the intended database, setting
`LOCI_COLLECTION_PREFIX` to the client's configured prefix (empty only for an
unprefixed store):

```python
import os
from qdrant_client import QdrantClient, models

db = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.getenv("QDRANT_API_KEY"))
prefix = os.environ.get("LOCI_COLLECTION_PREFIX", "")
for name in (f"{prefix}loci_data", f"{prefix}loci_summary"):
    if db.collection_exists(name):
        for field in ("x", "y", "z", "confidence"):
            db.create_payload_index(name, field, models.PayloadSchemaType.FLOAT, wait=True)
```

Deploy the cloud API before or with the SDK: older servers silently ignore
the `min_confidence`, `prev_state_id`, and `next_state_id` request fields and
omit them from responses. Euclidean raw scores are `-distance`; decayed scores
use `1 / (1 + distance) * exp(-lambda * age_ms)` and lie in `(0, 1]`.

## 4. Test the Retrieval Advantage on Live Qdrant

Run the current public client on live Qdrant with pinned versions/configuration.
MemoryStore results, historical query paths, and Rust primitive timings do not
establish this deployment's performance. Compare on the same data, hardware,
ranking objective, and equal recall:

1. Tuned single-collection Qdrant with indexed spatial/time payload filters.
2. Indexed timestamp-range filtering plus those spatial range filters, without
   Hilbert bucketing (the "time filter only" ablation).
3. Current LOCI; ablate overlap and adaptive resolution settings.

Use independent exact ground truth and tune ANN effort/candidate budgets fairly.
Measure recall, eligibility violations, p50/p95/p99, concurrent throughput,
ingest/build time, CPU/RAM/disk, collection count, and operating cost. Cover
cold/warm caches, narrow/wide ranges, sparse/dense regions, skewed trajectories,
long retention, and ingest/query contention. Use pilot data plus scale sweeps
beyond expected volume. Save seeds, queries, repetitions, uncertainty, raw
results, and full configuration.

Gate: meet agreed quality/latency/cost thresholds against the strongest baseline.
If tuned filters or the time-range-only baseline suffice, simplify or justify
LOCI through measured workflow benefit rather than speed.

## 5. Evaluate Prediction Separately

First establish useful incident retrieval without prediction. Add a real
predictor only if the workflow needs future analogs or novelty ranking. Hold
out episodes/sites/time periods, remove overlapping clips/source leakage, and
exclude future observations from the retrieval corpus.

- Compare predicted embeddings with current-state retrieval, a simple predictor,
  and no retrieval on a predefined downstream task.
- Measure useful-case retrieval/task accuracy, false alerts, novelty separation,
  calibration/shift behavior, full inference-plus-retrieval latency, and cost.
- Fix predictor horizon, corpus cutoff, embedding version, and labels; ablate
  prediction versus retrieval to separate their contributions.
- Fit calibration on training/validation data and evaluate held-out data without
  tuning thresholds on it. Report failures and uncertainty, not just demo traces.

Gate: retain prediction only if its task benefit justifies integration, compute
cost, latency, and operating complexity.

## 6. Decide on Paid Continuation

Repeat tasks with intended users against their recorded baseline. Measure time
to find a useful prior case, accepted regression cases, review burden,
source-link correctness, repeated usage, and cost per task.

At the agreed date, review benchmarks, failure/recovery results, the data
contract, user outcomes, deployment/support cost, and known limitations with
the budget owner. Seek a priced continuation with defined scope and ownership.
Continue when technical gates pass and a buyer commits based on outcomes.
Narrow the workflow when retrieval works but users gain little or do not return.
Stop scaling investment when outcomes or willingness to pay remain absent;
record what would need to change before another pilot.
