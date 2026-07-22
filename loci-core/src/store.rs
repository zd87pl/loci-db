//! In-memory vector store — RFC-0001 R5 stage (a).
//!
//! A Rust-native behavioral drop-in for the Python `MemoryStore`
//! (`loci/backends/memory.py`), exposed to Python as `loci_core.LociStore`
//! and wrapped by `loci.backends.rust_store.RustMemoryStore`.
//!
//! # Architecture
//!
//! Each named collection stores vectors in a single contiguous row-major
//! `Vec<f32>` arena of fixed dimension.  Per-row L2 norms are cached at
//! write time so cosine search never re-normalises the arena.  Point ids
//! map to row indices through a `HashMap`; payloads are one
//! `serde_json::Value` per row.
//!
//! # Tombstones
//!
//! Deletes never move rows: the row is marked dead (`alive[row] = false`),
//! its payload dropped, and the index pushed onto a free-list.  The next
//! insert of a *new* id reuses a free row in place (swap-free, so no id ->
//! row remapping is ever needed).  Consequence: physical row order is
//! insertion order only until the first delete — same as the Python
//! store's dict, whose iteration order is also an implementation detail.
//!
//! # Score convention
//!
//! Identical to the Python store: scores are always higher-is-better.
//! Cosine and dot are similarities; euclidean is the *negative* L2
//! distance.  Zero query vectors yield 0.0 cosine scores and zero-norm
//! stored vectors have their norm substituted with 1.0, both mirroring
//! the reference implementation.
//!
//! # Stage (b)/(c) seams (persistence, quantization)
//!
//! - `Collection` owns all per-collection state; persistence will replace
//!   the single arena with mmap-backed per-epoch segments behind the same
//!   row-addressed accessors (`row_vec`, `put`, `kill`).
//! - Payloads are stage-(a) row-wise JSON.  Stage (b) columnarizes the hot
//!   fields (`timestamp_ms`, `x`, `y`, `z`, `confidence`, `scene_id`,
//!   `hilbert_r*`) into typed columns keyed by the recorded
//!   `payload_indices`, keeping a residual JSON blob for `metadata`;
//!   `Filter::matches` then evaluates against columns instead of `Value`s.
//! - Quantization adds an int8 arena + per-row scale beside `norms`; the
//!   f32 arena stays for exact rerank.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors surfaced by [`Store`] operations.
///
/// The Python binding maps these onto the same exception types the Python
/// `MemoryStore` raises: `MissingCollection` -> `KeyError`, dimension
/// mismatches and bad filters -> `ValueError`, `Uncomparable` -> `TypeError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// Operation on a collection that must exist (upsert / set_payload).
    MissingCollection(String),
    /// A point's vector does not match the collection's dimension.
    PointDimension {
        point_id: String,
        got: usize,
        expected: usize,
        collection: String,
    },
    /// A query vector does not match the collection's dimension.
    QueryDimension { got: usize, expected: usize },
    /// A payload filter could not be interpreted.
    BadFilter(String),
    /// `payload_value_range` saw values of incomparable types.
    Uncomparable(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::MissingCollection(name) => write!(f, "{name}"),
            StoreError::PointDimension {
                point_id,
                got,
                expected,
                collection,
            } => write!(
                f,
                "vector for point '{point_id}' has dimension {got}, \
                 expected {expected} for collection '{collection}'"
            ),
            StoreError::QueryDimension { got, expected } => {
                write!(f, "query vector has dimension {got}, expected {expected}")
            }
            StoreError::BadFilter(msg) => write!(f, "invalid payload filter: {msg}"),
            StoreError::Uncomparable(field) => {
                write!(f, "payload field '{field}' mixes incomparable value types")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Distance metric for a collection (scores are always higher-is-better).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Distance {
    #[default]
    Cosine,
    Dot,
    /// Scored as negative L2 distance.
    Euclid,
}

impl Distance {
    /// Parse the Python-side metric string.  Mirrors the reference
    /// implementation exactly: anything that is not `"cosine"` or `"dot"`
    /// falls through to euclidean.
    pub fn parse(s: &str) -> Distance {
        match s {
            "cosine" => Distance::Cosine,
            "dot" => Distance::Dot,
            _ => Distance::Euclid,
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD-friendly kernels (autovectorized: 8 independent accumulator lanes)
// ---------------------------------------------------------------------------

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (ca, cb) in (&mut ac).zip(&mut bc) {
        for k in 0..8 {
            acc[k] += ca[k] * cb[k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
        s += x * y;
    }
    s
}

#[inline]
fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (ca, cb) in (&mut ac).zip(&mut bc) {
        for k in 0..8 {
            let d = ca[k] - cb[k];
            acc[k] += d * d;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
        let d = x - y;
        s += d * d;
    }
    s
}

#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    dot_f32(v, v).sqrt()
}

// ---------------------------------------------------------------------------
// Payload value helpers (Python comparison semantics over JSON values)
// ---------------------------------------------------------------------------

/// Numeric view of a JSON value.  Booleans count as numbers because Python's
/// `True == 1` / `False == 0` semantics leak into filter matching.
#[inline]
fn as_num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Equality with Python semantics: `5 == 5.0`, `True == 1`, deep for
/// arrays/objects, `None == None`.
fn json_eq(a: &Value, b: &Value) -> bool {
    if let (Some(x), Some(y)) = (as_num(a), as_num(b)) {
        return x == y;
    }
    match (a, b) {
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Null, Value::Null) => true,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => false,
    }
}

/// Canonical hash key for a numeric filter value (`5`, `5.0`, and `true`
/// must all collide; `-0.0` folds into `0.0`).
#[inline]
fn num_key(x: f64) -> u64 {
    (if x == 0.0 { 0.0 } else { x }).to_bits()
}

// ---------------------------------------------------------------------------
// Filters — semantics replicated from loci/backends/memory.py::_matches
// ---------------------------------------------------------------------------

/// Set-membership condition (`{"any": [...]}`) with O(1) lookup for the
/// common numeric (Hilbert bucket) and string cases.
#[derive(Debug, Clone, Default)]
struct AnySet {
    nums: HashSet<u64>,
    strs: HashSet<String>,
    /// Rare non-scalar entries (null, arrays, objects): linear `json_eq` scan,
    /// matching the Python list-membership semantics.
    others: Vec<Value>,
}

impl AnySet {
    fn build(items: &[Value]) -> AnySet {
        let mut set = AnySet::default();
        for item in items {
            if let Some(x) = as_num(item) {
                set.nums.insert(num_key(x));
            } else if let Value::String(s) = item {
                set.strs.insert(s.clone());
            } else {
                set.others.push(item.clone());
            }
        }
        set
    }

    fn contains(&self, v: &Value) -> bool {
        if let Some(x) = as_num(v) {
            if self.nums.contains(&num_key(x)) {
                return true;
            }
        }
        if let Value::String(s) = v {
            if self.strs.contains(s) {
                return true;
            }
        }
        self.others.iter().any(|o| json_eq(o, v))
    }
}

#[derive(Debug, Clone)]
enum Condition {
    /// `{field: value}` — exact match.
    Equals(Value),
    /// `{field: {"any": [...]}}` — set membership.
    AnyOf(AnySet),
    /// `{field: {"gte"/"lte"/"gt"/"lt": num}}` — numeric range.  A missing
    /// or non-numeric payload value fails every present bound (the Python
    /// store excludes `None`); an all-`None` range matches everything, like
    /// the reference's empty condition dict.
    Range {
        gte: Option<f64>,
        lte: Option<f64>,
        gt: Option<f64>,
        lt: Option<f64>,
    },
}

/// A parsed payload filter: the conjunction of per-field conditions.
#[derive(Debug, Clone, Default)]
pub struct Filter {
    conds: Vec<(String, Condition)>,
}

impl Filter {
    /// Parse the JSON object form used by the Python store
    /// (`{"field": value | {"any": [...]} | {"gte": .., "lte": ..}}`).
    pub fn parse(json: &str) -> Result<Filter, StoreError> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| StoreError::BadFilter(e.to_string()))?;
        Filter::from_value(&value)
    }

    fn from_value(value: &Value) -> Result<Filter, StoreError> {
        let Value::Object(map) = value else {
            return Err(StoreError::BadFilter("filter must be a JSON object".into()));
        };
        let mut conds = Vec::with_capacity(map.len());
        for (key, cond) in map {
            let parsed = match cond {
                Value::Object(obj) => {
                    if let Some(any) = obj.get("any") {
                        let Value::Array(items) = any else {
                            return Err(StoreError::BadFilter(format!(
                                "'any' condition for field '{key}' must be a list"
                            )));
                        };
                        Condition::AnyOf(AnySet::build(items))
                    } else {
                        let bound = |name: &str| -> Result<Option<f64>, StoreError> {
                            match obj.get(name) {
                                None => Ok(None),
                                Some(v) => as_num(v).map(Some).ok_or_else(|| {
                                    StoreError::BadFilter(format!(
                                        "'{name}' bound for field '{key}' must be numeric"
                                    ))
                                }),
                            }
                        };
                        Condition::Range {
                            gte: bound("gte")?,
                            lte: bound("lte")?,
                            gt: bound("gt")?,
                            lt: bound("lt")?,
                        }
                    }
                }
                other => Condition::Equals(other.clone()),
            };
            conds.push((key.clone(), parsed));
        }
        Ok(Filter { conds })
    }

    /// Check whether *payload* satisfies every condition.
    pub fn matches(&self, payload: &Value) -> bool {
        let obj = payload.as_object();
        for (key, cond) in &self.conds {
            let value = obj.and_then(|o| o.get(key.as_str()));
            match cond {
                Condition::Equals(want) => {
                    if !json_eq(value.unwrap_or(&Value::Null), want) {
                        return false;
                    }
                }
                Condition::AnyOf(set) => {
                    if !set.contains(value.unwrap_or(&Value::Null)) {
                        return false;
                    }
                }
                Condition::Range { gte, lte, gt, lt } => {
                    let n = value.and_then(as_num);
                    if let Some(b) = gte {
                        if !matches!(n, Some(x) if x >= *b) {
                            return false;
                        }
                    }
                    if let Some(b) = lte {
                        if !matches!(n, Some(x) if x <= *b) {
                            return false;
                        }
                    }
                    if let Some(b) = lt {
                        if !matches!(n, Some(x) if x < *b) {
                            return false;
                        }
                    }
                    if let Some(b) = gt {
                        if !matches!(n, Some(x) if x > *b) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Rows returned to callers
// ---------------------------------------------------------------------------

/// An owned copy of a stored point (copy-on-read, like the Python store).
#[derive(Debug, Clone)]
pub struct PointOut {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: Value,
    /// Present only for search results.
    pub score: Option<f64>,
}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Collection {
    dim: usize,
    distance: Distance,
    /// Row-major vector arena: row r occupies `arena[r*dim .. (r+1)*dim]`.
    arena: Vec<f32>,
    /// Cached per-row L2 norms (for cosine scoring).
    norms: Vec<f32>,
    ids: Vec<String>,
    payloads: Vec<Value>,
    alive: Vec<bool>,
    id_to_row: HashMap<String, usize>,
    /// Tombstoned rows available for reuse.
    free_rows: Vec<usize>,
    live_count: usize,
    /// Recorded but unused in stage (a); stage (b) keys typed payload
    /// columns off this set.
    payload_indices: HashSet<String>,
}

impl Collection {
    fn new(dim: usize, distance: Distance) -> Collection {
        Collection {
            dim,
            distance,
            ..Default::default()
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    fn row_vec(&self, row: usize) -> &[f32] {
        &self.arena[row * self.dim..(row + 1) * self.dim]
    }

    /// Insert or overwrite one point (copy-on-write: the caller's buffers
    /// are moved/copied into the arena, never aliased).
    fn put(&mut self, id: String, vector: &[f32], payload: Value) {
        let norm = l2_norm(vector);
        if let Some(&row) = self.id_to_row.get(&id) {
            let dim = self.dim;
            self.arena[row * dim..(row + 1) * dim].copy_from_slice(vector);
            self.norms[row] = norm;
            self.payloads[row] = payload;
        } else if let Some(row) = self.free_rows.pop() {
            let dim = self.dim;
            self.arena[row * dim..(row + 1) * dim].copy_from_slice(vector);
            self.norms[row] = norm;
            self.payloads[row] = payload;
            self.ids[row] = id.clone();
            self.alive[row] = true;
            self.id_to_row.insert(id, row);
            self.live_count += 1;
        } else {
            let row = self.rows();
            self.arena.extend_from_slice(vector);
            self.norms.push(norm);
            self.payloads.push(payload);
            self.ids.push(id.clone());
            self.alive.push(true);
            self.id_to_row.insert(id, row);
            self.live_count += 1;
        }
    }

    /// Tombstone a row whose id has already been removed from `id_to_row`.
    fn kill(&mut self, row: usize) {
        self.alive[row] = false;
        self.payloads[row] = Value::Null;
        self.ids[row].clear();
        self.free_rows.push(row);
        self.live_count -= 1;
    }

    fn point_out(&self, row: usize, score: Option<f64>) -> PointOut {
        PointOut {
            id: self.ids[row].clone(),
            vector: self.row_vec(row).to_vec(),
            payload: self.payloads[row].clone(),
            score,
        }
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// Named collections of spatiotemporal vectors — the stage (a) engine.
#[derive(Debug, Default)]
pub struct Store {
    collections: HashMap<String, Collection>,
}

impl Store {
    pub fn new() -> Store {
        Store::default()
    }

    // -- Collection lifecycle ------------------------------------------------

    /// Create a collection; a no-op if it already exists (Python parity).
    pub fn create_collection(&mut self, name: &str, vector_size: usize, distance: &str) {
        self.collections
            .entry(name.to_string())
            .or_insert_with(|| Collection::new(vector_size, Distance::parse(distance)));
    }

    pub fn collection_exists(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }

    pub fn delete_collection(&mut self, name: &str) {
        self.collections.remove(name);
    }

    pub fn create_payload_index(&mut self, collection: &str, field: &str) {
        if let Some(col) = self.collections.get_mut(collection) {
            col.payload_indices.insert(field.to_string());
        }
    }

    // -- Write ---------------------------------------------------------------

    /// Insert or update points.  Mirrors the Python store: a missing
    /// collection is an error (`KeyError` in Python); validation happens
    /// per point, so points before a dimension mismatch stay inserted.
    pub fn upsert(
        &mut self,
        collection: &str,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        payloads: Vec<Value>,
    ) -> Result<(), StoreError> {
        if !self.collections.contains_key(collection) {
            return Err(StoreError::MissingCollection(collection.to_string()));
        }
        for ((id, vector), payload) in ids.into_iter().zip(vectors).zip(payloads) {
            self.upsert_point(collection, id, &vector, payload)?;
        }
        Ok(())
    }

    /// Insert or update a single point (streaming form of [`Store::upsert`]).
    pub fn upsert_point(
        &mut self,
        collection: &str,
        id: String,
        vector: &[f32],
        payload: Value,
    ) -> Result<(), StoreError> {
        let col = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| StoreError::MissingCollection(collection.to_string()))?;
        if vector.len() != col.dim {
            return Err(StoreError::PointDimension {
                point_id: id,
                got: vector.len(),
                expected: col.dim,
                collection: collection.to_string(),
            });
        }
        col.put(id, vector, payload);
        Ok(())
    }

    /// Merge top-level keys of *payload* into an existing point's payload.
    /// Missing point: silent no-op.  Missing collection: error.
    pub fn set_payload(
        &mut self,
        collection: &str,
        point_id: &str,
        payload: Value,
    ) -> Result<(), StoreError> {
        let col = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| StoreError::MissingCollection(collection.to_string()))?;
        if let Some(&row) = col.id_to_row.get(point_id) {
            let Value::Object(update) = payload else {
                return Err(StoreError::BadFilter("payload must be a JSON object".into()));
            };
            if let Value::Object(existing) = &mut col.payloads[row] {
                for (k, v) in update {
                    existing.insert(k, v);
                }
            } else {
                col.payloads[row] = Value::Object(update);
            }
        }
        Ok(())
    }

    // -- Delete --------------------------------------------------------------

    /// Delete points by id; returns the number actually removed.
    pub fn delete_points(&mut self, collection: &str, ids: &[String]) -> usize {
        let Some(col) = self.collections.get_mut(collection) else {
            return 0;
        };
        let mut removed = 0;
        for id in ids {
            if let Some(row) = col.id_to_row.remove(id) {
                col.kill(row);
                removed += 1;
            }
        }
        removed
    }

    /// Delete points with `start_ms <= payload[field] < end_ms_exclusive`.
    /// Points missing *field* are never deleted.
    pub fn delete_points_in_time_range(
        &mut self,
        collection: &str,
        start_ms: i64,
        end_ms_exclusive: i64,
        field: &str,
    ) -> usize {
        let Some(col) = self.collections.get_mut(collection) else {
            return 0;
        };
        let (lo, hi) = (start_ms as f64, end_ms_exclusive as f64);
        let doomed: Vec<usize> = (0..col.rows())
            .filter(|&row| {
                col.alive[row]
                    && matches!(
                        col.payloads[row].get(field).and_then(as_num),
                        Some(v) if lo <= v && v < hi
                    )
            })
            .collect();
        for &row in &doomed {
            let id = std::mem::take(&mut col.ids[row]);
            col.id_to_row.remove(&id);
            col.kill(row);
        }
        doomed.len()
    }

    // -- Read ----------------------------------------------------------------

    /// Fetch points by id, preserving request order and skipping misses.
    pub fn retrieve(&self, collection: &str, ids: &[String]) -> Vec<PointOut> {
        let Some(col) = self.collections.get(collection) else {
            return vec![];
        };
        ids.iter()
            .filter_map(|id| col.id_to_row.get(id).map(|&row| col.point_out(row, None)))
            .collect()
    }

    /// Brute-force scored search with optional payload filtering.
    /// Results are sorted by score descending (higher-is-better for every
    /// metric; euclidean scores are negated distances).
    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        limit: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<PointOut>, StoreError> {
        let Some(col) = self.collections.get(collection) else {
            return Ok(vec![]);
        };
        if query.len() != col.dim {
            return Err(StoreError::QueryDimension {
                got: query.len(),
                expected: col.dim,
            });
        }
        if limit == 0 {
            return Ok(vec![]);
        }

        let q_norm = l2_norm(query);
        let mut scored: Vec<(f32, u32)> = Vec::new();
        for row in 0..col.rows() {
            if !col.alive[row] {
                continue;
            }
            if let Some(f) = filter {
                if !f.matches(&col.payloads[row]) {
                    continue;
                }
            }
            let v = col.row_vec(row);
            let score = match col.distance {
                Distance::Cosine => {
                    if q_norm == 0.0 {
                        0.0
                    } else {
                        let rn = col.norms[row];
                        let rn = if rn == 0.0 { 1.0 } else { rn };
                        dot_f32(query, v) / (rn * q_norm)
                    }
                }
                Distance::Dot => dot_f32(query, v),
                Distance::Euclid => -l2_sq_f32(query, v).sqrt(),
            };
            scored.push((score, row as u32));
        }

        let k = limit.min(scored.len());
        if k == 0 {
            return Ok(vec![]);
        }
        if scored.len() > k {
            scored.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            scored.truncate(k);
        }
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        Ok(scored
            .into_iter()
            .map(|(score, row)| col.point_out(row as usize, Some(score as f64)))
            .collect())
    }

    /// Filtered scan.  Unordered scrolls early-exit at *limit*; ordered
    /// scrolls stable-sort matches ascending by `payload[order_by]`
    /// (missing field sorts as 0, Python parity) before truncating.
    pub fn scroll(
        &self,
        collection: &str,
        filter: Option<&Filter>,
        limit: usize,
        order_by: Option<&str>,
    ) -> Vec<PointOut> {
        let Some(col) = self.collections.get(collection) else {
            return vec![];
        };
        let matching = (0..col.rows()).filter(|&row| {
            col.alive[row] && filter.map_or(true, |f| f.matches(&col.payloads[row]))
        });
        match order_by {
            None => matching
                .take(limit)
                .map(|row| col.point_out(row, None))
                .collect(),
            Some(field) => {
                let mut keyed: Vec<(OrderKey, usize)> = matching
                    .map(|row| (order_key(&col.payloads[row], field), row))
                    .collect();
                keyed.sort_by(|a, b| a.0.cmp_key(&b.0));
                keyed
                    .into_iter()
                    .take(limit)
                    .map(|(_, row)| col.point_out(row, None))
                    .collect()
            }
        }
    }

    // -- Stats ---------------------------------------------------------------

    pub fn total_points(&self) -> usize {
        self.collections.values().map(|c| c.live_count).sum()
    }

    pub fn collection_count(&self, name: &str) -> usize {
        self.collections.get(name).map_or(0, |c| c.live_count)
    }

    /// `(min, max)` of a payload field across a collection; `None` when no
    /// point carries the field or the collection is missing.
    pub fn payload_value_range(
        &self,
        collection: &str,
        field: &str,
    ) -> Result<Option<(Value, Value)>, StoreError> {
        let Some(col) = self.collections.get(collection) else {
            return Ok(None);
        };
        let mut min_max: Option<(Value, Value)> = None;
        for row in 0..col.rows() {
            if !col.alive[row] {
                continue;
            }
            let Some(v) = col.payloads[row].get(field) else {
                continue;
            };
            if v.is_null() {
                continue;
            }
            match &mut min_max {
                None => min_max = Some((v.clone(), v.clone())),
                Some((lo, hi)) => {
                    if cmp_scalar(v, lo, field)? == Ordering::Less {
                        *lo = v.clone();
                    }
                    if cmp_scalar(v, hi, field)? == Ordering::Greater {
                        *hi = v.clone();
                    }
                }
            }
        }
        Ok(min_max)
    }
}

/// Sort key for ordered scrolls: numbers first (by value), then strings.
#[derive(Debug, Clone, PartialEq)]
enum OrderKey {
    Num(f64),
    Str(String),
}

impl OrderKey {
    fn cmp_key(&self, other: &OrderKey) -> Ordering {
        match (self, other) {
            (OrderKey::Num(a), OrderKey::Num(b)) => a.total_cmp(b),
            (OrderKey::Str(a), OrderKey::Str(b)) => a.cmp(b),
            (OrderKey::Num(_), OrderKey::Str(_)) => Ordering::Less,
            (OrderKey::Str(_), OrderKey::Num(_)) => Ordering::Greater,
        }
    }
}

fn order_key(payload: &Value, field: &str) -> OrderKey {
    match payload.get(field) {
        Some(Value::String(s)) => OrderKey::Str(s.clone()),
        Some(v) => OrderKey::Num(as_num(v).unwrap_or(0.0)),
        None => OrderKey::Num(0.0),
    }
}

/// Compare two scalar payload values (numbers with numbers, strings with
/// strings); anything else is an error, mirroring Python's `min`/`max`
/// raising `TypeError` on mixed types.
fn cmp_scalar(a: &Value, b: &Value, field: &str) -> Result<Ordering, StoreError> {
    if let (Some(x), Some(y)) = (as_num(a), as_num(b)) {
        return x
            .partial_cmp(&y)
            .ok_or_else(|| StoreError::Uncomparable(field.to_string()));
    }
    if let (Value::String(x), Value::String(y)) = (a, b) {
        return Ok(x.cmp(y));
    }
    Err(StoreError::Uncomparable(field.to_string()))
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
    use pyo3::intern;
    use pyo3::prelude::*;
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
    use pyo3::IntoPyObjectExt;

    fn to_py_err(err: StoreError) -> PyErr {
        match err {
            StoreError::MissingCollection(name) => PyKeyError::new_err(name),
            StoreError::Uncomparable(_) => PyTypeError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }

    /// Convert a Python object into a `serde_json::Value` (pythonize-style,
    /// without the extra dependency).  This is the copy-on-write boundary:
    /// stored payloads never alias caller objects.
    fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
        if obj.is_none() {
            return Ok(Value::Null);
        }
        // bool is a subclass of int in Python: check it first.
        if obj.is_instance_of::<PyBool>() {
            return Ok(Value::Bool(obj.extract::<bool>()?));
        }
        if obj.is_instance_of::<PyInt>() {
            if let Ok(i) = obj.extract::<i64>() {
                return Ok(Value::Number(i.into()));
            }
            if let Ok(u) = obj.extract::<u64>() {
                return Ok(Value::Number(u.into()));
            }
            return float_value(obj.extract::<f64>()?);
        }
        if obj.is_instance_of::<PyFloat>() {
            return float_value(obj.extract::<f64>()?);
        }
        if let Ok(s) = obj.cast::<PyString>() {
            return Ok(Value::String(s.to_str()?.to_owned()));
        }
        if let Ok(d) = obj.cast::<PyDict>() {
            let mut map = serde_json::Map::with_capacity(d.len());
            for (k, v) in d.iter() {
                let key: String = k
                    .extract()
                    .map_err(|_| PyTypeError::new_err("payload keys must be strings"))?;
                map.insert(key, py_to_value(&v)?);
            }
            return Ok(Value::Object(map));
        }
        if let Ok(l) = obj.cast::<PyList>() {
            let mut arr = Vec::with_capacity(l.len());
            for v in l.iter() {
                arr.push(py_to_value(&v)?);
            }
            return Ok(Value::Array(arr));
        }
        if let Ok(t) = obj.cast::<PyTuple>() {
            let mut arr = Vec::with_capacity(t.len());
            for v in t.iter() {
                arr.push(py_to_value(&v)?);
            }
            return Ok(Value::Array(arr));
        }
        Err(PyTypeError::new_err(format!(
            "unsupported payload value type: {}",
            obj.get_type().name()?
        )))
    }

    fn float_value(x: f64) -> PyResult<Value> {
        serde_json::Number::from_f64(x)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("payload floats must be finite"))
    }

    /// Extract an embedding vector into a reusable buffer, with a fast
    /// path for lists of Python floats (the overwhelmingly common case),
    /// falling back to pyo3's generic sequence extraction for
    /// ints/tuples/etc.  The buffer is cleared first.
    fn extract_vector_into(obj: &Bound<'_, PyAny>, out: &mut Vec<f32>) -> PyResult<()> {
        out.clear();
        if let Ok(list) = obj.cast::<PyList>() {
            out.reserve(list.len());
            for item in list.iter() {
                match item.cast_exact::<PyFloat>() {
                    Ok(f) => out.push(f.value() as f32),
                    Err(_) => out.push(item.extract::<f32>()?),
                }
            }
        } else {
            *out = obj.extract()?;
        }
        Ok(())
    }

    /// Convert a stored `Value` back into fresh Python objects
    /// (copy-on-read: results never alias stored payloads).
    fn value_to_py(py: Python<'_>, v: &Value) -> PyResult<Py<PyAny>> {
        match v {
            Value::Null => Ok(py.None()),
            Value::Bool(b) => b.into_py_any(py),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i.into_py_any(py)
                } else if let Some(u) = n.as_u64() {
                    u.into_py_any(py)
                } else {
                    n.as_f64().unwrap_or(f64::NAN).into_py_any(py)
                }
            }
            Value::String(s) => s.as_str().into_py_any(py),
            Value::Array(items) => {
                let objs = items
                    .iter()
                    .map(|item| value_to_py(py, item))
                    .collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, objs)?.into_py_any(py)
            }
            Value::Object(map) => {
                let dict = PyDict::new(py);
                for (k, val) in map {
                    dict.set_item(k, value_to_py(py, val)?)?;
                }
                dict.into_py_any(py)
            }
        }
    }

    fn parse_filter(payload_filter: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Filter>> {
        match payload_filter {
            None => Ok(None),
            Some(obj) if obj.is_none() => Ok(None),
            Some(obj) => {
                let value = py_to_value(obj)?;
                Ok(Some(Filter::from_value(&value).map_err(to_py_err)?))
            }
        }
    }

    /// Build the `{"id", "vector", "payload"[, "score"]}` result dict.
    fn hit_to_dict<'py>(py: Python<'py>, hit: &PointOut) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item(intern!(py, "id"), hit.id.as_str())?;
        dict.set_item(intern!(py, "vector"), hit.vector.as_slice())?;
        dict.set_item(intern!(py, "payload"), value_to_py(py, &hit.payload)?)?;
        if let Some(score) = hit.score {
            dict.set_item(intern!(py, "score"), score)?;
        }
        Ok(dict)
    }

    fn hits_to_list<'py>(py: Python<'py>, hits: &[PointOut]) -> PyResult<Bound<'py, PyList>> {
        let dicts = hits
            .iter()
            .map(|h| hit_to_dict(py, h))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, dicts)
    }

    /// Native in-memory vector store (stage (a) of the embedded engine).
    ///
    /// Points, payloads, and filters cross the FFI boundary as plain
    /// Python dicts/lists, converted natively to `serde_json::Value`
    /// (both directions copy, giving the Python store's copy-on-write /
    /// copy-on-read guarantees for free).  The thin Python wrapper
    /// (`loci.backends.rust_store.RustMemoryStore`) re-exposes the exact
    /// `MemoryStore` API.
    #[pyclass(name = "LociStore")]
    pub struct PyLociStore {
        inner: Store,
    }

    #[pymethods]
    impl PyLociStore {
        #[new]
        fn new() -> Self {
            PyLociStore { inner: Store::new() }
        }

        #[pyo3(signature = (name, vector_size, distance = "cosine"))]
        fn create_collection(&mut self, name: &str, vector_size: usize, distance: &str) {
            self.inner.create_collection(name, vector_size, distance);
        }

        fn collection_exists(&self, name: &str) -> bool {
            self.inner.collection_exists(name)
        }

        fn delete_collection(&mut self, name: &str) {
            self.inner.delete_collection(name);
        }

        fn create_payload_index(&mut self, collection: &str, field_name: &str) {
            self.inner.create_payload_index(collection, field_name);
        }

        /// Insert or update points: a list of `{"id", "vector", "payload"}`
        /// dicts, exactly as the Python `MemoryStore.upsert` takes them.
        fn upsert(&mut self, collection: &str, points: &Bound<'_, PyList>) -> PyResult<()> {
            let py = points.py();
            let col = self
                .inner
                .collections
                .get_mut(collection)
                .ok_or_else(|| PyKeyError::new_err(collection.to_string()))?;
            let id_key = intern!(py, "id");
            let vector_key = intern!(py, "vector");
            let payload_key = intern!(py, "payload");
            let mut vector: Vec<f32> = Vec::new();
            for item in points.iter() {
                let point = item
                    .cast::<PyDict>()
                    .map_err(|_| PyTypeError::new_err("each point must be a dict"))?;
                let id_obj = point
                    .get_item(id_key)?
                    .ok_or_else(|| PyKeyError::new_err("id"))?;
                let id = match id_obj.extract::<String>() {
                    Ok(s) => s,
                    Err(_) => id_obj.str()?.to_string(),
                };
                extract_vector_into(
                    &point
                        .get_item(vector_key)?
                        .ok_or_else(|| PyKeyError::new_err("vector"))?,
                    &mut vector,
                )?;
                if vector.len() != col.dim {
                    return Err(to_py_err(StoreError::PointDimension {
                        point_id: id,
                        got: vector.len(),
                        expected: col.dim,
                        collection: collection.to_string(),
                    }));
                }
                let payload = py_to_value(
                    &point
                        .get_item(payload_key)?
                        .ok_or_else(|| PyKeyError::new_err("payload"))?,
                )?;
                col.put(id, &vector, payload);
            }
            Ok(())
        }

        fn set_payload(
            &mut self,
            collection: &str,
            point_id: &str,
            payload: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let payload = py_to_value(payload)?;
            self.inner
                .set_payload(collection, point_id, payload)
                .map_err(to_py_err)
        }

        fn delete_points(&mut self, collection: &str, ids: Vec<String>) -> usize {
            self.inner.delete_points(collection, &ids)
        }

        #[pyo3(signature = (collection, start_ms, end_ms_exclusive, field = "timestamp_ms"))]
        fn delete_points_in_time_range(
            &mut self,
            collection: &str,
            start_ms: i64,
            end_ms_exclusive: i64,
            field: &str,
        ) -> usize {
            self.inner
                .delete_points_in_time_range(collection, start_ms, end_ms_exclusive, field)
        }

        fn retrieve<'py>(
            &self,
            py: Python<'py>,
            collection: &str,
            ids: Vec<String>,
        ) -> PyResult<Bound<'py, PyList>> {
            hits_to_list(py, &self.inner.retrieve(collection, &ids))
        }

        #[pyo3(signature = (collection, query_vector, limit = 10, payload_filter = None))]
        fn search<'py>(
            &self,
            py: Python<'py>,
            collection: &str,
            query_vector: Vec<f32>,
            limit: usize,
            payload_filter: Option<&Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyList>> {
            let filter = parse_filter(payload_filter)?;
            let hits = self
                .inner
                .search(collection, &query_vector, limit, filter.as_ref())
                .map_err(to_py_err)?;
            hits_to_list(py, &hits)
        }

        #[pyo3(signature = (collection, payload_filter = None, limit = 10, order_by = None))]
        fn scroll<'py>(
            &self,
            py: Python<'py>,
            collection: &str,
            payload_filter: Option<&Bound<'py, PyAny>>,
            limit: usize,
            order_by: Option<&str>,
        ) -> PyResult<Bound<'py, PyList>> {
            let filter = parse_filter(payload_filter)?;
            hits_to_list(
                py,
                &self.inner.scroll(collection, filter.as_ref(), limit, order_by),
            )
        }

        #[getter]
        fn total_points(&self) -> usize {
            self.inner.total_points()
        }

        fn collection_count(&self, name: &str) -> usize {
            self.inner.collection_count(name)
        }

        /// `(min, max)` of a payload field across a collection, or `None`.
        fn payload_value_range(
            &self,
            py: Python<'_>,
            collection: &str,
            field: &str,
        ) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
            match self.inner.payload_value_range(collection, field) {
                Ok(None) => Ok(None),
                Ok(Some((lo, hi))) => Ok(Some((value_to_py(py, &lo)?, value_to_py(py, &hi)?))),
                Err(e) => Err(to_py_err(e)),
            }
        }
    }
}

#[cfg(feature = "python")]
pub use python::PyLociStore;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store_with(name: &str, dim: usize, distance: &str) -> Store {
        let mut s = Store::new();
        s.create_collection(name, dim, distance);
        s
    }

    fn put(s: &mut Store, col: &str, id: &str, vector: &[f32], payload: Value) {
        s.upsert(col, vec![id.to_string()], vec![vector.to_vec()], vec![payload])
            .unwrap();
    }

    fn ids(points: &[PointOut]) -> Vec<&str> {
        points.iter().map(|p| p.id.as_str()).collect()
    }

    fn filt(v: Value) -> Filter {
        Filter::from_value(&v).unwrap()
    }

    // -- Filter semantics ---------------------------------------------------

    #[test]
    fn filter_exact_match() {
        let f = filt(json!({"scene": "s1"}));
        assert!(f.matches(&json!({"scene": "s1"})));
        assert!(!f.matches(&json!({"scene": "s2"})));
        assert!(!f.matches(&json!({})));
    }

    #[test]
    fn filter_numeric_equality_crosses_int_float() {
        let f = filt(json!({"v": 5}));
        assert!(f.matches(&json!({"v": 5.0})));
        let f = filt(json!({"v": 5.0}));
        assert!(f.matches(&json!({"v": 5})));
    }

    #[test]
    fn filter_null_condition_matches_missing_and_null() {
        // Python: payload.get(key) is None == None -> match.
        let f = filt(json!({"prev": null}));
        assert!(f.matches(&json!({})));
        assert!(f.matches(&json!({"prev": null})));
        assert!(!f.matches(&json!({"prev": "x"})));
    }

    #[test]
    fn filter_any_membership() {
        let f = filt(json!({"hid": {"any": [5, 15]}}));
        assert!(f.matches(&json!({"hid": 5})));
        assert!(f.matches(&json!({"hid": 5.0}))); // Python: 5.0 in [5]
        assert!(f.matches(&json!({"hid": 15})));
        assert!(!f.matches(&json!({"hid": 10})));
        assert!(!f.matches(&json!({}))); // missing -> None not in list
    }

    #[test]
    fn filter_any_membership_strings() {
        let f = filt(json!({"scale": {"any": ["frame", "patch"]}}));
        assert!(f.matches(&json!({"scale": "patch"})));
        assert!(!f.matches(&json!({"scale": "sequence"})));
    }

    #[test]
    fn filter_range_bounds() {
        let f = filt(json!({"ts": {"gte": 300, "lte": 500}}));
        assert!(!f.matches(&json!({"ts": 299})));
        assert!(f.matches(&json!({"ts": 300})));
        assert!(f.matches(&json!({"ts": 500})));
        assert!(!f.matches(&json!({"ts": 501})));
        assert!(!f.matches(&json!({}))); // None excluded

        let f = filt(json!({"v": {"gt": 1, "lt": 4}}));
        assert!(!f.matches(&json!({"v": 1})));
        assert!(f.matches(&json!({"v": 2})));
        assert!(f.matches(&json!({"v": 3})));
        assert!(!f.matches(&json!({"v": 4})));
    }

    #[test]
    fn filter_empty_condition_dict_matches_everything() {
        // Python: a dict condition without any/gte/lte/gt/lt runs no checks.
        let f = filt(json!({"ts": {}}));
        assert!(f.matches(&json!({"ts": 1})));
        assert!(f.matches(&json!({})));
    }

    #[test]
    fn filter_combined_conditions_are_conjunctive() {
        let f = filt(json!({"hid": {"any": [5]}, "ts": {"gte": 150}}));
        assert!(f.matches(&json!({"hid": 5, "ts": 200})));
        assert!(!f.matches(&json!({"hid": 5, "ts": 100})));
        assert!(!f.matches(&json!({"hid": 10, "ts": 200})));
    }

    // -- Score conventions --------------------------------------------------

    #[test]
    fn cosine_scores_higher_is_better() {
        let mut s = store_with("c", 4, "cosine");
        put(&mut s, "c", "a", &[1.0, 0.0, 0.0, 0.0], json!({}));
        put(&mut s, "c", "b", &[0.0, 1.0, 0.0, 0.0], json!({}));
        put(&mut s, "c", "c", &[0.9, 0.1, 0.0, 0.0], json!({}));
        let hits = s.search("c", &[1.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(ids(&hits), ["a", "c", "b"]);
        assert!((hits[0].score.unwrap() - 1.0).abs() < 1e-6);
        assert!(hits[2].score.unwrap().abs() < 1e-6);
    }

    #[test]
    fn dot_scores_are_raw_products() {
        let mut s = store_with("d", 2, "dot");
        put(&mut s, "d", "a", &[3.0, 0.0], json!({}));
        put(&mut s, "d", "b", &[1.0, 1.0], json!({}));
        let hits = s.search("d", &[1.0, 0.0], 2, None).unwrap();
        assert_eq!(ids(&hits), ["a", "b"]);
        assert_eq!(hits[0].score.unwrap(), 3.0);
        assert_eq!(hits[1].score.unwrap(), 1.0);
    }

    #[test]
    fn euclid_scores_are_negative_distances() {
        let mut s = store_with("e", 2, "euclidean");
        put(&mut s, "e", "a", &[1.0, 0.0], json!({}));
        put(&mut s, "e", "b", &[10.0, 10.0], json!({}));
        let hits = s.search("e", &[1.0, 0.0], 2, None).unwrap();
        assert_eq!(ids(&hits), ["a", "b"]);
        assert_eq!(hits[0].score.unwrap(), 0.0); // exact match: -0 distance
        assert!(hits[1].score.unwrap() < 0.0); // farther -> more negative
    }

    #[test]
    fn unknown_distance_falls_back_to_euclid() {
        assert_eq!(Distance::parse("whatever"), Distance::Euclid);
    }

    #[test]
    fn zero_query_vector_cosine_scores_zero() {
        let mut s = store_with("c", 2, "cosine");
        put(&mut s, "c", "a", &[1.0, 0.0], json!({}));
        let hits = s.search("c", &[0.0, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].score.unwrap(), 0.0);
    }

    #[test]
    fn zero_stored_vector_norm_substituted_with_one() {
        let mut s = store_with("c", 2, "cosine");
        put(&mut s, "c", "z", &[0.0, 0.0], json!({}));
        let hits = s.search("c", &[1.0, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].score.unwrap(), 0.0); // dot=0 / (1 * 1) = 0
    }

    #[test]
    fn search_respects_limit_and_filter() {
        let mut s = store_with("c", 2, "cosine");
        for i in 0..20 {
            put(
                &mut s,
                "c",
                &format!("p{i}"),
                &[1.0, i as f32 * 0.01],
                json!({"i": i}),
            );
        }
        let f = filt(json!({"i": {"gte": 10}}));
        let hits = s.search("c", &[1.0, 0.0], 5, Some(&f)).unwrap();
        assert_eq!(hits.len(), 5);
        for h in &hits {
            assert!(h.payload["i"].as_i64().unwrap() >= 10);
        }
    }

    #[test]
    fn search_query_dimension_mismatch_is_error() {
        let s = store_with("c", 4, "cosine");
        assert!(matches!(
            s.search("c", &[1.0], 1, None),
            Err(StoreError::QueryDimension { got: 1, expected: 4 })
        ));
    }

    #[test]
    fn search_missing_collection_is_empty() {
        let s = Store::new();
        assert!(s.search("nope", &[1.0], 5, None).unwrap().is_empty());
    }

    // -- Upsert / retrieve / tombstones -------------------------------------

    #[test]
    fn upsert_overwrites_in_place() {
        let mut s = store_with("c", 2, "cosine");
        put(&mut s, "c", "a", &[1.0, 0.0], json!({"x": 1}));
        put(&mut s, "c", "a", &[0.0, 1.0], json!({"x": 2}));
        assert_eq!(s.collection_count("c"), 1);
        let got = s.retrieve("c", &["a".to_string()]);
        assert_eq!(got[0].vector, vec![0.0, 1.0]);
        assert_eq!(got[0].payload, json!({"x": 2}));
    }

    #[test]
    fn upsert_dimension_mismatch_keeps_earlier_points() {
        let mut s = store_with("c", 2, "cosine");
        let err = s
            .upsert(
                "c",
                vec!["a".into(), "b".into()],
                vec![vec![1.0, 0.0], vec![1.0]],
                vec![json!({}), json!({})],
            )
            .unwrap_err();
        assert!(matches!(err, StoreError::PointDimension { .. }));
        assert_eq!(s.collection_count("c"), 1); // "a" landed, "b" rejected
        let msg = err.to_string();
        assert!(msg.contains("dimension 1"));
        assert!(msg.contains("expected 2"));
    }

    #[test]
    fn upsert_missing_collection_is_error() {
        let mut s = Store::new();
        assert!(matches!(
            s.upsert("nope", vec!["a".into()], vec![vec![1.0]], vec![json!({})]),
            Err(StoreError::MissingCollection(_))
        ));
    }

    #[test]
    fn tombstone_delete_then_reuse_row() {
        let mut s = store_with("c", 2, "cosine");
        put(&mut s, "c", "a", &[1.0, 0.0], json!({"k": "a"}));
        put(&mut s, "c", "b", &[0.0, 1.0], json!({"k": "b"}));
        assert_eq!(s.delete_points("c", &["a".to_string(), "zz".to_string()]), 1);
        assert_eq!(s.collection_count("c"), 1);
        assert!(s.retrieve("c", &["a".to_string()]).is_empty());

        // Dead rows are invisible to search and scroll.
        let hits = s.search("c", &[1.0, 0.0], 10, None).unwrap();
        assert_eq!(ids(&hits), ["b"]);
        assert_eq!(ids(&s.scroll("c", None, 10, None)), ["b"]);

        // A new insert reuses the tombstoned row: the arena does not grow.
        let rows_before = s.collections["c"].rows();
        put(&mut s, "c", "c2", &[0.5, 0.5], json!({"k": "c2"}));
        assert_eq!(s.collections["c"].rows(), rows_before);
        assert_eq!(s.collection_count("c"), 2);
        let hits = s.search("c", &[0.5, 0.5], 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "c2");
    }

    #[test]
    fn delete_points_in_time_range_is_end_exclusive() {
        let mut s = store_with("c", 1, "cosine");
        for i in 0..5 {
            put(&mut s, "c", &format!("p{i}"), &[1.0], json!({"timestamp_ms": i * 100}));
        }
        assert_eq!(s.delete_points_in_time_range("c", 100, 300, "timestamp_ms"), 2);
        let left: Vec<i64> = s
            .scroll("c", None, 10, None)
            .iter()
            .map(|p| p.payload["timestamp_ms"].as_i64().unwrap())
            .collect();
        assert_eq!(left, vec![0, 300, 400]);
    }

    #[test]
    fn delete_points_in_time_range_skips_missing_field() {
        let mut s = store_with("c", 1, "cosine");
        put(&mut s, "c", "a", &[1.0], json!({"timestamp_ms": 100}));
        put(&mut s, "c", "b", &[1.0], json!({}));
        assert_eq!(s.delete_points_in_time_range("c", 0, 1000, "timestamp_ms"), 1);
        assert_eq!(ids(&s.scroll("c", None, 10, None)), ["b"]);
    }

    #[test]
    fn delete_on_missing_collection_returns_zero() {
        let mut s = Store::new();
        assert_eq!(s.delete_points("nope", &["a".to_string()]), 0);
        assert_eq!(s.delete_points_in_time_range("nope", 0, 100, "timestamp_ms"), 0);
    }

    // -- set_payload ---------------------------------------------------------

    #[test]
    fn set_payload_merges_top_level_keys() {
        let mut s = store_with("c", 1, "cosine");
        put(&mut s, "c", "a", &[1.0], json!({"x": 1, "meta": {"a": 1}}));
        s.set_payload("c", "a", json!({"y": 2, "meta": {"b": 2}})).unwrap();
        let got = s.retrieve("c", &["a".to_string()]);
        // Top-level replacement, not deep merge (dict.update semantics).
        assert_eq!(got[0].payload, json!({"x": 1, "y": 2, "meta": {"b": 2}}));
        // Missing point: silent no-op.
        s.set_payload("c", "nope", json!({"y": 2})).unwrap();
    }

    // -- Scroll --------------------------------------------------------------

    #[test]
    fn scroll_unordered_early_exits_at_limit() {
        let mut s = store_with("c", 1, "cosine");
        for i in 0..10 {
            put(&mut s, "c", &format!("p{i}"), &[1.0], json!({"i": i}));
        }
        assert_eq!(s.scroll("c", None, 3, None).len(), 3);
        let f = filt(json!({"i": {"gte": 5}}));
        assert_eq!(s.scroll("c", Some(&f), 2, None).len(), 2);
    }

    #[test]
    fn scroll_order_by_ascending_with_missing_field_as_zero() {
        let mut s = store_with("c", 1, "cosine");
        put(&mut s, "c", "a", &[1.0], json!({"ts": 100}));
        put(&mut s, "c", "b", &[1.0], json!({"ts": 300}));
        put(&mut s, "c", "c", &[1.0], json!({"ts": 200}));
        put(&mut s, "c", "d", &[1.0], json!({}));
        let out = s.scroll("c", None, 10, Some("ts"));
        assert_eq!(ids(&out), ["d", "a", "c", "b"]); // missing ts sorts as 0
        let out = s.scroll("c", None, 2, Some("ts"));
        assert_eq!(ids(&out), ["d", "a"]);
    }

    // -- Stats ---------------------------------------------------------------

    #[test]
    fn total_points_and_collection_count_track_live_rows() {
        let mut s = store_with("c1", 1, "cosine");
        s.create_collection("c2", 1, "cosine");
        put(&mut s, "c1", "a", &[1.0], json!({}));
        put(&mut s, "c2", "b", &[1.0], json!({}));
        put(&mut s, "c2", "c", &[1.0], json!({}));
        assert_eq!(s.total_points(), 3);
        s.delete_points("c2", &["b".to_string()]);
        assert_eq!(s.total_points(), 2);
        assert_eq!(s.collection_count("c2"), 1);
        s.delete_collection("c2");
        assert_eq!(s.total_points(), 1);
        assert!(!s.collection_exists("c2"));
    }

    #[test]
    fn payload_value_range_min_max() {
        let mut s = store_with("c", 1, "cosine");
        assert_eq!(s.payload_value_range("c", "ts").unwrap(), None);
        assert_eq!(s.payload_value_range("nope", "ts").unwrap(), None);
        put(&mut s, "c", "a", &[1.0], json!({"ts": 300}));
        put(&mut s, "c", "b", &[1.0], json!({"ts": 100}));
        put(&mut s, "c", "c", &[1.0], json!({"ts": 200}));
        put(&mut s, "c", "d", &[1.0], json!({})); // skipped
        let (lo, hi) = s.payload_value_range("c", "ts").unwrap().unwrap();
        assert_eq!((lo, hi), (json!(100), json!(300)));
        // Tombstoned extremes drop out of the range.
        s.delete_points("c", &["b".to_string()]);
        let (lo, _) = s.payload_value_range("c", "ts").unwrap().unwrap();
        assert_eq!(lo, json!(200));
    }

    #[test]
    fn create_collection_is_idempotent() {
        let mut s = store_with("c", 4, "cosine");
        put(&mut s, "c", "a", &[1.0, 0.0, 0.0, 0.0], json!({}));
        s.create_collection("c", 8, "dot"); // no-op: keeps dim 4 + contents
        assert_eq!(s.collection_count("c"), 1);
        assert!(s.search("c", &[1.0, 0.0, 0.0, 0.0], 1, None).is_ok());
    }

    #[test]
    fn kernels_match_naive_reference() {
        // Exercise the remainder path (len not divisible by 8).
        let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..19).map(|i| (18 - i) as f32 * 0.25).collect();
        let naive_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let naive_l2: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((dot_f32(&a, &b) - naive_dot).abs() < 1e-3);
        assert!((l2_sq_f32(&a, &b) - naive_l2).abs() < 1e-3);
    }
}
