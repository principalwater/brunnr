<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backends

`MemoryBackend` defines the durable memory seam:

- `store`: persist durable memory.
- `find`: retrieve relevant memory.
- `hybrid_rrf`: fuse keyword and vector retrieval channels with reciprocal rank fusion.
- `get_node`: drill down by deterministic `node_id` or memory id.

`VectorStore` is lower level. It owns only collection/index creation, upsert, search, get, and
capability reporting. Embedding, **chunk-on-store**, **small-to-big retrieval**, L0-L3 memory
tiering, payload schema, and RRF all live in `VectorMemoryBackend<V: VectorStore>` — so every
vector engine inherits them for free.

Because chunking and small-to-big are done once in `VectorMemoryBackend`, **any** vector backend
automatically gets bounded, coherent retrieval: `store` splits content into ~400-token chunks with
parent linkage, and `find` matches on the precise chunk but returns the surrounding **parent-section
window** (same-parent hits collapsed into one), bounded by an adaptive budget — with the full
document reachable by `node_id` drill-down. See [memory.md §3.5](memory.md) for the chunking +
small-to-big algorithm and the adaptive budget.

## Semantic query cache

`CachingMemoryBackend<B, V>` wraps any `MemoryBackend` with a `SemanticCache`: a repeated or
**paraphrased** query whose embedding is within a cosine threshold of a recent query is served
from cache instead of re-running search — cutting embedding + ANN work in agent loops that recall
around the same task. The cache is bounded (LRU) with an optional TTL; only plain text queries are
cached (a `node_id` or tenancy filter bypasses it), and every `store` clears the cache so a new
write is never hidden by a stale read. `SemanticCache` is embedder-agnostic via the
`QueryVectorizer` seam; under feature `vector` the pinned `TextEmbedder` is adapted to it.

## Hybrid And RRF

Artesian uses reciprocal rank fusion with `rank_constant = 60.0` by default. A document at rank `r`
contributes `1 / (rank_constant + r)` to its fused score. Duplicate `node_id` results across
channels merge into one hit, preserving deterministic drill-down.

Vector engines may advertise `supports_server_side_hybrid`. When they do, `VectorMemoryBackend`
delegates hybrid search to the engine. When they do not, Artesian runs keyword and vector searches
separately and fuses them with the same RRF implementation.

## FilesBackend

`FilesBackend` stores OKF markdown records under `.artesian/memory/YYYY-MM-DD/<id>.md`. It writes
YAML `---` frontmatter with required `type: memory`, recommended `tags`/`timestamp`, and Artesian
extensions such as `node_id`, `tier`, and optional tenancy fields. It still reads legacy TOML
`+++` records.

Hybrid behavior:

- Keyword search is local text matching over content, tags, and metadata.
- Vector search is not available.
- `hybrid_rrf` uses the default `MemoryBackend` implementation, so both channels are keyword
  searches unless a caller supplies different query text.

## SqliteVecBackend

`SqliteVecBackend` is `VectorMemoryBackend<SqliteVecVectorStore>`.

Storage:

- `rusqlite` owns the local database file.
- `sqlite-vec` provides the `vec0` vector table.
- SQLite FTS5 provides keyword/BM25 search.
- Payload JSON is stored beside the vector rows for deterministic `get_node` and idempotent
  backfill.
- Connections use WAL and `busy_timeout`; writers are serialized inside the process.

Hybrid behavior:

- `supports_server_side_hybrid = false`.
- Artesian runs FTS5 BM25 keyword search and sqlite-vec vector search separately.
- Results are fused by Artesian RRF.

Default CLI config stores the SQLite file at `.artesian/memory.sqlite3` when `backend = "sqlite-vec"`.

## QdrantBackend

`QdrantBackend` is `VectorMemoryBackend<QdrantVectorStore>`.

Storage:

- Qdrant owns the collection and vector index.
- Artesian stores the normalized memory payload in Qdrant point payload.
- Upserts use `wait=true` for read-after-write behavior.
- Payload indexes are created for `node_id` and tenancy fields.
- The first shared embedding default is pinned to `intfloat/multilingual-e5-small` with 384
  dimensions.

Hybrid behavior:

- `supports_server_side_hybrid = false` today because Artesian does not yet configure a sparse
  vector channel.
- Artesian runs vector search through Qdrant and keyword fallback over Qdrant payload scroll, then
  fuses with RRF.
- Future sparse support can flip capabilities without changing `MemoryBackend` callers.

Run a local Qdrant for development:

```shell
docker compose -f deploy/qdrant/compose.yml up -d
QDRANT_URL=http://127.0.0.1:6333 \
  cargo test -p aquifer --features qdrant --test qdrant -- --ignored
```

Do not hardcode Qdrant hosts in code. On default ports, Artesian accepts one `QDRANT_URL` /
`qdrant_url`: `:6333` is treated as REST and derives gRPC `:6334`; `:6334` derives REST `:6333`.
Use `QDRANT_REST_URL` / `qdrant_rest_url` only when the REST API is not the default sibling of the
configured gRPC endpoint. CLI setup/import preflights both endpoints before writing memory.

## PgVectorBackend

`PgVectorStore` (feature `pgvector`) adapts PostgreSQL + pgvector to the `VectorStore` trait, so a
team already running Postgres can use it as the shared memory store with no extra service. It is
exercised by a gated integration test (`#[ignore]` unless the database URL is set).

## Adding a vector backend (the `VectorStore` adapter pattern)

A new vector engine is a thin adapter, not a fork. Implement the six `VectorStore` methods and the
generic `VectorMemoryBackend<V>` gives you embedding, chunk-on-store, RRF hybrid, reranking, L0-L3
tiering, payload tenancy, and `node_id` drill-down for free — no core change.

Worked example: `crates/aquifer/src/pgvector.rs` (feature `pgvector`).

1. **Feature + deps** — add a Cargo feature and optional client deps; gate the module with
   `#[cfg(feature = "<name>")]`.
2. **Store type** — `struct YourVectorStore` holding the connection/config, with `connect(config)`.
3. **`impl VectorStore`**:
   - `ensure_collection` — create the collection/table with the right vector dimension and distance;
   - `ensure_payload_index` — index the tenancy/keyword payload fields;
   - `upsert` — write points `{ id, vector, payload }`;
   - `search` — vector ANN and/or keyword, honoring the normalized `Filter` (eq / in / range /
     exists, with must / should / must_not);
   - `get` — fetch a point by id (used for dedup and drill-down);
   - `capabilities` — advertise e.g. `supports_server_side_hybrid`; return `false` and Artesian runs
     RRF itself.
   Optionally `impl VectorCollectionAdmin` for snapshot / migrate support.
4. **Alias** — `pub type YourBackend = VectorMemoryBackend<YourVectorStore>;`.
5. **Gated test** — a `#[ignore]` integration test proving store → find → hybrid against a live
   instance; read the host from an env var, never hardcode it.

Keep the trait minimal: do not push embedding, RRF, or chunking into the adapter — those stay in
`VectorMemoryBackend` so every engine behaves identically. Never log credentials.

## Reserved backends

`TencentDBBackend` remains a reserved backend name behind the same `MemoryBackend` trait.
