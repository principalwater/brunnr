<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backends

`MemoryBackend` defines the durable memory seam:

- `store`: persist durable memory.
- `find`: retrieve relevant memory.
- `hybrid_rrf`: fuse keyword and vector retrieval channels with reciprocal rank fusion.
- `get_node`: drill down by deterministic `node_id` or memory id.

`VectorStore` is lower level. It owns only collection/index creation, upsert, search, get, and
capability reporting. Embedding, L0-L3 memory tiering, payload schema, and RRF live in
`VectorMemoryBackend<V: VectorStore>`.

## Hybrid And RRF

Brunnr uses reciprocal rank fusion with `rank_constant = 60.0` by default. A document at rank `r`
contributes `1 / (rank_constant + r)` to its fused score. Duplicate `node_id` results across
channels merge into one hit, preserving deterministic drill-down.

Vector engines may advertise `supports_server_side_hybrid`. When they do, `VectorMemoryBackend`
delegates hybrid search to the engine. When they do not, Brunnr runs keyword and vector searches
separately and fuses them with the same RRF implementation.

## FilesBackend

`FilesBackend` stores date-tagged markdown records under `.brunnr/memory/YYYY-MM-DD/<id>.md`.

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

Hybrid behavior:

- `supports_server_side_hybrid = false`.
- Brunnr runs FTS5 BM25 keyword search and sqlite-vec vector search separately.
- Results are fused by Brunnr RRF.

Default CLI config stores the SQLite file at `.brunnr/memory.sqlite3` when `backend = "sqlite-vec"`.

## QdrantBackend

`QdrantBackend` is `VectorMemoryBackend<QdrantVectorStore>`.

Storage:

- Qdrant owns the collection and vector index.
- Brunnr stores the normalized memory payload in Qdrant point payload.
- The first shared embedding default is pinned to `intfloat/multilingual-e5-small` with 384
  dimensions.

Hybrid behavior:

- `supports_server_side_hybrid = false` today because Brunnr does not yet configure a sparse
  vector channel.
- Brunnr runs vector search through Qdrant and keyword fallback over Qdrant payload scroll, then
  fuses with RRF.
- Future sparse support can flip capabilities without changing `MemoryBackend` callers.

Run a local Qdrant for development:

```shell
docker compose -f deploy/qdrant/compose.yml up -d
QDRANT_URL=http://127.0.0.1:6334 cargo test -p mimisbrunnr --features qdrant --test qdrant -- --ignored
```

Do not hardcode Qdrant hosts in tests or docs examples. Use `QDRANT_URL` or `qdrant_url` in
`brunnr.toml`.

## Future Backends

`TencentDBBackend` remains a reserved backend name behind the same `MemoryBackend` trait.
