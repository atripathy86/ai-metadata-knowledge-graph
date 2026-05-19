# Known Issues

This file tracks known bugs, performance problems, and code-quality issues identified during review.
Items marked **Document only** are architectural improvements planned for a future milestone.

---

## Bugs

| # | Title | File(s) | Description |
|---|-------|---------|-------------|
| 1 | `model.py` queries `Dataset` nodes instead of `Model` nodes | `aimkg-recommender-UI/utils/model.py` lines 64–99 | `get_result_pipelines()` iterates over `model_ids` but anchors the Cypher query on `Dataset` nodes. `get_explanations()` also labels results `'Label': 'Dataset'`. Includes a `# TODO: change the query here` comment acknowledging the issue. |
| 2 | `num_res` parameter silently overridden to `3` | `aimkg-recommender-UI/utils/task.py` line 204, `aimkg-recommender-UI/utils/model.py` line 106 | Both `get_similar_tasks()` and `get_similar_models()` accept a `num_res` parameter from the caller but immediately reassign it to the literal `3`, so the API always returns exactly 3 results regardless of the `num_recommendations` field sent by the client. |
| 3 | No null check after `find_file_path()` — opaque crash on missing embedding files | `aimkg-recommender-UI/utils/task.py` lines 214–216, `aimkg-recommender-UI/utils/model.py` lines 110–112 | `find_file_path()` returns `None` when the `.h5` file cannot be found. The return value is passed directly into `h5py.File()`, producing a cryptic `TypeError` instead of a helpful `FileNotFoundError`. Same pattern exists in `dataset.py` and `pipeline.py`. |
| 4 | Token splitting uses different delimiters across modules | `aimkg-recommender-UI/utils/task.py` line 105, `aimkg-recommender-UI/utils/model.py` line 36, `aimkg-recommender-UI/utils/dataset.py` | `create_tokens()` splits on `" "` (space) in `task.py` but on `"-"` (hyphen) in `model.py` and `dataset.py`. This makes IOU-based similarity scores incomparable across entity types and causes missed matches. |
| 5 | Typos in task category and modality vocabulary lists | `aimkg-recommender-UI/utils/task.py` lines 7, 9 | `'reccommendation'` (double `c`) and `'visualizaiton'` (transposed letters) in `task_category_vocab`. Tasks named "recommendation" or "visualization" are incorrectly labelled with category/modality `"none"`. |

---

## Performance

| # | Title | File(s) | Description |
|---|-------|---------|-------------|
| 8 | Embedding files reloaded from disk on every API request | `aimkg-recommender-UI/utils/task.py`, `model.py`, `dataset.py`, `pipeline.py` | Each similarity call opens and fully reads the corresponding `.h5` file (the model embedding file alone is ~833 MB). Should be loaded once at startup and cached in memory. |
| 9 | Full graph node set fetched from Neo4j on every recommendation request | `aimkg-recommender-UI/utils/task.py`, `model.py`, `dataset.py` | `get_similar_tasks()` calls `get_tasks()` on every request, which runs `MATCH (n:Task) RETURN properties(n)` — a full table scan into Python memory. Same pattern for Model and Dataset nodes. Should be cached at startup. |
| 10 | N+1 Neo4j queries when fetching result node details | `aimkg-recommender-UI/utils/task.py` lines 244–246, `aimkg-recommender-UI/utils/model.py` lines 135–136 | After finding the top-N similar IDs, one separate Neo4j query is issued per ID. Should be replaced with a single batched `WHERE n.itemID IN $ids` query. |
| 11 | `find_file_path()` does recursive filesystem scan on every request | `aimkg-recommender-UI/utils/task.py` lines 52–56, `aimkg-recommender-UI/utils/model.py` lines 28–31, `aimkg-recommender-UI/utils/dataset.py` | Uses `glob.iglob(..., recursive=True)` at request time to locate `.h5` files. Paths should be resolved once at startup or via environment variables. |

---

## Code Quality

| # | Title | File(s) | Description |
|---|-------|---------|-------------|
| 12 | Vocabulary lists and helper functions duplicated across four modules | `task.py`, `model.py`, `dataset.py`, `pipeline.py` | `task_category_vocab`, modality vocab lists, `compute_category()`, `compute_modality()`, `create_tokens()`, `compute_IOU()`, `find_file_path()`, and Neo4j connection boilerplate are copy-pasted into all four files. Any bug fix must be applied in four places. Should be extracted into a shared `similarity_utils.py`. |
| 13 | Module-level side effects make imports slow and untestable | `task.py`, `model.py`, `dataset.py`, `pipeline.py` | `SentenceTransformer(...)` (model load) and `Neo4jConnection(...)` (DB connection) run at import time. Unit tests cannot import these modules without a live Neo4j instance and the transformer model. Should be moved to lazy init or a startup hook in `app.py`. |
| 14 | Debug `print()` statements throughout production code; logging is disabled | `aimkg-recommender-UI/app.py`, `task.py`, `model.py` | `logging` setup is commented out in `app.py`. Numerous `print()` calls (e.g. `app.py:40,44,45`, `task.py:235,243`, `model.py:124,133`) remain in production paths with no timestamps or log levels. Should use the `logging` module. |
| 15 | `get_explanations()` in `model.py` contains placeholder/stub content | `aimkg-recommender-UI/utils/model.py` lines 89–99 | Copied from `task.py` but never adapted: returns `'Label': 'Dataset'`, hardcoded empty strings for similarity scores, and has a `# TODO: Modify as per model features` comment. Results in incorrect labels on the UI for model-based recommendations. |
| 16 | No version pins in `requirements.txt` files | `aimkg-recommender-UI/requirements.txt`, `ai-pipeline-knowledge-graph/requirements.txt`, `ai-pipeline-datasources/requirements.txt` | All dependencies (Flask, neo4j, sentence-transformers, torch, h5py) are listed without version constraints, making builds non-reproducible and vulnerable to silent breaking changes. |
| 17 | Port hardcoded in `app.py` instead of read from environment | `aimkg-recommender-UI/app.py` line 82 | `app.run(port=9089)` is hardcoded. Should read from `os.getenv("APP_PORT", 9089)` to support different deployment environments without source changes. |
| 18 | Docker base image uses end-of-life, unpinned Python 3.9 | `aimkg-recommender-UI/Dockerfile` line 1 | `python:3.9-slim` is past end-of-life (Oct 2025) and receives no security patches. The floating tag also makes builds non-reproducible. Should be upgraded to `python:3.12-slim`. |
| 22 | Wrong progress label in `compute_embeddings.py` for model embeddings | `aimkg-recommender-UI/utils/compute_embeddings.py` | `model_embeddings()` prints `"Computing dataset embeddings.."` — the label is copy-pasted and not updated. Should print `"Computing model embeddings.."`. |
| 23 | Stale TODO comment in `d3_graph.py` — deduplication already implemented | `aimkg-recommender-UI/utils/d3_graph.py` line 19 | `# TODO: Handle node duplicates` exists a few lines above the code that already handles node deduplication. Should be removed. |

---

## Enhancements (Document Only — Not Scheduled)

These items require new infrastructure and are planned for a future milestone.

| # | Title | Description |
|---|-------|-------------|
| 19 | Add unit and integration tests | No test files exist anywhere in the project. Critical logic (`compute_IOU`, `compute_category`, `create_tokens`, `neo4j_to_d3`, Flask endpoints) has no automated coverage. Prerequisite: resolve issue #13 (module-level side effects). |
| 20 | Add input validation to Flask API endpoints | `app.py` performs no validation before passing request data to downstream functions. Missing fields, wrong types, or oversized strings all produce unhandled `500` errors. Endpoints affected: `POST /recommendation/query`, `POST /search/query`, `POST /search/cypher_query`. |
| 21 | Define and document a consistent API response schema | Response format varies between endpoints; no structured error responses; no OpenAPI spec. Should define a consistent envelope (e.g. `{"data": ..., "error": null}`), appropriate HTTP status codes, an `openapi.yaml` spec, and a `/health` endpoint. |
