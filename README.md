# Product Recommendation System in Node.js

This is a Node.js product recommendation prototype.

I built it to compare different retrieval methods step by step instead of jumping directly to one final approach.

Right now the project includes:

- exact brute-force search
- `k`-means clustering
- binary space partitioning (BSP)
- HNSW with `hnswlib-node`
- product quantization (PQ) for compression

The main idea is simple:

- each product is treated like a vector
- the user/query is also treated like a vector
- we search for the nearest product vectors
- then we compare speed, recall, and storage

## What is done now

This repo is already beyond just the starting stage.

Current work done:

- sample product catalog with generated embeddings
- real catalog loading from JSON or JSONL files
- recommendation API with startup-loaded catalog
- exact baseline recommender
- recall, latency, and storage comparison
- `k`-means recommender
- BSP recommender
- HNSW recommender
- PQ compression flow
- benchmark/demo script

So this is the current working prototype, not only Day 1 work.

## Tech used

- Node.js 20+
- ES modules
- `hnswlib-node` as an optional dependency

## Project files

- `src/demo.js` - runs the benchmark and prints sample recommendations
- `src/data/sampleCatalog.js` - creates sample product vectors and query vectors
- `src/recommenders/exactRecommender.js` - exact baseline search
- `src/recommenders/kmeansRecommender.js` - cluster based retrieval
- `src/recommenders/bspRecommender.js` - BSP tree based retrieval
- `src/recommenders/hnswRecommender.js` - HNSW based retrieval
- `src/recommenders/pqRecommender.js` - compressed retrieval using PQ
- `src/compression/productQuantizer.js` - PQ training and encoding logic
- `src/algorithms/kmeans.js` - shared `k`-means logic

## How the system works

Flow is:

1. create product vectors
2. create query vectors
3. search for nearest products
4. compare methods using recall and speed

The current demo uses synthetic data so the project can run immediately.

## Methods used

### 1. Exact search

This is the baseline.

- checks all product vectors
- gives the ground truth nearest neighbors
- used to compare the approximate methods

### 2. `k`-means

Here the product vectors are first grouped into clusters.

For a query:

- find the nearest centroids
- collect products from those clusters
- rerank only those candidates

This reduces the number of products we need to search directly.

### 3. BSP

This method builds a binary space partition tree.

- the vector space is split recursively
- products are stored in leaves
- query search visits the most promising leaves first

This gives a tree-based approximate search baseline.

### 4. HNSW

This uses `hnswlib-node` for approximate nearest neighbor search.

Among the current methods, this usually gives the best speed and recall balance.

### 5. PQ compression

This part is for saving storage.

The steps are:

1. split each vector into smaller parts
2. run `k`-means on each part
3. store centroid ids instead of full vectors
4. estimate distances using the compressed codes
5. rerank top candidates

This helps reduce memory while still keeping useful retrieval quality.

## How to run

Install dependencies:

```bash
npm install
```

Run the main demo:

```bash
npm run demo
```

Run a smaller smoke test:

```bash
npm run smoke
```

Run a larger benchmark:

```bash
npm run demo:large
```

Run the API server:

```bash
npm run server
```

Run the API with the sample real catalog:

```bash
npm run server:sample
```

Custom run example:

```bash
node src/demo.js --products 5000 --queries 50 --topK 10 --clusters 48 --subspaces 4 --codebook 32
```

Real data example:

```bash
node src/demo.js --data examples/real-catalog.sample.json --topK 3
```

If you have products and queries in separate files:

```bash
node src/demo.js --data your-products.json --queries-file your-queries.json --topK 10
```

Supported input formats:

- JSON array of products
- JSON object with `products` and optional `queries`
- JSONL / NDJSON for product records

Accepted vector fields:

- `vector`
- `embedding`
- `values`

The loader normalizes vectors by default.
If you want to keep the original scale, run with:

```bash
node src/demo.js --data your-products.json --keep-scale
```

## API

The API loads product data once when the server starts.

Main route:

- `POST /recommend`

Useful route:

- `GET /health`

Example request:

```json
{
  "vector": [0.91, 0.8, 0.13, 0.1, 0.62, 0.56, 0.12, 0.08],
  "topK": 3,
  "category": "electronics",
  "minPrice": 100,
  "maxPrice": 700
}
```

Example curl:

```bash
curl -X POST http://localhost:3000/recommend -H "Content-Type: application/json" -d "{\"vector\":[0.91,0.8,0.13,0.1,0.62,0.56,0.12,0.08],\"topK\":3,\"category\":\"electronics\"}"
```

Request fields:

- `vector` or `embedding`
- `topK`
- `category` as a string or list
- `minPrice`
- `maxPrice`
- optional `method`: `auto`, `hnsw`, or `exact`

How it works:

- by default it uses HNSW if available
- if HNSW is not available, it falls back to exact search
- if filters are used, it does exact search on the filtered product set

## What the benchmark checks

The demo compares:

- `recall@k`
- average query latency
- candidate count
- PQ storage estimate

## HNSW note

`hnswlib-node` is optional.

If it installs properly, HNSW runs normally.
If not, the rest of the project still works.

## For real project use

To make this work on a real product catalog, replace:

- `src/data/sampleCatalog.js` with real product embeddings
- `src/demo.js` with an API route or service entry point
- synthetic queries with real user or session embeddings
- simple reranking with business rules like stock, price, or category filters

## What I finished today

This is the current working prototype.

It includes:

- Node.js recommendation system setup
- real data loading for benchmark runs
- recommendation API with basic filters
- exact search baseline
- `k`-means retrieval
- BSP retrieval
- HNSW retrieval
- product quantization compression
- benchmark and demo code

## Next steps

Next practical steps are:

- load real product data
- add filtering and reranking
- benchmark on real catalog data
- choose the best retrieval method for production
