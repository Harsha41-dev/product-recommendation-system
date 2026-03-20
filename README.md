# Product Recommendation System in Node.js

This repository contains a working product recommendation prototype built in Node.js.

It includes:

- exact brute-force retrieval
- `k`-means clustering for candidate generation
- binary space partitioning (BSP) tree search
- HNSW-based approximate nearest neighbor search using `hnswlib-node`
- `k`-means-based compression using product quantization (PQ)

The current version is meant to be a practical prototype you can run, explain, and extend.

## Current Status

This push includes a full prototype, not just Day 1 work.

What is already done:

- sample product catalog generation with vector embeddings
- exact baseline recommender
- recall, latency, and storage comparison
- `k`-means recommender
- BSP recommender
- HNSW recommender
- PQ compression flow
- runnable benchmark demo

## Tech Stack

- Node.js 20+
- ES modules
- optional native dependency: `hnswlib-node`

## Project Structure

Main files:

- `src/demo.js` runs the benchmark and sample recommendations
- `src/data/sampleCatalog.js` creates sample product vectors and query vectors
- `src/recommenders/exactRecommender.js` exact cosine-similarity baseline
- `src/recommenders/kmeansRecommender.js` cluster-first retrieval
- `src/recommenders/bspRecommender.js` BSP tree retrieval
- `src/recommenders/hnswRecommender.js` HNSW retrieval
- `src/recommenders/pqRecommender.js` compressed retrieval using PQ
- `src/compression/productQuantizer.js` PQ training and encoding logic
- `src/algorithms/kmeans.js` shared `k`-means implementation

## How It Works

The recommendation flow is:

1. represent each product as a vector
2. represent a user/session/query as a vector
3. search for nearest product vectors
4. compare retrieval quality and speed across methods

The demo uses synthetic embeddings so the project works out of the box.

## Retrieval Methods

### 1. Exact Search

This is the brute-force baseline.

- checks every product vector
- gives ground-truth nearest neighbors
- used to measure recall for approximate methods

### 2. `k`-Means Retrieval

This method:

- clusters product vectors
- finds the nearest centroids for a query
- searches only inside those cluster members

This is a simple way to reduce the candidate set before reranking.

### 3. BSP Retrieval

This method:

- splits the vector space recursively
- stores products in leaf nodes
- visits the most promising leaves first

It is useful as a tree-based approximate search baseline.

### 4. HNSW Retrieval

This method uses `hnswlib-node` for graph-based approximate nearest neighbor search.

It usually gives the best speed/recall tradeoff among the methods in this prototype.

### 5. PQ Compression

This method compresses product vectors to save space.

It works like this:

1. split each vector into smaller sub-vectors
2. run `k`-means on each subspace
3. replace each sub-vector with a centroid id
4. estimate distances using compressed codes
5. rerank the top candidates

This helps reduce storage while keeping useful retrieval quality.

## Setup

Install dependencies:

```bash
npm install
```

Run the demo:

```bash
npm run demo
```

Quick smoke test:

```bash
npm run smoke
```

Larger benchmark:

```bash
npm run demo:large
```

You can also run with custom options:

```bash
node src/demo.js --products 5000 --queries 50 --topK 10 --clusters 48 --subspaces 4 --codebook 32
```

## Benchmark Metrics

The demo compares methods using:

- `recall@k`
- average query latency
- candidate count
- storage estimate for PQ

## Notes About HNSW

`hnswlib-node` is listed as an optional dependency.

If it installs successfully, the HNSW benchmark runs normally.
If it does not install on a machine, the rest of the project still works.

## What To Replace For Real Use

For a real recommendation system, replace:

- `src/data/sampleCatalog.js` with your real product embeddings
- `src/demo.js` with an API route, batch job, or service entry point
- synthetic query vectors with real user/session/query embeddings
- simple reranking with real business rules such as stock, price, or category filters

## What To Do Next

Good next steps after this push:

- load real product data
- expose retrieval through an API
- add filtering and reranking
- benchmark on your actual catalog
- choose one retrieval strategy for production

## Commit Scope For Today

This README matches the current repo state as a complete prototype milestone.

If you push today, you are pushing:

- the initial Node.js recommendation prototype
- all retrieval approaches implemented so far
- compression support
- benchmark/demo code
