import { performance } from "node:perf_hooks";
import path from "node:path";
import { loadRealCatalog } from "./data/realCatalog.js";
import { createQueryProfiles, createSampleCatalog } from "./data/sampleCatalog.js";
import { mean, recallAtK } from "./evaluation/metrics.js";
import { createBSPRecommender } from "./recommenders/bspRecommender.js";
import { createExactRecommender } from "./recommenders/exactRecommender.js";
import { createHnswRecommender } from "./recommenders/hnswRecommender.js";
import { createKMeansRecommender } from "./recommenders/kmeansRecommender.js";
import { createPQRecommender } from "./recommenders/pqRecommender.js";

function getFlagValue(flag, fallback) {
  const index = process.argv.indexOf(flag);

  if (index === -1) {
    return fallback;
  }

  if (index === process.argv.length - 1) {
    return fallback;
  }

  return process.argv[index + 1];
}

function hasFlag(flag) {
  return process.argv.includes(flag);
}

function resolveSubspaceCount(dimension, requestedSubspaceCount) {
  let best = 1;
  let limit = requestedSubspaceCount;

  if (limit > dimension) {
    limit = dimension;
  }

  for (let value = 1; value <= limit; value += 1) {
    if (dimension % value === 0) {
      best = value;
    }
  }

  return best;
}

async function benchmarkRecommender(label, recommender, queries, topK, referenceResults) {
  const buildStart = performance.now();
  let buildInfo = await recommender.build();

  if (!buildInfo) {
    buildInfo = {};
  }

  const buildMs = performance.now() - buildStart;
  const queryLatencies = [];
  const recalls = [];
  let sampleResults = [];
  let lastMeta = {};

  for (let i = 0; i < queries.length; i += 1) {
    const query = queries[i];
    const queryStart = performance.now();
    const searchResult = recommender.search(query.vector, topK);
    const queryMs = performance.now() - queryStart;

    queryLatencies.push(queryMs);
    recalls.push(recallAtK(referenceResults[i], searchResult.results));
    lastMeta = searchResult.meta;

    if (i === 0) {
      sampleResults = searchResult.results;
    }
  }

  let candidates = "-";
  if (lastMeta && lastMeta.candidateCount != null) {
    candidates = lastMeta.candidateCount;
  }

  let notes = "";
  if (buildInfo.note) {
    notes = buildInfo.note;
  } else if (lastMeta && lastMeta.note) {
    notes = lastMeta.note;
  }

  return {
    label: label,
    buildMs: Number(buildMs.toFixed(2)),
    avgQueryMs: Number(mean(queryLatencies).toFixed(3)),
    recallAtK: Number(mean(recalls).toFixed(3)),
    candidates: candidates,
    notes: notes,
    sampleResults: sampleResults,
    buildInfo: buildInfo
  };
}

function printSampleResults(title, results) {
  const rows = [];
  const limit = Math.min(5, results.length);

  for (let i = 0; i < limit; i += 1) {
    rows.push({
      id: results[i].id,
      category: results[i].category,
      score: results[i].score
    });
  }

  console.log(`\n[${title}]`);
  console.table(rows);
}

async function main() {
  const productCount = Number(getFlagValue("--products", 5000));
  const queryCount = Number(getFlagValue("--queries", 50));
  const topK = Number(getFlagValue("--topK", 10));
  const dimension = Number(getFlagValue("--dim", 32));
  const clusterCount = Number(getFlagValue("--clusters", 32));
  const subspaceCount = Number(getFlagValue("--subspaces", 4));
  const codebookSize = Number(getFlagValue("--codebook", 16));
  const seed = Number(getFlagValue("--seed", 42));
  const dataPath = getFlagValue("--data", "");
  const queriesPath = getFlagValue("--queries-file", "");
  const keepScale = hasFlag("--keep-scale");

  let products = [];
  let queries = [];
  let activeDimension = dimension;
  let sourceLabel = "synthetic sample data";
  let querySourceLabel = "generated synthetic queries";

  if (dataPath) {
    const realCatalog = await loadRealCatalog({
      dataPath: dataPath,
      queriesPath: queriesPath,
      queryCount: queryCount,
      seed: seed,
      normalizeVectors: !keepScale
    });

    products = realCatalog.products;
    queries = realCatalog.queries;
    activeDimension = realCatalog.meta.dimension;
    sourceLabel = `file data (${path.basename(realCatalog.meta.dataPath)})`;

    if (realCatalog.meta.querySource === "query-file") {
      querySourceLabel = `queries file (${path.basename(realCatalog.meta.queriesPath)})`;
    } else if (realCatalog.meta.querySource === "data-file") {
      querySourceLabel = "queries from data file";
    } else {
      querySourceLabel = "queries sampled from products";
    }
  } else {
    const catalog = createSampleCatalog({
      productCount: productCount,
      dimension: dimension,
      seed: seed
    });

    products = catalog.products;
    const categoryCenters = catalog.categoryCenters;
    queries = createQueryProfiles({
      count: queryCount,
      seed: seed + 1,
      categoryCenters: categoryCenters
    });
  }

  let effectiveClusterCount = clusterCount;
  if (effectiveClusterCount > products.length) {
    effectiveClusterCount = products.length;
  }

  let effectiveCodebookSize = codebookSize;
  if (effectiveCodebookSize > products.length) {
    effectiveCodebookSize = products.length;
  }

  const effectiveSubspaceCount = resolveSubspaceCount(activeDimension, subspaceCount);

  const exact = createExactRecommender(products);
  await exact.build();

  const referenceResults = [];
  for (let i = 0; i < queries.length; i += 1) {
    const result = exact.search(queries[i].vector, topK);
    referenceResults.push(result.results);
  }

  const benchmarks = [];

  benchmarks.push(
    await benchmarkRecommender(
      "exact",
      createExactRecommender(products),
      queries,
      topK,
      referenceResults
    )
  );

  benchmarks.push(
    await benchmarkRecommender(
      "kmeans",
      createKMeansRecommender(products, {
        clusterCount: effectiveClusterCount,
        probeClusters: Math.min(4, effectiveClusterCount)
      }),
      queries,
      topK,
      referenceResults
    )
  );

  benchmarks.push(
    await benchmarkRecommender(
      "bsp",
      createBSPRecommender(products, {
        leafSize: 32,
        maxLeaves: 10
      }),
      queries,
      topK,
      referenceResults
    )
  );

  benchmarks.push(
    await benchmarkRecommender(
      "pq",
      createPQRecommender(products, {
        subspaceCount: effectiveSubspaceCount,
        codebookSize: effectiveCodebookSize,
        rerankCandidates: Math.max(topK * 6, 50)
      }),
      queries,
      topK,
      referenceResults
    )
  );

  benchmarks.push(
    await benchmarkRecommender(
      "hnsw",
      createHnswRecommender(products, {
        efSearch: 64
      }),
      queries,
      topK,
      referenceResults
    )
  );

  console.log("");
  console.log(
    `Catalog size: ${products.length} products | Query count: ${queries.length} | Dimension: ${activeDimension}`
  );
  console.log(`Source: ${sourceLabel} | Query source: ${querySourceLabel}`);
  console.log(
    `Clusters: ${effectiveClusterCount} | Subspaces: ${effectiveSubspaceCount} | Codebook size: ${effectiveCodebookSize}`
  );

  const summaryRows = [];
  for (let i = 0; i < benchmarks.length; i += 1) {
    summaryRows.push({
      method: benchmarks[i].label,
      buildMs: benchmarks[i].buildMs,
      avgQueryMs: benchmarks[i].avgQueryMs,
      recallAtK: benchmarks[i].recallAtK,
      candidates: benchmarks[i].candidates,
      notes: benchmarks[i].notes
    });
  }
  console.table(summaryRows);

  let pqBenchmark = null;
  for (let i = 0; i < benchmarks.length; i += 1) {
    if (benchmarks[i].label === "pq") {
      pqBenchmark = benchmarks[i];
      break;
    }
  }

  if (pqBenchmark && pqBenchmark.buildInfo && pqBenchmark.buildInfo.storage) {
    console.log("PQ storage estimate:", pqBenchmark.buildInfo.storage);
  }

  console.log("");
  console.log(`Sample recommendations for ${queries[0].id} (${queries[0].segment}):`);

  for (let i = 0; i < benchmarks.length; i += 1) {
    printSampleResults(benchmarks[i].label, benchmarks[i].sampleResults);
  }
}

main().catch(function (error) {
  console.error(error);
  process.exitCode = 1;
});
