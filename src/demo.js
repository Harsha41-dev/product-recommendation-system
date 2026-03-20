import { performance } from "node:perf_hooks";
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

  const catalog = createSampleCatalog({
    productCount: productCount,
    dimension: dimension
  });

  const products = catalog.products;
  const categoryCenters = catalog.categoryCenters;
  const queries = createQueryProfiles({
    count: queryCount,
    categoryCenters: categoryCenters
  });

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
        clusterCount: clusterCount,
        probeClusters: Math.min(4, clusterCount)
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
        subspaceCount: subspaceCount,
        codebookSize: codebookSize,
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
    `Catalog size: ${productCount} products | Query count: ${queryCount} | Dimension: ${dimension}`
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
