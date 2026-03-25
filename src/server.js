import express from "express";
import path from "node:path";
import { loadRealCatalog } from "./data/realCatalog.js";
import { createSampleCatalog } from "./data/sampleCatalog.js";
import { createExactRecommender } from "./recommenders/exactRecommender.js";
import { createHnswRecommender } from "./recommenders/hnswRecommender.js";
import { normalize } from "./utils/vector.js";

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

function getRequestVector(body) {
  if (Array.isArray(body.vector)) {
    return body.vector;
  }

  if (Array.isArray(body.embedding)) {
    return body.embedding;
  }

  if (Array.isArray(body.values)) {
    return body.values;
  }

  return null;
}

function toQueryVector(rawVector, dimension, shouldNormalize) {
  if (!Array.isArray(rawVector) || rawVector.length === 0) {
    throw new Error("Request body must include a vector array.");
  }

  if (rawVector.length !== dimension) {
    throw new Error(`Query vector dimension must be ${dimension}.`);
  }

  const numericVector = [];

  for (let i = 0; i < rawVector.length; i += 1) {
    const value = Number(rawVector[i]);

    if (!Number.isFinite(value)) {
      throw new Error(`Query vector has a non-numeric value at index ${i}.`);
    }

    numericVector.push(value);
  }

  if (shouldNormalize) {
    return normalize(numericVector);
  }

  return numericVector;
}

function getRequestedTopK(body) {
  let topK = 10;

  if (body.topK != null) {
    topK = Number(body.topK);
  }

  if (!Number.isInteger(topK) || topK < 1) {
    throw new Error("topK must be a positive integer.");
  }

  if (topK > 100) {
    topK = 100;
  }

  return topK;
}

function getRequestedMethod(body) {
  let method = "auto";

  if (body.method != null) {
    method = String(body.method).toLowerCase();
  }

  if (method !== "auto" && method !== "hnsw" && method !== "exact") {
    throw new Error("method must be auto, hnsw, or exact.");
  }

  return method;
}

function buildCategoryFilter(body) {
  if (body.category == null) {
    return null;
  }

  const categoryValues = Array.isArray(body.category) ? body.category : [body.category];
  const allowed = new Set();

  for (let i = 0; i < categoryValues.length; i += 1) {
    allowed.add(String(categoryValues[i]).toLowerCase());
  }

  return allowed;
}

function getNumericFilter(value, label) {
  if (value == null) {
    return null;
  }

  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    throw new Error(`${label} must be numeric.`);
  }

  return numericValue;
}

function findFilteredCandidates(products, body) {
  const categoryFilter = buildCategoryFilter(body);
  const minPrice = getNumericFilter(body.minPrice, "minPrice");
  const maxPrice = getNumericFilter(body.maxPrice, "maxPrice");
  const candidateIndices = [];
  let hasFilters = false;

  if (minPrice != null && maxPrice != null && minPrice > maxPrice) {
    throw new Error("minPrice cannot be greater than maxPrice.");
  }

  if (categoryFilter || minPrice != null || maxPrice != null) {
    hasFilters = true;
  }

  for (let i = 0; i < products.length; i += 1) {
    const product = products[i];
    let matches = true;

    if (categoryFilter) {
      const categoryName = String(product.category).toLowerCase();
      if (!categoryFilter.has(categoryName)) {
        matches = false;
      }
    }

    if (matches && minPrice != null) {
      if (product.price == null || product.price < minPrice) {
        matches = false;
      }
    }

    if (matches && maxPrice != null) {
      if (product.price == null || product.price > maxPrice) {
        matches = false;
      }
    }

    if (matches) {
      candidateIndices.push(i);
    }
  }

  return {
    hasFilters: hasFilters,
    candidateIndices: candidateIndices
  };
}

function formatResults(results, topK) {
  const items = [];
  const limit = Math.min(topK, results.length);

  for (let i = 0; i < limit; i += 1) {
    items.push({
      id: results[i].id,
      title: results[i].title,
      category: results[i].category,
      price: results[i].price,
      score: results[i].score
    });
  }

  return items;
}

async function loadProductsForServer(options) {
  if (options.dataPath) {
    const fileCatalog = await loadRealCatalog({
      dataPath: options.dataPath,
      queryCount: 1,
      seed: options.seed,
      normalizeVectors: options.normalizeVectors
    });

    return {
      products: fileCatalog.products,
      dimension: fileCatalog.meta.dimension,
      sourceLabel: `file data (${path.basename(fileCatalog.meta.dataPath)})`
    };
  }

  const catalog = createSampleCatalog({
    productCount: options.productCount,
    dimension: options.dimension,
    seed: options.seed
  });

  return {
    products: catalog.products,
    dimension: options.dimension,
    sourceLabel: "synthetic sample data"
  };
}

async function startServer() {
  const port = Number(getFlagValue("--port", process.env.PORT || 3000));
  const dataPath = getFlagValue("--data", "");
  const productCount = Number(getFlagValue("--products", 5000));
  const dimension = Number(getFlagValue("--dim", 32));
  const seed = Number(getFlagValue("--seed", 42));
  const keepScale = hasFlag("--keep-scale");

  const catalog = await loadProductsForServer({
    dataPath: dataPath,
    productCount: productCount,
    dimension: dimension,
    seed: seed,
    normalizeVectors: !keepScale
  });

  const products = catalog.products;
  const vectorDimension = catalog.dimension;
  const exact = createExactRecommender(products);
  const hnsw = createHnswRecommender(products, {
    efSearch: 64
  });

  await exact.build();
  const hnswBuildInfo = await hnsw.build();
  const hnswReady = !!(hnswBuildInfo && hnswBuildInfo.ready);

  const app = express();
  app.use(express.json({ limit: "2mb" }));

  app.get("/", function (req, res) {
    res.json({
      name: "product-recommendation-system",
      status: "ok",
      routes: ["/health", "/recommend"],
      source: catalog.sourceLabel,
      productCount: products.length,
      dimension: vectorDimension
    });
  });

  app.get("/health", function (req, res) {
    res.json({
      status: "ok",
      source: catalog.sourceLabel,
      productCount: products.length,
      dimension: vectorDimension,
      hnswReady: hnswReady
    });
  });

  app.post("/recommend", function (req, res) {
    try {
      const body = req.body || {};
      const queryVector = toQueryVector(
        getRequestVector(body),
        vectorDimension,
        !keepScale
      );
      const topK = getRequestedTopK(body);
      const requestedMethod = getRequestedMethod(body);
      const filterInfo = findFilteredCandidates(products, body);
      let result = null;
      let usedMethod = "exact";
      let note = "";

      if (filterInfo.hasFilters) {
        if (filterInfo.candidateIndices.length === 0) {
          return res.json({
            meta: {
              method: "exact",
              source: catalog.sourceLabel,
              productCount: products.length,
              filteredCount: 0,
              topK: topK,
              note: "No products matched the filters."
            },
            results: []
          });
        }

        result = exact.search(queryVector, topK, {
          candidateIndices: filterInfo.candidateIndices
        });
        usedMethod = "exact";

        if (requestedMethod === "hnsw") {
          note = "Used exact search because filters need a filtered candidate set.";
        }
      } else if (requestedMethod === "exact") {
        result = exact.search(queryVector, topK);
        usedMethod = "exact";
      } else {
        const fetchCount = Math.min(Math.max(topK * 5, 20), products.length);
        result = hnsw.search(queryVector, fetchCount);
        usedMethod = hnswReady ? "hnsw" : "exact";

        if (!hnswReady) {
          note = "HNSW not available, exact search was used.";
        }
      }

      const responseResults = formatResults(result.results, topK);

      return res.json({
        meta: {
          method: usedMethod,
          source: catalog.sourceLabel,
          productCount: products.length,
          filteredCount: filterInfo.hasFilters ? filterInfo.candidateIndices.length : products.length,
          topK: topK,
          candidateCount: result.meta.candidateCount,
          note: note || result.meta.note || ""
        },
        results: responseResults
      });
    } catch (error) {
      return res.status(400).json({
        error: error.message
      });
    }
  });

  app.use(function (error, req, res, next) {
    if (error) {
      return res.status(400).json({
        error: "Invalid JSON body."
      });
    }

    return next();
  });

  app.listen(port, function () {
    console.log(`Recommendation API running on port ${port}`);
    console.log(`Loaded ${products.length} products from ${catalog.sourceLabel}`);

    if (hnswReady) {
      console.log("HNSW index is ready");
    } else {
      console.log("HNSW is not available, exact fallback will be used");
    }
  });
}

startServer().catch(function (error) {
  console.error(error);
  process.exitCode = 1;
});
