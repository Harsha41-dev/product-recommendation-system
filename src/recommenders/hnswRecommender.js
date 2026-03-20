import { createExactRecommender } from "./exactRecommender.js";
import { cosineSimilarity } from "../utils/vector.js";

async function loadHnswModule() {
  try {
    const importedModule = await import("hnswlib-node");

    if (importedModule.HierarchicalNSW) {
      return importedModule;
    }

    if (importedModule.default) {
      return importedModule.default;
    }

    return null;
  } catch (error) {
    return null;
  }
}

export function createHnswRecommender(products, inputOptions = {}) {
  const options = {
    space: "cosine",
    efSearch: 64
  };

  if (inputOptions.space != null) {
    options.space = inputOptions.space;
  }

  if (inputOptions.efSearch != null) {
    options.efSearch = inputOptions.efSearch;
  }

  const exact = createExactRecommender(products);
  let index = null;
  let available = false;

  async function build() {
    const hnswModule = await loadHnswModule();

    if (!hnswModule || !hnswModule.HierarchicalNSW) {
      available = false;
      return {
        ready: false,
        note: "Optional dependency hnswlib-node is not installed."
      };
    }

    const dimension = products[0].vector.length;
    index = new hnswModule.HierarchicalNSW(options.space, dimension);
    index.initIndex(products.length);

    if (typeof index.setEfSearch === "function") {
      index.setEfSearch(options.efSearch);
    }

    for (let i = 0; i < products.length; i += 1) {
      index.addPoint(products[i].vector, i);
    }

    available = true;
    return { ready: true };
  }

  function search(queryVector, topK) {
    if (!available) {
      const fallback = exact.search(queryVector, topK);

      return {
        results: fallback.results,
        meta: {
          candidateCount: fallback.meta.candidateCount,
          fallback: true,
          note: "HNSW unavailable, used exact search."
        }
      };
    }

    const rawResult = index.searchKnn(queryVector, topK);
    let neighborIndices = rawResult.neighbors;

    if (!neighborIndices) {
      neighborIndices = rawResult.labels;
    }

    if (!neighborIndices) {
      neighborIndices = [];
    }

    const results = [];
    for (let i = 0; i < neighborIndices.length; i += 1) {
      const productIndex = neighborIndices[i];
      const score = cosineSimilarity(queryVector, products[productIndex].vector);

      results.push({
        ...products[productIndex],
        score: Number(score.toFixed(4))
      });
    }

    return {
      results: results,
      meta: {
        candidateCount: neighborIndices.length
      }
    };
  }

  return {
    build: build,
    search: search
  };
}
