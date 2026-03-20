import { createProductQuantizer } from "../compression/productQuantizer.js";
import { createExactRecommender } from "./exactRecommender.js";

export function createPQRecommender(products, inputOptions = {}) {
  const options = {
    subspaceCount: 4,
    codebookSize: 16,
    maxIterations: 20,
    rerankCandidates: 40,
    seed: 42
  };

  if (inputOptions.subspaceCount != null) {
    options.subspaceCount = inputOptions.subspaceCount;
  }

  if (inputOptions.codebookSize != null) {
    options.codebookSize = inputOptions.codebookSize;
  }

  if (inputOptions.maxIterations != null) {
    options.maxIterations = inputOptions.maxIterations;
  }

  if (inputOptions.rerankCandidates != null) {
    options.rerankCandidates = inputOptions.rerankCandidates;
  }

  if (inputOptions.seed != null) {
    options.seed = inputOptions.seed;
  }

  const exact = createExactRecommender(products);
  const quantizer = createProductQuantizer(options);
  let codes = [];

  async function build() {
    const vectors = [];

    for (let i = 0; i < products.length; i += 1) {
      vectors.push(products[i].vector);
    }

    quantizer.fit(vectors);
    codes = quantizer.encode(vectors);

    return {
      ready: true,
      storage: quantizer.estimateStorageBytes(products.length, products[0].vector.length)
    };
  }

  function search(queryVector, topK) {
    const distanceTable = quantizer.buildDistanceTable(queryVector);
    let candidateCount = options.rerankCandidates;

    if (topK > candidateCount) {
      candidateCount = topK;
    }

    const candidates = [];

    for (let i = 0; i < codes.length; i += 1) {
      candidates.push({
        index: i,
        distance: quantizer.estimateSquaredDistance(distanceTable, codes[i])
      });
    }

    candidates.sort(function (left, right) {
      return left.distance - right.distance;
    });

    const candidateIndices = [];
    for (let i = 0; i < candidateCount && i < candidates.length; i += 1) {
      candidateIndices.push(candidates[i].index);
    }

    const result = exact.search(queryVector, topK, {
      candidateIndices: candidateIndices
    });

    return {
      results: result.results,
      meta: {
        candidateCount: result.meta.candidateCount,
        storage: quantizer.estimateStorageBytes(products.length, products[0].vector.length)
      }
    };
  }

  return {
    build: build,
    search: search
  };
}
