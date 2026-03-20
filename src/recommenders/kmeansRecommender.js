import { fitKMeans, nearestCentroidIndices } from "../algorithms/kmeans.js";
import { createExactRecommender } from "./exactRecommender.js";

export function createKMeansRecommender(products, inputOptions = {}) {
  const options = {
    clusterCount: 32,
    probeClusters: 3,
    maxIterations: 25,
    seed: 42
  };

  if (inputOptions.clusterCount != null) {
    options.clusterCount = inputOptions.clusterCount;
  }

  if (inputOptions.probeClusters != null) {
    options.probeClusters = inputOptions.probeClusters;
  }

  if (inputOptions.maxIterations != null) {
    options.maxIterations = inputOptions.maxIterations;
  }

  if (inputOptions.seed != null) {
    options.seed = inputOptions.seed;
  }

  const exact = createExactRecommender(products);
  let centroids = [];
  let clusters = [];
  let iterations = 0;

  async function build() {
    const vectors = products.map(function (product) {
      return product.vector;
    });

    const model = fitKMeans(vectors, {
      clusterCount: options.clusterCount,
      maxIterations: options.maxIterations,
      seed: options.seed
    });

    centroids = model.centroids;
    clusters = model.clusters;
    iterations = model.iterations;

    return {
      ready: true,
      iterations: model.iterations
    };
  }

  function search(queryVector, topK, searchOptions = {}) {
    let probeClusters = options.probeClusters;

    if (searchOptions.probeClusters != null) {
      probeClusters = searchOptions.probeClusters;
    }

    probeClusters = Math.min(probeClusters, centroids.length);

    const clusterIds = nearestCentroidIndices(queryVector, centroids, probeClusters);
    const candidateIndices = [];

    for (let i = 0; i < clusterIds.length; i += 1) {
      const clusterId = clusterIds[i];
      const clusterItems = clusters[clusterId];

      for (let j = 0; j < clusterItems.length; j += 1) {
        candidateIndices.push(clusterItems[j]);
      }
    }

    const result = exact.search(queryVector, topK, {
      candidateIndices: candidateIndices
    });

    return {
      results: result.results,
      meta: {
        candidateCount: result.meta.candidateCount,
        clusterIds: clusterIds,
        iterations: iterations
      }
    };
  }

  return {
    build: build,
    search: search
  };
}
