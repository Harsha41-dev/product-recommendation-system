import { cosineSimilarity } from "../utils/vector.js";

export function createExactRecommender(products) {
  const vectors = products.map(function (product) {
    return product.vector;
  });

  async function build() {
    return { ready: true };
  }

  function rankIndices(queryVector, candidateIndices, topK) {
    const scoredItems = [];

    for (let i = 0; i < candidateIndices.length; i += 1) {
      const index = candidateIndices[i];
      const score = cosineSimilarity(queryVector, vectors[index]);
      scoredItems.push({
        index: index,
        score: score
      });
    }

    scoredItems.sort(function (left, right) {
      return right.score - left.score;
    });

    return scoredItems.slice(0, topK);
  }

  function search(queryVector, topK, options = {}) {
    let candidateIndices = options.candidateIndices;

    if (!candidateIndices) {
      candidateIndices = [];
      for (let i = 0; i < products.length; i += 1) {
        candidateIndices.push(i);
      }
    }

    const rankedItems = rankIndices(queryVector, candidateIndices, topK);
    const results = [];

    for (let i = 0; i < rankedItems.length; i += 1) {
      const item = rankedItems[i];
      results.push({
        ...products[item.index],
        score: Number(item.score.toFixed(4))
      });
    }

    return {
      results: results,
      meta: {
        candidateCount: candidateIndices.length
      }
    };
  }

  return {
    build: build,
    search: search
  };
}
