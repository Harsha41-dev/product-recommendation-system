import { createExactRecommender } from "./exactRecommender.js";
import { MinHeap } from "../utils/minHeap.js";
import { varianceByDimension } from "../utils/vector.js";

function chooseSplitDimension(vectors, indices) {
  const variances = varianceByDimension(vectors, indices);
  let bestIndex = 0;

  for (let i = 1; i < variances.length; i += 1) {
    if (variances[i] > variances[bestIndex]) {
      bestIndex = i;
    }
  }

  return bestIndex;
}

function buildTree(vectors, indices, leafSize) {
  if (indices.length <= leafSize) {
    return {
      isLeaf: true,
      indices: indices
    };
  }

  const splitDimension = chooseSplitDimension(vectors, indices);
  const sortedIndices = [...indices].sort(function (left, right) {
    return vectors[left][splitDimension] - vectors[right][splitDimension];
  });

  const middle = Math.floor(sortedIndices.length / 2);
  const leftIndices = sortedIndices.slice(0, middle);
  const rightIndices = sortedIndices.slice(middle);

  if (leftIndices.length === 0 || rightIndices.length === 0) {
    return {
      isLeaf: true,
      indices: indices
    };
  }

  const splitValue =
    (vectors[sortedIndices[middle - 1]][splitDimension] +
      vectors[sortedIndices[middle]][splitDimension]) /
    2;

  return {
    isLeaf: false,
    splitDimension: splitDimension,
    splitValue: splitValue,
    left: buildTree(vectors, leftIndices, leafSize),
    right: buildTree(vectors, rightIndices, leafSize)
  };
}

export function createBSPRecommender(products, inputOptions = {}) {
  const options = {
    leafSize: 32,
    maxLeaves: 10
  };

  if (inputOptions.leafSize != null) {
    options.leafSize = inputOptions.leafSize;
  }

  if (inputOptions.maxLeaves != null) {
    options.maxLeaves = inputOptions.maxLeaves;
  }

  const exact = createExactRecommender(products);
  let root = null;

  async function build() {
    const indices = [];
    const vectors = [];

    for (let i = 0; i < products.length; i += 1) {
      indices.push(i);
      vectors.push(products[i].vector);
    }

    root = buildTree(vectors, indices, options.leafSize);
    return { ready: true };
  }

  function search(queryVector, topK, searchOptions = {}) {
    let maxLeaves = options.maxLeaves;

    if (searchOptions.maxLeaves != null) {
      maxLeaves = searchOptions.maxLeaves;
    }

    const queue = new MinHeap();
    const candidateSet = new Set();
    let visitedLeaves = 0;

    queue.push({
      priority: 0,
      node: root
    });

    while (!queue.isEmpty() && visitedLeaves < maxLeaves) {
      const current = queue.pop();
      const node = current.node;

      if (node.isLeaf) {
        for (let i = 0; i < node.indices.length; i += 1) {
          candidateSet.add(node.indices[i]);
        }
        visitedLeaves += 1;
      } else {
        const delta = queryVector[node.splitDimension] - node.splitValue;
        let nearNode = node.left;
        let farNode = node.right;

        if (delta > 0) {
          nearNode = node.right;
          farNode = node.left;
        }

        queue.push({
          priority: current.priority,
          node: nearNode
        });

        queue.push({
          priority: current.priority + Math.abs(delta),
          node: farNode
        });
      }
    }

    const candidateIndices = Array.from(candidateSet);
    const result = exact.search(queryVector, topK, {
      candidateIndices: candidateIndices
    });

    return {
      results: result.results,
      meta: {
        candidateCount: result.meta.candidateCount,
        visitedLeaves: visitedLeaves
      }
    };
  }

  return {
    build: build,
    search: search
  };
}
