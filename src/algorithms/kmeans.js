import { createSeededRandom, randomInt } from "../utils/random.js";
import { meanOfVectors, squaredDistance } from "../utils/vector.js";

function chooseWeightedIndex(weights, random) {
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  if (totalWeight === 0) {
    return randomInt(weights.length, random);
  }

  let threshold = random() * totalWeight;
  for (let index = 0; index < weights.length; index += 1) {
    threshold -= weights[index];
    if (threshold <= 0) {
      return index;
    }
  }

  return weights.length - 1;
}

function initializeCentroids(vectors, clusterCount, random) {
  const centroids = [];
  const firstIndex = randomInt(vectors.length, random);
  centroids.push([...vectors[firstIndex]]);

  while (centroids.length < clusterCount) {
    const distances = vectors.map((vector) => {
      let minDistance = Number.POSITIVE_INFINITY;
      for (const centroid of centroids) {
        minDistance = Math.min(minDistance, squaredDistance(vector, centroid));
      }
      return minDistance;
    });

    const index = chooseWeightedIndex(distances, random);
    centroids.push([...vectors[index]]);
  }

  return centroids;
}

function assignClusters(vectors, centroids) {
  const assignments = [];
  const clusters = Array.from({ length: centroids.length }, () => []);

  for (let vectorIndex = 0; vectorIndex < vectors.length; vectorIndex += 1) {
    const vector = vectors[vectorIndex];
    let bestCluster = 0;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (let centroidIndex = 0; centroidIndex < centroids.length; centroidIndex += 1) {
      const distance = squaredDistance(vector, centroids[centroidIndex]);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestCluster = centroidIndex;
      }
    }

    assignments.push(bestCluster);
    clusters[bestCluster].push(vectorIndex);
  }

  return { assignments, clusters };
}

export function nearestCentroidIndices(queryVector, centroids, count = 1) {
  return centroids
    .map((centroid, index) => ({
      index,
      distance: squaredDistance(queryVector, centroid)
    }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, count)
    .map((item) => item.index);
}

export function fitKMeans(vectors, options = {}) {
  const clusterCount = options.clusterCount;
  let maxIterations = 25;
  let tolerance = 1e-5;
  let seed = 42;

  if (options.maxIterations != null) {
    maxIterations = options.maxIterations;
  }

  if (options.tolerance != null) {
    tolerance = options.tolerance;
  }

  if (options.seed != null) {
    seed = options.seed;
  }

  if (!Array.isArray(vectors) || vectors.length === 0) {
    throw new Error("fitKMeans requires at least one vector.");
  }

  if (clusterCount < 1 || clusterCount > vectors.length) {
    throw new Error("clusterCount must be between 1 and the number of vectors.");
  }

  const random = createSeededRandom(seed);
  let centroids = initializeCentroids(vectors, clusterCount, random);
  let previousAssignments = [];

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const assignmentResult = assignClusters(vectors, centroids);
    const assignments = assignmentResult.assignments;
    const clusters = assignmentResult.clusters;
    const nextCentroids = centroids.map((centroid, index) => {
      if (clusters[index].length === 0) {
        return [...vectors[randomInt(vectors.length, random)]];
      }
      return meanOfVectors(clusters[index].map((itemIndex) => vectors[itemIndex]));
    });

    let movement = 0;
    for (let centroidIndex = 0; centroidIndex < centroids.length; centroidIndex += 1) {
      movement += squaredDistance(centroids[centroidIndex], nextCentroids[centroidIndex]);
    }

    const sameAssignments =
      previousAssignments.length === assignments.length &&
      assignments.every((value, index) => value === previousAssignments[index]);

    centroids = nextCentroids;
    previousAssignments = assignments;

    if (sameAssignments || movement <= tolerance) {
      return {
        centroids,
        assignments,
        clusters,
        iterations: iteration + 1
      };
    }
  }

  const finalAssignmentResult = assignClusters(vectors, centroids);
  return {
    centroids,
    assignments: finalAssignmentResult.assignments,
    clusters: finalAssignmentResult.clusters,
    iterations: maxIterations
  };
}
