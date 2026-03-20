import { fitKMeans, nearestCentroidIndices } from "../algorithms/kmeans.js";
import { sliceVector, squaredDistance } from "../utils/vector.js";

export function createProductQuantizer(inputOptions = {}) {
  const options = {
    subspaceCount: 4,
    codebookSize: 16,
    maxIterations: 20,
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

  if (inputOptions.seed != null) {
    options.seed = inputOptions.seed;
  }

  let dimension = 0;
  let subspaceSize = 0;
  let codebooks = [];

  function fit(vectors) {
    dimension = vectors[0].length;

    if (dimension % options.subspaceCount !== 0) {
      throw new Error("Vector dimension must be divisible by subspaceCount.");
    }

    subspaceSize = dimension / options.subspaceCount;
    codebooks = [];

    for (let subspaceIndex = 0; subspaceIndex < options.subspaceCount; subspaceIndex += 1) {
      const start = subspaceIndex * subspaceSize;
      const end = start + subspaceSize;
      const subVectors = [];

      for (let i = 0; i < vectors.length; i += 1) {
        subVectors.push(sliceVector(vectors[i], start, end));
      }

      const model = fitKMeans(subVectors, {
        clusterCount: options.codebookSize,
        maxIterations: options.maxIterations,
        seed: options.seed + subspaceIndex
      });

      codebooks.push(model.centroids);
    }

    return api;
  }

  function encodeVector(vector) {
    const codes = [];

    for (let subspaceIndex = 0; subspaceIndex < codebooks.length; subspaceIndex += 1) {
      const start = subspaceIndex * subspaceSize;
      const end = start + subspaceSize;
      const subVector = sliceVector(vector, start, end);
      const clusterIndex = nearestCentroidIndices(subVector, codebooks[subspaceIndex], 1)[0];
      codes.push(clusterIndex);
    }

    return codes;
  }

  function encode(vectors) {
    const allCodes = [];

    for (let i = 0; i < vectors.length; i += 1) {
      allCodes.push(encodeVector(vectors[i]));
    }

    return allCodes;
  }

  function buildDistanceTable(queryVector) {
    const table = [];

    for (let subspaceIndex = 0; subspaceIndex < codebooks.length; subspaceIndex += 1) {
      const start = subspaceIndex * subspaceSize;
      const end = start + subspaceSize;
      const querySlice = sliceVector(queryVector, start, end);
      const distances = [];

      for (let i = 0; i < codebooks[subspaceIndex].length; i += 1) {
        distances.push(squaredDistance(querySlice, codebooks[subspaceIndex][i]));
      }

      table.push(distances);
    }

    return table;
  }

  function estimateSquaredDistance(table, code) {
    let distance = 0;

    for (let subspaceIndex = 0; subspaceIndex < code.length; subspaceIndex += 1) {
      distance += table[subspaceIndex][code[subspaceIndex]];
    }

    return distance;
  }

  function estimateStorageBytes(vectorCount, vectorDimension) {
    const bytesPerFloat = 4;
    const codebookBytes = options.codebookSize * vectorDimension * bytesPerFloat;
    const bitsPerCode = Math.ceil(Math.log2(options.codebookSize));
    const bytesPerProductCode = Math.ceil((bitsPerCode * options.subspaceCount) / 8);
    const rawBytes = vectorCount * vectorDimension * bytesPerFloat;
    const compressedBytes = codebookBytes + vectorCount * bytesPerProductCode;

    return {
      rawBytes: rawBytes,
      compressedBytes: compressedBytes,
      compressionRatio: Number((rawBytes / compressedBytes).toFixed(2)),
      bytesPerProductCode: bytesPerProductCode
    };
  }

  const api = {
    fit: fit,
    encodeVector: encodeVector,
    encode: encode,
    buildDistanceTable: buildDistanceTable,
    estimateSquaredDistance: estimateSquaredDistance,
    estimateStorageBytes: estimateStorageBytes
  };

  return api;
}
