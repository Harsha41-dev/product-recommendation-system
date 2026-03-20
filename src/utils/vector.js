export function dot(left, right) {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) {
    sum += left[index] * right[index];
  }
  return sum;
}

export function magnitude(vector) {
  return Math.sqrt(dot(vector, vector));
}

export function normalize(vector) {
  const length = magnitude(vector) || 1;
  return vector.map((value) => value / length);
}

export function add(left, right) {
  return left.map((value, index) => value + right[index]);
}

export function scale(vector, factor) {
  return vector.map((value) => value * factor);
}

export function squaredDistance(left, right) {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) {
    const difference = left[index] - right[index];
    sum += difference * difference;
  }
  return sum;
}

export function cosineSimilarity(left, right) {
  return dot(left, right) / ((magnitude(left) || 1) * (magnitude(right) || 1));
}

export function zeros(length) {
  return Array(length).fill(0);
}

export function sliceVector(vector, start, end) {
  return vector.slice(start, end);
}

export function meanOfVectors(vectors) {
  const dimension = vectors[0].length;
  const sums = zeros(dimension);
  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) {
      sums[index] += vector[index];
    }
  }
  return sums.map((value) => value / vectors.length);
}

export function varianceByDimension(vectors, indices) {
  const dimension = vectors[0].length;
  const means = zeros(dimension);

  for (const itemIndex of indices) {
    const vector = vectors[itemIndex];
    for (let dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex += 1) {
      means[dimensionIndex] += vector[dimensionIndex];
    }
  }

  for (let dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex += 1) {
    means[dimensionIndex] /= indices.length;
  }

  const variances = zeros(dimension);
  for (const itemIndex of indices) {
    const vector = vectors[itemIndex];
    for (let dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex += 1) {
      const difference = vector[dimensionIndex] - means[dimensionIndex];
      variances[dimensionIndex] += difference * difference;
    }
  }

  return variances.map((value) => value / indices.length);
}

