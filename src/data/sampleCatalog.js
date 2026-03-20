import { createSeededRandom, randomInt, randomNormal } from "../utils/random.js";
import { add, normalize, scale } from "../utils/vector.js";

const DEFAULT_CATEGORIES = [
  "electronics",
  "fashion",
  "home",
  "beauty",
  "sports",
  "books",
  "grocery",
  "toys"
];

function createCenterVector(dimension, random) {
  const vector = [];
  for (let index = 0; index < dimension; index += 1) {
    vector.push(randomNormal(random));
  }
  return normalize(vector);
}

function createNoisyVector(baseVector, random, noiseScale = 0.22) {
  const noise = baseVector.map(() => randomNormal(random));
  return normalize(add(baseVector, scale(noise, noiseScale)));
}

function productTitle(category, index) {
  return `${category[0].toUpperCase()}${category.slice(1)} Product ${index + 1}`;
}

export function createSampleCatalog(options = {}) {
  let productCount = 5000;
  let dimension = 32;
  let seed = 42;
  let categories = DEFAULT_CATEGORIES;

  if (options.productCount != null) {
    productCount = options.productCount;
  }

  if (options.dimension != null) {
    dimension = options.dimension;
  }

  if (options.seed != null) {
    seed = options.seed;
  }

  if (options.categories != null) {
    categories = options.categories;
  }

  const random = createSeededRandom(seed);
  const categoryCenters = {};

  for (let i = 0; i < categories.length; i += 1) {
    const category = categories[i];
    categoryCenters[category] = createCenterVector(dimension, random);
  }

  const products = [];
  for (let index = 0; index < productCount; index += 1) {
    const category = categories[randomInt(categories.length, random)];
    const vector = createNoisyVector(categoryCenters[category], random);
    const price = Number((5 + random() * 300).toFixed(2));

    products.push({
      id: `product-${String(index + 1).padStart(5, "0")}`,
      title: productTitle(category, index),
      category,
      price,
      vector
    });
  }

  return {
    products,
    categoryCenters
  };
}

export function createQueryProfiles(options = {}) {
  let count = 50;
  let seed = 999;
  const categoryCenters = options.categoryCenters;

  if (options.count != null) {
    count = options.count;
  }

  if (options.seed != null) {
    seed = options.seed;
  }

  const random = createSeededRandom(seed);
  const categories = Object.keys(categoryCenters);
  const queries = [];

  for (let index = 0; index < count; index += 1) {
    const primaryCategory = categories[randomInt(categories.length, random)];
    const secondaryCategory = categories[randomInt(categories.length, random)];
    const blendedCenter = normalize(
      add(
        scale(categoryCenters[primaryCategory], 0.75),
        scale(categoryCenters[secondaryCategory], 0.25)
      )
    );

    queries.push({
      id: `query-${String(index + 1).padStart(3, "0")}`,
      segment: `${primaryCategory}+${secondaryCategory}`,
      vector: createNoisyVector(blendedCenter, random, 0.16)
    });
  }

  return queries;
}
