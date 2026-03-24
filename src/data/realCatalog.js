import fs from "node:fs/promises";
import path from "node:path";
import { createSeededRandom, shuffle } from "../utils/random.js";
import { normalize } from "../utils/vector.js";

function parseJsonLines(content, filePath) {
  const lines = content.split(/\r?\n/);
  const items = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();

    if (!line) {
      continue;
    }

    try {
      items.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`Could not parse JSONL in ${filePath} at line ${i + 1}.`);
    }
  }

  return items;
}

async function readDataFile(filePath) {
  const fullPath = path.resolve(filePath);
  const content = await fs.readFile(fullPath, "utf8");
  const extension = path.extname(fullPath).toLowerCase();

  if (extension === ".jsonl" || extension === ".ndjson") {
    return parseJsonLines(content, fullPath);
  }

  try {
    return JSON.parse(content);
  } catch (error) {
    throw new Error(`Could not parse JSON in ${fullPath}.`);
  }
}

function getVectorFromRecord(record) {
  if (Array.isArray(record.vector)) {
    return record.vector;
  }

  if (Array.isArray(record.embedding)) {
    return record.embedding;
  }

  if (Array.isArray(record.values)) {
    return record.values;
  }

  return null;
}

function toNumericVector(vector, label, shouldNormalize) {
  if (!Array.isArray(vector) || vector.length === 0) {
    throw new Error(`${label} is missing a valid vector array.`);
  }

  const numericVector = [];

  for (let i = 0; i < vector.length; i += 1) {
    const value = Number(vector[i]);

    if (!Number.isFinite(value)) {
      throw new Error(`${label} has a non-numeric vector value at index ${i}.`);
    }

    numericVector.push(value);
  }

  if (shouldNormalize) {
    return normalize(numericVector);
  }

  return numericVector;
}

function mapProductRecord(record, index, shouldNormalize) {
  const vector = toNumericVector(
    getVectorFromRecord(record),
    `Product at index ${index}`,
    shouldNormalize
  );

  let id = `product-${String(index + 1).padStart(5, "0")}`;
  if (record.id != null) {
    id = String(record.id);
  } else if (record.productId != null) {
    id = String(record.productId);
  }

  let title = id;
  if (record.title != null) {
    title = String(record.title);
  } else if (record.name != null) {
    title = String(record.name);
  }

  let category = "unknown";
  if (record.category != null) {
    category = String(record.category);
  } else if (record.segment != null) {
    category = String(record.segment);
  }

  let price = null;
  if (record.price != null) {
    const numericPrice = Number(record.price);
    if (Number.isFinite(numericPrice)) {
      price = numericPrice;
    }
  }

  return {
    id: id,
    title: title,
    category: category,
    price: price,
    vector: vector
  };
}

function mapQueryRecord(record, index, shouldNormalize) {
  const vector = toNumericVector(
    getVectorFromRecord(record),
    `Query at index ${index}`,
    shouldNormalize
  );

  let id = `query-${String(index + 1).padStart(3, "0")}`;
  if (record.id != null) {
    id = String(record.id);
  } else if (record.queryId != null) {
    id = String(record.queryId);
  }

  let segment = "custom-query";
  if (record.segment != null) {
    segment = String(record.segment);
  } else if (record.category != null) {
    segment = String(record.category);
  } else if (record.name != null) {
    segment = String(record.name);
  }

  return {
    id: id,
    segment: segment,
    vector: vector
  };
}

function splitBundleData(parsedData, label) {
  if (Array.isArray(parsedData)) {
    return {
      items: parsedData,
      queries: null
    };
  }

  if (!parsedData || typeof parsedData !== "object") {
    throw new Error(`${label} must be a JSON array, JSONL file, or an object with products.`);
  }

  if (Array.isArray(parsedData.products)) {
    return {
      items: parsedData.products,
      queries: Array.isArray(parsedData.queries) ? parsedData.queries : null
    };
  }

  throw new Error(`${label} must contain a products array.`);
}

function validateDimensions(items, label) {
  const dimension = items[0].vector.length;

  for (let i = 1; i < items.length; i += 1) {
    if (items[i].vector.length !== dimension) {
      throw new Error(`${label} vectors do not all have the same dimension.`);
    }
  }

  return dimension;
}

function createQueriesFromProducts(products, requestedCount, seed) {
  const random = createSeededRandom(seed);
  const indices = [];

  for (let i = 0; i < products.length; i += 1) {
    indices.push(i);
  }

  const shuffledIndices = shuffle(indices, random);
  const queryCount = Math.min(requestedCount, products.length);
  const queries = [];

  for (let i = 0; i < queryCount; i += 1) {
    const product = products[shuffledIndices[i]];

    queries.push({
      id: `query-${String(i + 1).padStart(3, "0")}`,
      segment: product.category,
      vector: product.vector
    });
  }

  return queries;
}

function limitQueries(queries, requestedCount, seed) {
  if (requestedCount >= queries.length) {
    return queries;
  }

  const random = createSeededRandom(seed);
  const indices = [];

  for (let i = 0; i < queries.length; i += 1) {
    indices.push(i);
  }

  const shuffledIndices = shuffle(indices, random);
  const limitedQueries = [];

  for (let i = 0; i < requestedCount; i += 1) {
    limitedQueries.push(queries[shuffledIndices[i]]);
  }

  return limitedQueries;
}

export async function loadRealCatalog(options = {}) {
  const dataPath = options.dataPath;
  const queriesPath = options.queriesPath;
  const shouldNormalize = options.normalizeVectors !== false;
  let queryCount = 50;
  let seed = 42;

  if (options.queryCount != null) {
    queryCount = options.queryCount;
  }

  if (options.seed != null) {
    seed = options.seed;
  }

  if (!dataPath) {
    throw new Error("loadRealCatalog requires a dataPath.");
  }

  const parsedData = await readDataFile(dataPath);
  const bundledData = splitBundleData(parsedData, "Data file");
  const productRecords = bundledData.items;

  if (!Array.isArray(productRecords) || productRecords.length === 0) {
    throw new Error("Data file does not contain any products.");
  }

  const products = [];
  for (let i = 0; i < productRecords.length; i += 1) {
    products.push(mapProductRecord(productRecords[i], i, shouldNormalize));
  }

  const dimension = validateDimensions(products, "Product");
  let queries = [];
  let querySource = "sampled-products";

  if (queriesPath) {
    const parsedQueries = await readDataFile(queriesPath);
    const queryRecords = Array.isArray(parsedQueries) ? parsedQueries : parsedQueries.queries;

    if (!Array.isArray(queryRecords) || queryRecords.length === 0) {
      throw new Error("Queries file does not contain any queries.");
    }

    for (let i = 0; i < queryRecords.length; i += 1) {
      queries.push(mapQueryRecord(queryRecords[i], i, shouldNormalize));
    }

    querySource = "query-file";
  } else if (bundledData.queries && bundledData.queries.length > 0) {
    for (let i = 0; i < bundledData.queries.length; i += 1) {
      queries.push(mapQueryRecord(bundledData.queries[i], i, shouldNormalize));
    }

    querySource = "data-file";
  } else {
    queries = createQueriesFromProducts(products, queryCount, seed);
  }

  if (queries.length === 0) {
    throw new Error("No queries were loaded or generated.");
  }

  queries = limitQueries(queries, queryCount, seed);

  const queryDimension = validateDimensions(queries, "Query");
  if (queryDimension !== dimension) {
    throw new Error("Product and query vectors must have the same dimension.");
  }

  return {
    products: products,
    queries: queries,
    meta: {
      source: "file",
      dataPath: path.resolve(dataPath),
      queriesPath: queriesPath ? path.resolve(queriesPath) : null,
      querySource: querySource,
      dimension: dimension
    }
  };
}
