export function recallAtK(referenceResults, candidateResults) {
  const referenceIds = new Set(referenceResults.map((item) => item.id));
  const overlap = candidateResults.filter((item) => referenceIds.has(item.id)).length;
  return overlap / Math.max(referenceResults.length, 1);
}

export function mean(values) {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

