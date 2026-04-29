function toFiniteNumber(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function isNumber(value) {
  return toFiniteNumber(value) !== null;
}

export function normalizeFusionWeights(result) {
  return normalizeTriplet(result?.fusion_weights || result?.evidence_weights || result?.decision_weights);
}

export function normalizeTriplet(value) {
  if (value && !Array.isArray(value)) {
    const semantic = toFiniteNumber(value.semantic ?? value.rgb);
    const frequency = toFiniteNumber(value.frequency ?? value.freq);
    const noise = toFiniteNumber(value.noise);
    if ([semantic, frequency, noise].some(isNumber)) {
      return {
        semantic,
        frequency,
        noise,
      };
    }
  }

  if (Array.isArray(value) && value.length >= 3) {
    return {
      semantic: toFiniteNumber(value[0]),
      frequency: toFiniteNumber(value[1]),
      noise: toFiniteNumber(value[2]),
    };
  }
  return {
    semantic: null,
    frequency: null,
    noise: null,
  };
}

export function normalizeBranchScores(result) {
  const branchLike =
    result?.branch_scores ||
    result?.branch_score ||
    result?.branch_evidence ||
    result?.branch_support ||
    null;

  return normalizeTriplet(branchLike);
}

export function normalizeDecisionWeights(result, rootResult = null) {
  return normalizeTriplet(
    result?.decision_weights ||
      rootResult?.decision_weights ||
      result?.branch_contribution ||
      rootResult?.branch_contribution ||
      result?.decision_weight ||
      rootResult?.decision_weight,
  );
}

export function normalizeEvidenceWeights(result, rootResult = null) {
  return normalizeTriplet(
    result?.evidence_weights ||
      rootResult?.evidence_weights ||
      result?.fusion_weights ||
      rootResult?.fusion_weights ||
      result?.branch_scores ||
      rootResult?.branch_scores,
  );
}

export function normalizeModeResult(result, rootResult = null) {
  if (!result) {
    return result;
  }

  const resultWithFallbacks = {
    ...rootResult,
    ...result,
  };

  return {
    ...result,
    fusion_weights: normalizeFusionWeights(resultWithFallbacks),
    decision_weights: normalizeDecisionWeights(result, rootResult),
    evidence_weights: normalizeEvidenceWeights(result, rootResult),
    branch_scores: normalizeBranchScores(resultWithFallbacks),
    artifacts: {
      ...(rootResult?.artifacts || {}),
      ...(result?.artifacts || {}),
    },
    srm_image: result?.srm_image ?? rootResult?.srm_image ?? null,
    spectrum_image: result?.spectrum_image ?? rootResult?.spectrum_image ?? null,
    fusion_evidence_image: result?.fusion_evidence_image ?? rootResult?.fusion_evidence_image ?? null,
  };
}

export function getEvidenceImage(result, rootResult, artifactKey, legacyKey) {
  return (
    result?.artifacts?.[artifactKey] ??
    result?.[legacyKey] ??
    rootResult?.artifacts?.[artifactKey] ??
    rootResult?.[legacyKey] ??
    null
  );
}

export function formatOptionalMetric(value, digits = 4) {
  return isNumber(value) ? value.toFixed(digits) : 'N/A';
}
