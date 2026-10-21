import type { Observation } from "../ledger/index.js";
import { nowTimestamp, truncateRecordContent } from "../serialize.js";

export function normalizeSourceEntryIds(
  sourceEntryIds: readonly string[] | undefined,
  allowedSourceEntryIds: readonly string[],
): string[] | undefined {
  if (!sourceEntryIds || sourceEntryIds.length === 0) return undefined;
  const allowedOrder = new Map<string, number>();
  for (let i = 0; i < allowedSourceEntryIds.length; i++)
    allowedOrder.set(allowedSourceEntryIds[i], i);

  const seen = new Set<string>();
  const valid: string[] = [];
  for (const id of sourceEntryIds) {
    if (!allowedOrder.has(id) || seen.has(id)) continue;
    seen.add(id);
    valid.push(id);
  }
  if (valid.length === 0) return undefined;
  return valid.sort((a, b) => (allowedOrder.get(a) ?? 0) - (allowedOrder.get(b) ?? 0));
}

export function deriveObservationTimestamp(
  sourceEntryIds: readonly string[],
  sourceEntryTimestamps: Record<string, string> | undefined,
): string {
  let latest: string | undefined;
  if (sourceEntryTimestamps) {
    for (const id of sourceEntryIds) {
      const timestamp = sourceEntryTimestamps[id];
      if (!timestamp || timestamp.startsWith("?")) continue;
      if (!latest || timestamp > latest) latest = timestamp;
    }
  }
  return latest ?? nowTimestamp();
}

export function normalizeSupportingObservationIds(
  supportingObservationIds: readonly string[] | undefined,
  allowedObservationIds: readonly string[],
): string[] | undefined {
  if (!supportingObservationIds || supportingObservationIds.length === 0) return undefined;
  const allowedOrder = new Map<string, number>();
  for (let i = 0; i < allowedObservationIds.length; i++) {
    if (!allowedOrder.has(allowedObservationIds[i])) allowedOrder.set(allowedObservationIds[i], i);
  }

  const seen = new Set<string>();
  for (const id of supportingObservationIds) {
    if (!allowedOrder.has(id)) return undefined;
    seen.add(id);
  }
  if (seen.size === 0) return undefined;
  return Array.from(seen).sort((a, b) => (allowedOrder.get(a) ?? 0) - (allowedOrder.get(b) ?? 0));
}

export function normalizeReflectionContent(content: string): string | undefined {
  const normalized = truncateRecordContent(content.trim());
  if (!normalized || /\r|\n/.test(normalized)) return undefined;
  return normalized;
}

export function normalizeDropObservationIds(
  ids: readonly string[] | undefined,
  observations: readonly Observation[],
): string[] | undefined {
  if (!ids || ids.length === 0) return undefined;
  const allowed = new Map(observations.map((observation) => [observation.id, observation]));
  const result: string[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    if (!allowed.has(id) || seen.has(id)) continue;
    seen.add(id);
    result.push(id);
  }
  return result.length > 0 ? result : undefined;
}
