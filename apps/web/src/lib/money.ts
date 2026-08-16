import type { MoneyByHour, MoneyPoint } from '../api/types';

/**
 * Convert a backend `money_by_hour` array into chart points.
 *
 * The contract is explicit: the array has length `horizon_hours + 1`, index 0
 * is the money on hand BEFORE hour 1 runs, and index h is the money at the end
 * of hour h. So the array index *is* the hour — plotting index 0 as hour 1
 * would shift every line by one hour and misreport when the first sale lands.
 *
 * `null` means "this session has no such line" and stays `null`: it must render
 * as an absent series, never as zeros, because a flat zero line reads as a real
 * and catastrophic run.
 */
export function hourlyToPoints(values: MoneyByHour | null | undefined): MoneyPoint[] | null {
  if (values === null || values === undefined) return null;
  const points: MoneyPoint[] = [];
  values.forEach((value, hour) => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      points.push({ hour, value });
    }
  });
  return points.length > 0 ? points : null;
}
