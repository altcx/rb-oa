import { describe, expect, it } from 'vitest';
import {
  LATENCY_BUDGET_MS,
  formatCount,
  formatEstimate,
  formatFieldValue,
  formatHour,
  formatMoney,
  formatMoneyAxis,
  formatMoneyDelta,
  formatMs,
  formatMsAsSeconds,
  formatPercent,
  formatSeconds,
  latencyBand,
  latencyColorClass,
  summarizeJson,
} from '../lib/format';

describe('money formatting', () => {
  it('always shows two decimals so near-equal scores stay distinguishable', () => {
    expect(formatMoney(12480.75)).toBe('$12,480.75');
    expect(formatMoney(12480.7)).toBe('$12,480.70');
    expect(formatMoney(12480)).toBe('$12,480.00');
  });

  it('never abbreviates a magnitude', () => {
    expect(formatMoney(1_250_000)).toBe('$1,250,000.00');
    expect(formatMoney(1_250_000)).not.toContain('M');
  });

  it('keeps values that differ by a cent visibly different', () => {
    expect(formatMoney(11930.2)).not.toBe(formatMoney(11930.21));
  });

  it('handles zero and negatives', () => {
    expect(formatMoney(0)).toBe('$0.00');
    expect(formatMoney(-42.5)).toBe('-$42.50');
  });

  it('returns an em dash for non-finite input rather than NaN', () => {
    expect(formatMoney(Number.NaN)).toBe('—');
    expect(formatMoney(Number.POSITIVE_INFINITY)).toBe('—');
  });

  it('rounds only on axis ticks, and only to whole dollars', () => {
    expect(formatMoneyAxis(12480.75)).toBe('$12,481');
    expect(formatMoneyAxis(0)).toBe('$0');
  });
});

describe('deltas', () => {
  it('always carries an explicit sign', () => {
    expect(formatMoneyDelta(578.2)).toBe('+$578.20');
    expect(formatMoneyDelta(-578.2)).toBe('−$578.20');
    expect(formatMoneyDelta(0)).toBe('±$0.00');
  });
});

describe('counts, times and percents', () => {
  it('groups integers', () => {
    expect(formatCount(1240)).toBe('1,240');
    expect(formatCount(0)).toBe('0');
  });

  it('formats milliseconds and seconds consistently', () => {
    expect(formatMs(3420)).toBe('3,420 ms');
    expect(formatMsAsSeconds(3420)).toBe('3.42 s');
    expect(formatMsAsSeconds(4000)).toBe('4.00 s');
    expect(formatSeconds(24.55)).toBe('24.6 s');
    expect(formatEstimate(75)).toBe('~75 s');
  });

  it('formats percents to one decimal', () => {
    expect(formatPercent(0.0783)).toBe('7.8%');
    expect(formatPercent(1)).toBe('100.0%');
  });

  it('labels hours', () => {
    expect(formatHour(4)).toBe('h4');
    expect(formatHour(12)).toBe('h12');
  });
});

describe('latency budget', () => {
  it('bands against the 4 second budget', () => {
    expect(LATENCY_BUDGET_MS).toBe(4000);
    expect(latencyBand(1800)).toBe('good');
    expect(latencyBand(2500)).toBe('good');
    expect(latencyBand(2501)).toBe('warn');
    expect(latencyBand(4000)).toBe('warn');
    expect(latencyBand(4001)).toBe('over');
  });

  it('maps bands to distinct colour classes', () => {
    const classes = new Set([
      latencyColorClass(1000),
      latencyColorClass(3000),
      latencyColorClass(9000),
    ]);
    expect(classes.size).toBe(3);
  });
});

describe('field values', () => {
  it('does not normalise numbers that a reader must tell apart', () => {
    expect(formatFieldValue(12.5)).toBe('12.5');
    expect(formatFieldValue(1.25)).toBe('1.25');
    expect(formatFieldValue(125)).toBe('125');
    expect(formatFieldValue(-0.42)).toBe('-0.42');
  });

  it('renders the other JSON scalars', () => {
    expect(formatFieldValue('press')).toBe('press');
    expect(formatFieldValue(true)).toBe('true');
    expect(formatFieldValue(null)).toBe('null');
    expect(formatFieldValue([1, 2])).toBe('[1,2]');
  });
});

describe('summarizeJson', () => {
  it('summarises objects by key and arrays by length', () => {
    expect(summarizeJson({ a: 1, b: 2 })).toBe('{ a, b }');
    expect(summarizeJson([1, 2, 3])).toBe('3 items');
    expect(summarizeJson([1])).toBe('1 item');
  });

  it('truncates long summaries', () => {
    expect(summarizeJson('x'.repeat(200)).length).toBeLessThanOrEqual(88);
  });
});
