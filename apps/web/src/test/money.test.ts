import { describe, expect, it } from 'vitest';
import { hourlyToPoints } from '../lib/money';
import { mockChart, mockChartAfterCalibration, mockCurrentCurve } from '../mock/fixtures';

describe('hourlyToPoints — array index IS the hour', () => {
  it('maps index 0 to hour 0, not hour 1', () => {
    const points = hourlyToPoints([4820.5, 5900.25, 7010.75]);
    expect(points).toEqual([
      { hour: 0, value: 4820.5 },
      { hour: 1, value: 5900.25 },
      { hour: 2, value: 7010.75 },
    ]);
  });

  it('treats index 0 as starting money, before hour 1 has run', () => {
    const points = hourlyToPoints(mockCurrentCurve)!;
    expect(points[0]?.hour).toBe(0);
    // Starting money, not a first-hour result.
    expect(points[0]?.value).toBe(4820.5);
  });

  it('produces horizon_hours + 1 points', () => {
    const horizon = mockChart.horizon_hours!;
    expect(mockChart.current).toHaveLength(horizon + 1);
    expect(hourlyToPoints(mockChart.current)).toHaveLength(horizon + 1);
  });

  it('keeps the last index at the final hour', () => {
    const points = hourlyToPoints([1, 2, 3, 4])!;
    expect(points[points.length - 1]?.hour).toBe(3);
  });
});

describe('hourlyToPoints — absent lines stay absent', () => {
  it('passes null through rather than coercing to zeros', () => {
    expect(hourlyToPoints(null)).toBeNull();
    expect(hourlyToPoints(undefined)).toBeNull();
  });

  it('never invents a zero-valued point for a missing line', () => {
    const points = hourlyToPoints(null);
    expect(points).not.toEqual([]);
    expect(points).not.toEqual([{ hour: 0, value: 0 }]);
  });

  it('treats an empty array as absent, not as a run worth zero', () => {
    expect(hourlyToPoints([])).toBeNull();
  });

  it('drops non-finite entries without shifting the hours of the rest', () => {
    const points = hourlyToPoints([10, Number.NaN, 30])!;
    expect(points).toEqual([
      { hour: 0, value: 10 },
      { hour: 2, value: 30 },
    ]);
  });

  it('preserves a real zero — money on hand can genuinely be zero', () => {
    const points = hourlyToPoints([0, 100])!;
    expect(points[0]).toEqual({ hour: 0, value: 0 });
  });
});

describe('chart fixtures model the null -> present flip', () => {
  it('has no observed line before calibration', () => {
    expect(mockChart.observed).toBeNull();
    expect(mockChart.best).toBeNull();
    expect(mockChart.current).not.toBeNull();
  });

  it('gains the observed and best lines after calibration', () => {
    expect(mockChartAfterCalibration.observed).not.toBeNull();
    expect(mockChartAfterCalibration.best).not.toBeNull();
  });
});
