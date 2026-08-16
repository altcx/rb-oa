import '@testing-library/jest-dom/vitest';

// jsdom reports every element as 0x0, so recharts' ResponsiveContainer warns on
// every render. That is a jsdom artefact, not a defect under test — drop it so
// real failures stay visible.
const realWarn = console.warn.bind(console);
console.warn = (...args: unknown[]) => {
  if (typeof args[0] === 'string' && args[0].includes('of chart should be greater than 0')) return;
  realWarn(...args);
};

// jsdom has no ResizeObserver; recharts' ResponsiveContainer wants one.
if (!('ResizeObserver' in globalThis)) {
  class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverStub;
}
