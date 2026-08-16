import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { Dashboard } from '../components/Dashboard';
import { Hud } from '../components/Hud';
import { useStore } from '../state/store';
import { initialAppState } from '../state/reducer';

/** Drain the mock REST delays and the scripted socket replay. */
async function replay(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  // Mock mode is read at call time from localStorage, so flipping it here is
  // enough to route every request at fixtures instead of the network.
  window.localStorage.setItem('puzzle-copilot.mock', '1');
  vi.useFakeTimers();
  useStore.setState({ ...initialAppState });
});

afterEach(() => {
  vi.useRealTimers();
  window.localStorage.removeItem('puzzle-copilot.mock');
});

describe('mock-mode smoke render', () => {
  it('renders all three dashboard panes, the chart and the leaderboard', async () => {
    render(<Dashboard />);
    await replay(500);

    expect(screen.getByText('Puzzle Copilot')).toBeInTheDocument();
    expect(screen.getByTestId('filmstrip')).toBeInTheDocument();
    expect(screen.getByTestId('state-inspector')).toBeInTheDocument();

    // Chart legend names all three series and the LP bound.
    expect(screen.getByText('Current config')).toBeInTheDocument();
    expect(screen.getByText('Optimizer best')).toBeInTheDocument();
    expect(screen.getByText('Observed run')).toBeInTheDocument();
    expect(screen.getAllByText(/LP bound/).length).toBeGreaterThan(0);

    // REST fixtures hydrated captures, state and the leaderboard.
    expect(screen.getByTestId('review-counter')).toHaveTextContent(
      '40 fields, 3 disputed, 3 left to review',
    );
    expect(screen.getByText('HIGH WATER')).toBeInTheDocument();
    expect(screen.getByText('$12,480.75')).toBeInTheDocument();
    expect(useStore.getState().captures.length).toBeGreaterThan(0);
    expect(screen.queryByText(/Backend unreachable/)).not.toBeInTheDocument();
  });

  it('replays the scripted socket session end to end', async () => {
    render(<Dashboard />);
    await replay(40_000);

    const s = useStore.getState();
    // capture -> extraction progress -> state
    expect(s.captures[0]?.id).toBe('cap_005');
    expect(s.extractionLog.length).toBeGreaterThanOrEqual(5);
    expect(s.statePayload?.verdicts).toHaveLength(40);
    // streaming tokens + exactly two tool-call cards, both resolved
    const tools = s.chat.filter((c) => c.kind === 'tool');
    expect(tools).toHaveLength(2);
    expect(tools.every((t) => t.kind === 'tool' && t.output !== null)).toBe(true);
    expect(s.chat.some((c) => c.kind === 'agent' && c.text.length > 100)).toBe(true);
    // optimizer improved across 8 events and never claimed to beat the bound
    expect(s.optimizerHistory).toHaveLength(8);
    const first = s.optimizerHistory[0]!;
    const last = s.optimizerHistory[7]!;
    expect(last.best_value).toBeGreaterThan(first.best_value);
    expect(last.best_value).toBeLessThan(last.bound);
    expect(s.warnings.length).toBeGreaterThan(0);
  });

  it('renders the HUD at exactly 240x80 with a latency readout', async () => {
    const { container } = render(<Hud />);
    await replay(3_000);

    const root = container.querySelector('.hud-root');
    expect(root).toBeInTheDocument();
    expect(root?.className).toContain('w-[240px]');
    expect(root?.className).toContain('h-[80px]');
    expect(screen.getByRole('button', { name: /capture/i })).toBeInTheDocument();
    expect(screen.getByText(/Ctrl\+Shift\+C/)).toBeInTheDocument();
    expect(screen.getByText(/\/ 4\.00 s budget/)).toBeInTheDocument();
    expect(screen.getByAltText('Last capture')).toBeInTheDocument();
  });
});
