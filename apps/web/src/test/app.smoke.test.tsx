import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Dashboard } from '../components/Dashboard';
import { Hud } from '../components/Hud';
import { useStore } from '../state/store';
import { initialAppState } from '../state/reducer';
import { resetMockState } from '../mock/mockApi';

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
  resetMockState();
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

    // Before calibration the observed and best lines do not exist. They must be
    // named as absent in words — never drawn as a flat zero run.
    expect(screen.getByText(/not recorded yet/)).toBeInTheDocument();
    expect(screen.getByText(/optimizer has not run/)).toBeInTheDocument();
    expect(screen.queryByText('$0.00')).not.toBeInTheDocument();

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

    // only the terminal optimizer event is final
    expect(s.optimizer?.final).toBe(true);
    expect(screen.getByText('FINAL')).toBeInTheDocument();

    // calibration landed, so the observed line exists now
    expect(s.calibration?.matched).toBe(true);
    expect(screen.queryByText(/not recorded yet/)).not.toBeInTheDocument();

    // the provenance alarm is attached to the assistant message that made the claim
    const alarm = screen.getByTestId('provenance-warning');
    expect(alarm).toBeInTheDocument();
    expect(alarm).toHaveTextContent('$186.00');
    expect(screen.getByText('Unverified numbers')).toBeInTheDocument();

    // the extraction tally showed without opening the inspector
    expect(screen.getByTestId('extraction-tally')).toHaveTextContent('37 of 40 fields agreed');
  });

  it('re-extracts the selected capture and reports elapsed time and delta use', async () => {
    render(<Dashboard />);
    await replay(500);

    // fireEvent, not userEvent: user-event's own delays deadlock against the
    // fake timers this suite uses to drive the replay.
    const click = (name: RegExp) =>
      act(() => {
        fireEvent.click(screen.getByRole('button', { name }));
      });

    // First pass is a full extraction.
    await click(/re-extract selected/i);
    await replay(2_000);
    const firstResult = screen.getByTestId('extract-result');
    expect(firstResult).toHaveTextContent('3,420 ms');
    expect(screen.queryByText(/DELTA RE-READ/)).not.toBeInTheDocument();

    // Second pass over the same capture is a fast delta re-read, and says so.
    await click(/re-extract selected/i);
    await replay(2_000);
    const second = screen.getByTestId('extract-result');
    expect(second).toHaveTextContent('1,180 ms');
    expect(screen.getByText(/DELTA RE-READ/)).toBeInTheDocument();
    expect(second).toHaveTextContent('37 auto-confirmed');
    expect(second).toHaveTextContent('3 disputed');
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
