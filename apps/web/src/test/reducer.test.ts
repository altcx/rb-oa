import { describe, expect, it } from 'vitest';
import type { Capture, ServerEvent, Verdict } from '../api/types';
import {
  appReducer,
  applyServerEvent,
  initialAppState,
  openDisputes,
  reviewCounts,
  settledVerdicts,
  type AppState,
} from '../state/reducer';
import { mockState } from '../mock/fixtures';

const capture: Capture = {
  id: 'cap_777',
  created_at: '2026-08-16T10:11:18Z',
  monitor: 1,
  region: null,
  thumb_url: 'data:image/svg+xml;utf8,<svg/>',
  puzzle_type: 'factory_line',
  latency_ms: 3420,
};

const feed = (state: AppState, ...events: ServerEvent[]): AppState =>
  events.reduce(applyServerEvent, state);

describe('socket reducer — one branch per contract event type', () => {
  it('capture: prepends newest-first and selects it', () => {
    const older: Capture = { ...capture, id: 'cap_1' };
    const s1 = applyServerEvent(initialAppState, { type: 'capture', capture: older });
    const s2 = applyServerEvent(s1, { type: 'capture', capture });
    expect(s2.captures.map((c) => c.id)).toEqual(['cap_777', 'cap_1']);
    expect(s2.selectedCaptureId).toBe('cap_777');
  });

  it('capture: replaces a duplicate id instead of double-listing it', () => {
    const s1 = applyServerEvent(initialAppState, { type: 'capture', capture });
    const s2 = applyServerEvent(s1, {
      type: 'capture',
      capture: { ...capture, latency_ms: 999 },
    });
    expect(s2.captures).toHaveLength(1);
    expect(s2.captures[0]?.latency_ms).toBe(999);
  });

  it('extraction_progress: keeps the latest stage and the full log', () => {
    const s = feed(
      initialAppState,
      { type: 'extraction_progress', stage: 'decode', ms: 210 },
      { type: 'extraction_progress', stage: 'reconcile', ms: 3420 },
    );
    expect(s.extraction).toEqual({ stage: 'reconcile', ms: 3420 });
    expect(s.extractionLog).toHaveLength(2);
  });

  it('extraction_progress: carries the field tally when the terminal stage has it', () => {
    const s = applyServerEvent(initialAppState, {
      type: 'extraction_progress',
      stage: 'speculative_done',
      ms: 3420,
      disputed: 19,
      auto_confirmed: 86,
    });
    expect(s.extraction).toEqual({
      stage: 'speculative_done',
      ms: 3420,
      disputed: 19,
      auto_confirmed: 86,
    });
    // 86 of 105 fields agreed, without opening the inspector.
    expect(s.extraction!.auto_confirmed! + s.extraction!.disputed!).toBe(105);
  });

  it('state: installs the payload and clears prior local confirmations', () => {
    const seeded: AppState = { ...initialAppState, accepted: { 'a.b': 1 } };
    const s = applyServerEvent(seeded, {
      type: 'state',
      state: mockState.state,
      verdicts: mockState.verdicts,
      unresolved: mockState.unresolved,
    });
    expect(s.statePayload?.verdicts).toHaveLength(40);
    expect(s.accepted).toEqual({});
    expect(s.statePayload?.unresolved).toEqual(mockState.unresolved);
  });

  it('agent_token: coalesces consecutive tokens into one bubble', () => {
    const s = feed(
      initialAppState,
      { type: 'agent_token', text: 'Board ' },
      { type: 'agent_token', text: 'read. ' },
      { type: 'agent_token', text: 'Lathes bind.' },
    );
    expect(s.chat).toHaveLength(1);
    const first = s.chat[0];
    expect(first?.kind).toBe('agent');
    if (first?.kind === 'agent') {
      expect(first.text).toBe('Board read. Lathes bind.');
      expect(first.complete).toBe(false);
    }
  });

  it('agent_tool_call: closes the open bubble and adds a pending card', () => {
    const s = feed(
      initialAppState,
      { type: 'agent_token', text: 'thinking' },
      { type: 'agent_tool_call', id: 't1', name: 'read_rules', input: { flags: ['a'] } },
    );
    expect(s.chat).toHaveLength(2);
    expect(s.chat[0]?.kind === 'agent' && s.chat[0].complete).toBe(true);
    const tool = s.chat[1];
    expect(tool?.kind).toBe('tool');
    if (tool?.kind === 'tool') {
      expect(tool.output).toBeNull();
      expect(tool.ms).toBeNull();
    }
  });

  it('agent_tool_result: fills the matching card in place', () => {
    const s = feed(
      initialAppState,
      { type: 'agent_tool_call', id: 't1', name: 'read_rules', input: { flags: ['a'] } },
      { type: 'agent_tool_result', id: 't1', name: 'read_rules', output: { ok: true }, ms: 684 },
    );
    expect(s.chat).toHaveLength(1);
    const tool = s.chat[0];
    if (tool?.kind === 'tool') {
      expect(tool.output).toEqual({ ok: true });
      expect(tool.ms).toBe(684);
    } else {
      throw new Error('expected a tool item');
    }
  });

  it('agent_tool_result: surfaces an orphan result rather than dropping it', () => {
    const s = applyServerEvent(initialAppState, {
      type: 'agent_tool_result',
      id: 'ghost',
      name: 'simulate_config',
      output: { final_cash: 1 },
      ms: 12,
    });
    expect(s.chat).toHaveLength(1);
    expect(s.chat[0]?.kind).toBe('tool');
  });

  it('agent_done: completes the bubble and appends the closing notice', () => {
    const s = feed(
      initialAppState,
      { type: 'agent_token', text: 'plan' },
      { type: 'agent_done', message: 'Plan ready.' },
    );
    expect(s.chat[0]?.kind === 'agent' && s.chat[0].complete).toBe(true);
    expect(s.chat[1]).toMatchObject({ kind: 'notice', text: 'Plan ready.' });
  });

  it('optimizer_progress: keeps the latest and the best/bound history', () => {
    const base = {
      type: 'optimizer_progress' as const,
      job_id: 'opt_1',
      final: false,
      money_by_hour: [4820.5, 5900.25],
      actions: [],
      top_warning: null,
    };
    const s = feed(
      initialAppState,
      { ...base, best_value: 100, bound: 140, iterations: 10, elapsed_s: 1 },
      { ...base, best_value: 120, bound: 140, iterations: 25, elapsed_s: 2.5 },
    );
    expect(s.optimizer?.best_value).toBe(120);
    expect(s.optimizer?.final).toBe(false);
    expect(s.optimizerHistory).toHaveLength(2);
    expect(s.optimizerHistory[0]?.best_value).toBe(100);
  });

  it('optimizer_progress: a final event marks the answer, and null curves survive', () => {
    const s = feed(
      initialAppState,
      {
        type: 'optimizer_progress',
        job_id: 'opt_1',
        final: false,
        best_value: 100,
        bound: 140,
        iterations: 10,
        elapsed_s: 1,
        money_by_hour: null,
        actions: [],
        top_warning: null,
      },
      {
        type: 'optimizer_progress',
        job_id: 'opt_1',
        final: true,
        best_value: 138,
        bound: 140,
        iterations: 90,
        elapsed_s: 9,
        money_by_hour: [4820.5, 6000],
        actions: [],
        top_warning: null,
      },
    );
    expect(s.optimizer?.final).toBe(true);
    expect(s.optimizer?.best_value).toBe(138);
    expect(s.optimizerHistory).toHaveLength(2);
  });

  it('calibration: stores the result that unlocks the observed line', () => {
    const s = applyServerEvent(initialAppState, {
      type: 'calibration',
      result: { matched: true, resolved_flags: { 'power.brownout': 'scales' } },
    });
    expect(s.calibration?.matched).toBe(true);
  });

  it('provenance_warning: attaches to the offending assistant message', () => {
    const s = feed(
      initialAppState,
      { type: 'agent_token', text: 'It pays $186.00 an hour.' },
      { type: 'agent_done', message: 'done' },
      {
        type: 'provenance_warning',
        numbers: ['$186.00'],
        text: 'these numbers do not appear in any tool result from this conversation',
      },
    );
    const agent = s.chat.find((c) => c.kind === 'agent');
    expect(agent?.kind).toBe('agent');
    if (agent?.kind === 'agent') {
      expect(agent.provenance).toHaveLength(1);
      expect(agent.provenance[0]?.numbers).toEqual(['$186.00']);
    }
    // It is attached, not appended as a separate floating item.
    expect(s.chat.some((c) => c.kind === 'provenance')).toBe(false);
  });

  it('provenance_warning: targets the most recent assistant message', () => {
    const s = feed(
      initialAppState,
      { type: 'agent_token', text: 'first answer' },
      { type: 'agent_tool_call', id: 't1', name: 'x', input: null },
      { type: 'agent_token', text: 'second answer with $9.99' },
      { type: 'provenance_warning', numbers: ['$9.99'], text: 'unsourced' },
    );
    const agents = s.chat.filter((c) => c.kind === 'agent');
    expect(agents).toHaveLength(2);
    expect(agents[0]?.kind === 'agent' && agents[0].provenance).toHaveLength(0);
    expect(agents[1]?.kind === 'agent' && agents[1].provenance).toHaveLength(1);
  });

  it('provenance_warning: surfaces standalone when there is no message to attach to', () => {
    const s = applyServerEvent(initialAppState, {
      type: 'provenance_warning',
      numbers: ['42'],
      text: 'unsourced',
    });
    expect(s.chat).toHaveLength(1);
    expect(s.chat[0]?.kind).toBe('provenance');
  });

  it('warning: accumulates dismissable warnings with unique ids', () => {
    const s = feed(
      initialAppState,
      { type: 'warning', text: 'first' },
      { type: 'warning', text: 'second' },
    );
    expect(s.warnings.map((w) => w.text)).toEqual(['first', 'second']);
    expect(new Set(s.warnings.map((w) => w.id)).size).toBe(2);

    const dismissed = appReducer(s, { kind: 'warning/dismiss', id: s.warnings[0]!.id });
    expect(dismissed.warnings.map((w) => w.text)).toEqual(['second']);
  });

  it('covers every event type in the contract', () => {
    const types: Array<ServerEvent['type']> = [
      'capture',
      'extraction_progress',
      'state',
      'agent_token',
      'agent_tool_call',
      'agent_tool_result',
      'agent_done',
      'optimizer_progress',
      'warning',
      'calibration',
      'provenance_warning',
    ];
    expect(types).toHaveLength(11);
  });

  it('ignores an unknown event type instead of throwing', () => {
    const rogue = { type: 'not_a_real_event', payload: 1 } as unknown as ServerEvent;
    expect(() => applyServerEvent(initialAppState, rogue)).not.toThrow();
  });
});

describe('review selectors', () => {
  const seeded = appReducer(initialAppState, { kind: 'hydrate/state', payload: mockState });

  it('counts 40 fields with 3 disputed', () => {
    const c = reviewCounts(seeded);
    expect(c.fields).toBe(40);
    expect(c.disputed).toBe(3);
    expect(c.remaining).toBe(3);
    expect(c.agreed).toBe(37);
    expect(c.majorityRemaining).toBe(2);
  });

  it('orders disputes hardest-first (split before majority), not document order', () => {
    const order = openDisputes(seeded).map((v) => v.status);
    expect(order[0]).toBe('split');
    expect(order.slice(1)).toEqual(['majority', 'majority']);
  });

  it('accept moves a field out of the dispute list and into the agreed set', () => {
    const target = openDisputes(seeded)[1] as Verdict;
    const next = appReducer(seeded, {
      kind: 'accept',
      path: target.path,
      value: target.value,
    });
    expect(openDisputes(next)).toHaveLength(2);
    expect(reviewCounts(next).agreed).toBe(38);
    expect(settledVerdicts(next).some((v) => v.path === target.path)).toBe(true);
  });

  it('accept-all-majority settles every majority but leaves splits alone', () => {
    const next = appReducer(seeded, { kind: 'accept-all-majority' });
    const c = reviewCounts(next);
    expect(c.remaining).toBe(1);
    expect(openDisputes(next)[0]?.status).toBe('split');
    expect(c.majorityRemaining).toBe(0);
  });

  it('accept-all-majority is a no-op once nothing is left to accept', () => {
    const once = appReducer(seeded, { kind: 'accept-all-majority' });
    const twice = appReducer(once, { kind: 'accept-all-majority' });
    expect(twice).toBe(once);
  });
});

describe('local actions', () => {
  it('hydrate/captures selects the newest capture when none is selected', () => {
    const s = appReducer(initialAppState, {
      kind: 'hydrate/captures',
      captures: [capture, { ...capture, id: 'cap_1' }],
    });
    expect(s.selectedCaptureId).toBe('cap_777');
  });

  it('select-capture overrides the selection', () => {
    const s1 = appReducer(initialAppState, { kind: 'hydrate/captures', captures: [capture] });
    const s2 = appReducer(s1, { kind: 'select-capture', captureId: 'cap_other' });
    expect(s2.selectedCaptureId).toBe('cap_other');
  });

  it('chat/send appends a user turn', () => {
    const s = appReducer(initialAppState, { kind: 'chat/send', text: 'why the lathes?' });
    expect(s.chat[0]).toMatchObject({ kind: 'user', text: 'why the lathes?' });
  });
});
