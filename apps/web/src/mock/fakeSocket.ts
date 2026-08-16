import type { Capture, ClientEvent, ServerEvent } from '../api/types';
import { onMockCapture } from './bus';
import {
  MOCK_LP_BOUND,
  mockCaptureImage,
  mockIncomingCapture,
  mockOptimizerCurve,
  mockState,
} from './fixtures';

export type SocketStatus = 'connecting' | 'open' | 'closed' | 'mock';

export interface SocketLike {
  send(event: ClientEvent): void;
  close(): void;
}

interface ScriptStep {
  /** Delay after the previous step, in ms. */
  after: number;
  event: ServerEvent;
}

const AGENT_OPENING =
  'Board read. The lathe line is the binding constraint — machines[2] and machines[3] together ' +
  'cap gizmo output at roughly 222/h while the assembler can absorb 76/h of finished goods, so ' +
  'gizmos are throttled upstream, not at the assembler. Checking the contract penalties before ' +
  'I recommend the pivot. ';

const AGENT_MIDDLE =
  'Penalty exposure on Halvern is the one number I do not trust yet — the jury split three ways ' +
  'between $1.25, $12.50 and $125.00 per late unit, and at 35 units/h those differ by more than ' +
  'the entire optimizer gap. Confirm that field before you commit. ';

const AGENT_CLOSING =
  'Recommendation: hold widgets through h3 to bank setup cost, pivot the lathes to gizmo at h4, ' +
  'and keep the fifth machine off until the brownout rule is pinned down. Projected $13,140.00 ' +
  'against an LP bound of $14,260.00.';

/** Split text into token-sized chunks the way a streaming API would. */
function tokenize(text: string): string[] {
  return text.match(/\S+\s*/g) ?? [text];
}

function tokenSteps(text: string, after = 22): ScriptStep[] {
  return tokenize(text).map((t) => ({ after, event: { type: 'agent_token', text: t } as const }));
}

function optimizerSteps(): ScriptStep[] {
  const steps: ScriptStep[] = [];
  const bests = [11_902.55, 12_480.75, 12_744.1, 13_002.4, 13_118.9, 13_140.25, 13_140.25, 13_204.6];
  for (let i = 0; i < bests.length; i++) {
    const best = bests[i] as number;
    steps.push({
      after: 620,
      event: {
        type: 'optimizer_progress',
        job_id: 'opt_mock_1',
        best_value: best,
        bound: MOCK_LP_BOUND,
        iterations: 1_240 + i * 3_115,
        elapsed_s: Math.round((2.4 + i * 3.1) * 10) / 10,
        money_by_hour: mockOptimizerCurve(i),
        actions:
          i < 2
            ? [
                {
                  target: 'machines[2]',
                  setting: 'product',
                  value: 'gizmo',
                  reason: 'lathe margin on gizmo exceeds widget by $3.10/unit at current prices',
                },
              ]
            : [
                {
                  target: 'machines[2]',
                  setting: 'product',
                  value: 'gizmo',
                  reason: 'lathe margin on gizmo exceeds widget by $3.10/unit at current prices',
                },
                {
                  target: 'machines[3]',
                  setting: 'product',
                  value: 'gizmo',
                  reason: 'second lathe pays for the h4 setup within 41 minutes',
                },
                {
                  target: 'staff',
                  setting: 'operators',
                  value: 7,
                  reason: 'seventh operator clears the assembler queue before the h8 demand step',
                },
              ],
        top_warning:
          i >= 5
            ? 'Plan assumes no gizmo spoilage — flag inventory.spoilage_applies_to_gizmo is unresolved.'
            : null,
      },
    });
  }
  return steps;
}

const SCRIPT: ScriptStep[] = [
  { after: 400, event: { type: 'capture', capture: mockIncomingCapture } },
  { after: 180, event: { type: 'extraction_progress', stage: 'decode + downscale', ms: 210 } },
  { after: 260, event: { type: 'extraction_progress', stage: 'jury: claude-sonnet-4.5', ms: 1180 } },
  { after: 300, event: { type: 'extraction_progress', stage: 'jury: gpt-5-mini', ms: 1640 } },
  { after: 280, event: { type: 'extraction_progress', stage: 'jury: gemini-2.5-flash', ms: 2310 } },
  { after: 240, event: { type: 'extraction_progress', stage: 'reconcile votes', ms: 3420 } },
  {
    after: 200,
    event: {
      type: 'state',
      state: mockState.state,
      verdicts: mockState.verdicts,
      unresolved: mockState.unresolved,
    },
  },
  ...tokenSteps(AGENT_OPENING),
  {
    after: 240,
    event: {
      type: 'agent_tool_call',
      id: 'tool_1',
      name: 'read_rules',
      input: { session_id: 'sess_9f2c1a', flags: ['contracts.penalty_caps_at_contract_value'] },
    },
  },
  {
    after: 700,
    event: {
      type: 'agent_tool_result',
      id: 'tool_1',
      name: 'read_rules',
      output: {
        'contracts.penalty_caps_at_contract_value': true,
        note: 'cap = units_per_hour * unit_price * hours_late',
        confidence: 0.91,
      },
      ms: 684,
    },
  },
  ...tokenSteps(AGENT_MIDDLE),
  {
    after: 240,
    event: {
      type: 'agent_tool_call',
      id: 'tool_2',
      name: 'simulate_config',
      input: {
        horizon_hours: 12,
        machines: [
          { id: 0, product: 'widget' },
          { id: 1, product: 'widget' },
          { id: 2, product: 'gizmo', from_hour: 4 },
          { id: 3, product: 'gizmo', from_hour: 4 },
          { id: 4, product: 'assembly' },
        ],
        staff: { operators: 7, overtime_from_hour: null },
      },
    },
  },
  {
    after: 1200,
    event: {
      type: 'agent_tool_result',
      id: 'tool_2',
      name: 'simulate_config',
      output: {
        final_cash: 13140.25,
        peak_power_kw: 468,
        brownout_hours: 0,
        late_units: 0,
        by_hour: [980.4, 2104.9, 3288.1, 4402.6, 5711.3, 7044.8, 8390.2, 9701.5],
      },
      ms: 1187,
    },
  },
  ...tokenSteps(AGENT_CLOSING),
  { after: 200, event: { type: 'agent_done', message: 'Plan ready — 3 actions, 1 blocking unknown.' } },
  {
    after: 300,
    event: {
      type: 'warning',
      text: 'contracts[1].penalty_per_late_unit is still split 3 ways; the plan below assumes $12.50.',
    },
  },
  ...optimizerSteps(),
];

/**
 * Replays a realistic session with no server: capture -> extraction progress ->
 * state with 3 disputed fields of 40 -> streaming agent tokens -> two tool-call
 * cards -> optimizer progress improving over 8 events.
 */
export class FakeSocket implements SocketLike {
  private timers: ReturnType<typeof setTimeout>[] = [];
  private closed = false;
  private replyCount = 0;
  private captureSeq = 5;
  private readonly unsubscribe: () => void;

  constructor(
    private readonly onEvent: (event: ServerEvent) => void,
    private readonly onStatus: (status: SocketStatus) => void,
  ) {
    this.onStatus('mock');
    this.unsubscribe = onMockCapture(() => this.onCaptureRequested());
    this.run(SCRIPT);
  }

  /** A capture POST in mock mode pushes a fresh capture + re-extraction. */
  private onCaptureRequested(): void {
    if (this.closed) return;
    this.captureSeq += 1;
    const seq = this.captureSeq;
    const latency = 2400 + ((seq * 617) % 2600);
    const capture: Capture = {
      id: `cap_${String(seq).padStart(3, '0')}`,
      created_at: new Date().toISOString(),
      monitor: 1,
      region: null,
      thumb_url: mockCaptureImage(seq),
      puzzle_type: 'factory_line',
      latency_ms: latency,
    };
    this.run([
      { after: 120, event: { type: 'capture', capture } },
      { after: 160, event: { type: 'extraction_progress', stage: 'decode + downscale', ms: 190 } },
      {
        after: 260,
        event: { type: 'extraction_progress', stage: 'jury: 3 models', ms: Math.round(latency * 0.7) },
      },
      { after: 240, event: { type: 'extraction_progress', stage: 'reconcile votes', ms: latency } },
      {
        after: 180,
        event: {
          type: 'state',
          state: mockState.state,
          verdicts: mockState.verdicts,
          unresolved: mockState.unresolved,
        },
      },
    ]);
  }

  private run(script: ScriptStep[]): void {
    let t = 0;
    for (const step of script) {
      t += step.after;
      this.timers.push(
        setTimeout(() => {
          if (!this.closed) this.onEvent(step.event);
        }, t),
      );
    }
  }

  send(event: ClientEvent): void {
    if (this.closed) return;
    if (event.type === 'cancel') {
      this.onEvent({ type: 'warning', text: `Cancelled job ${event.job_id}.` });
      return;
    }
    this.replyCount += 1;
    const reply =
      this.replyCount === 1
        ? `Looking at "${event.text.slice(0, 60)}" against the confirmed board. ` +
          'The lathes are the lever: every hour they run widgets past h4 costs about $186.00 in ' +
          'foregone gizmo margin. '
        : 'Re-running that against the current best config. The bound has not moved, so the ' +
          'remaining gap is scheduling, not capacity. ';
    this.run([
      ...tokenSteps(reply, 26),
      { after: 200, event: { type: 'agent_done', message: 'Answered.' } },
    ]);
  }

  close(): void {
    this.closed = true;
    this.unsubscribe();
    for (const t of this.timers) clearTimeout(t);
    this.timers = [];
    this.onStatus('closed');
  }
}
