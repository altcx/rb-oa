import type {
  Capture,
  LeaderboardEntry,
  ModelsResponse,
  MoneyPoint,
  RulesResponse,
  SessionSummary,
  StatePayload,
  Verdict,
} from '../api/types';

/* ------------------------------------------------------------------ */
/* Synthetic capture images                                            */
/* ------------------------------------------------------------------ */

export const CAPTURE_W = 1280;
export const CAPTURE_H = 720;

/**
 * A deterministic SVG "screenshot" of the game board, as a data URI.
 * Real enough that the zoomed crop windows land on legible numbers.
 */
function boardSvg(seed: number): string {
  const cells: string[] = [];
  const cols = 6;
  const rows = 4;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const x = 60 + c * 195;
      const y = 130 + r * 140;
      const v = ((i * 37 + seed * 13) % 97) + 3;
      const money = (v * 12.5 + seed).toFixed(2);
      cells.push(
        `<rect x="${x}" y="${y}" width="175" height="118" rx="6" fill="#15181d" stroke="#2c333d"/>` +
          `<text x="${x + 12}" y="${y + 30}" fill="#7d8794" font-family="monospace" font-size="14">M${i + 1}</text>` +
          `<text x="${x + 12}" y="${y + 66}" fill="#e6edf3" font-family="monospace" font-size="30">${v}</text>` +
          `<text x="${x + 12}" y="${y + 98}" fill="#199e70" font-family="monospace" font-size="18">$${money}</text>`,
      );
    }
  }
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${CAPTURE_W}" height="${CAPTURE_H}" viewBox="0 0 ${CAPTURE_W} ${CAPTURE_H}">` +
    `<rect width="${CAPTURE_W}" height="${CAPTURE_H}" fill="#0b0d10"/>` +
    `<text x="60" y="70" fill="#e6edf3" font-family="monospace" font-size="34">FACTORY LINE — TICK ${seed}</text>` +
    `<text x="900" y="70" fill="#d95926" font-family="monospace" font-size="26">CASH $${(4820.5 + seed * 91.25).toFixed(2)}</text>` +
    cells.join('') +
    `</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const mockCaptureImage = boardSvg;

/* ------------------------------------------------------------------ */
/* Sessions & captures                                                 */
/* ------------------------------------------------------------------ */

export const MOCK_SESSION_ID = 'sess_9f2c1a';

export const mockSessions: SessionSummary[] = [
  { id: MOCK_SESSION_ID, created_at: '2026-08-16T10:04:11Z', puzzle_type: 'factory_line' },
  { id: 'sess_71ba03', created_at: '2026-08-15T21:47:02Z', puzzle_type: 'factory_line' },
  { id: 'sess_3ce980', created_at: '2026-08-15T18:12:55Z', puzzle_type: 'rail_yard' },
];

function capture(i: number, monitor: number, latency: number, iso: string): Capture {
  return {
    id: `cap_${String(i).padStart(3, '0')}`,
    created_at: iso,
    monitor,
    region: i % 3 === 0 ? { x: 240, y: 120, w: 800, h: 460 } : null,
    thumb_url: boardSvg(i),
    puzzle_type: 'factory_line',
    latency_ms: latency,
  };
}

export const mockCaptures: Capture[] = [
  capture(4, 1, 2180, '2026-08-16T10:09:40Z'),
  capture(3, 1, 5310, '2026-08-16T10:08:02Z'),
  capture(2, 1, 3740, '2026-08-16T10:06:31Z'),
  capture(1, 1, 2960, '2026-08-16T10:05:12Z'),
];

/** The capture the live socket replay pushes in. */
export const mockIncomingCapture: Capture = capture(5, 1, 3420, '2026-08-16T10:11:18Z');

/* ------------------------------------------------------------------ */
/* Extracted state: 40 fields, 3 disputed                              */
/* ------------------------------------------------------------------ */

const AGREED_FIELDS: Array<[string, string | number | boolean]> = [
  ['clock.tick', 47],
  ['clock.seconds_remaining', 613],
  ['cash.on_hand', 4820.5],
  ['cash.debt', 1200.0],
  ['cash.interest_rate_per_hour', 0.015],
  ['market.price_widget', 18.75],
  ['market.price_gizmo', 27.4],
  ['market.demand_widget_per_hour', 240],
  ['market.demand_gizmo_per_hour', 96],
  ['machines[0].kind', 'press'],
  ['machines[0].throughput_per_hour', 132],
  ['machines[0].upkeep_per_hour', 41.25],
  ['machines[1].kind', 'press'],
  ['machines[1].throughput_per_hour', 128],
  ['machines[1].upkeep_per_hour', 41.25],
  ['machines[2].kind', 'lathe'],
  ['machines[2].upkeep_per_hour', 58.0],
  ['machines[3].kind', 'lathe'],
  ['machines[3].throughput_per_hour', 104],
  ['machines[3].upkeep_per_hour', 58.0],
  ['machines[4].kind', 'assembler'],
  ['machines[4].throughput_per_hour', 76],
  ['machines[4].upkeep_per_hour', 92.5],
  ['inventory.raw', 1840],
  ['inventory.widget', 312],
  ['inventory.gizmo', 47],
  ['staff.operators', 6],
  ['staff.wage_per_hour', 22.0],
  ['staff.fatigue', 0.31],
  ['power.draw_kw', 418],
  ['power.cap_kw', 500],
  ['power.tariff_per_kwh', 0.19],
  ['contracts[0].buyer', 'Northgate'],
  ['contracts[0].units_per_hour', 60],
  ['contracts[0].unit_price', 21.0],
  ['contracts[1].buyer', 'Halvern'],
  ['contracts[1].units_per_hour', 35],
];

const DISPUTED: Verdict[] = [
  {
    path: 'contracts[1].penalty_per_late_unit',
    value: 12.5,
    status: 'split',
    votes: {
      'anthropic/claude-sonnet-4.5': 12.5,
      'openai/gpt-5-mini': 1.25,
      'google/gemini-2.5-flash': 125.0,
    },
    alternatives: [1.25, 125.0],
    crop: { capture_id: 'cap_004', box: [645, 545, 190, 90] },
  },
  {
    path: 'machines[2].throughput_per_hour',
    value: 118,
    status: 'majority',
    votes: {
      'anthropic/claude-sonnet-4.5': 118,
      'openai/gpt-5-mini': 118,
      'google/gemini-2.5-flash': 116,
    },
    alternatives: [116],
    crop: { capture_id: 'cap_004', box: [450, 265, 190, 120] },
  },
  {
    path: 'market.demand_curve_slope',
    value: -0.42,
    status: 'majority',
    votes: {
      'anthropic/claude-sonnet-4.5': -0.42,
      'openai/gpt-5-mini': -0.48,
      'google/gemini-2.5-flash': -0.42,
    },
    alternatives: [-0.48],
    crop: { capture_id: 'cap_004', box: [60, 405, 220, 110] },
  },
];

function agreedVerdict(path: string, value: string | number | boolean, i: number): Verdict {
  const col = i % 6;
  const row = Math.floor(i / 6) % 4;
  return {
    path,
    value,
    status: 'unanimous',
    votes: {
      'anthropic/claude-sonnet-4.5': value,
      'openai/gpt-5-mini': value,
      'google/gemini-2.5-flash': value,
    },
    alternatives: [],
    crop: { capture_id: 'cap_004', box: [60 + col * 195, 130 + row * 140, 175, 118] },
  };
}

export const mockVerdicts: Verdict[] = [
  ...AGREED_FIELDS.map(([p, v], i) => agreedVerdict(p, v, i)),
  ...DISPUTED,
];

function buildStateObject(): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const v of mockVerdicts) out[v.path] = v.value;
  return out;
}

export const mockState: StatePayload = {
  state: buildStateObject() as StatePayload['state'],
  verdicts: mockVerdicts,
  unresolved: ['contracts[1].penalty_per_late_unit'],
};

/* ------------------------------------------------------------------ */
/* Rules                                                               */
/* ------------------------------------------------------------------ */

export const mockRules: RulesResponse = {
  rules: {
    'production.parallel_lines': true,
    'production.setup_cost_applies_on_switch': true,
    'production.setup_seconds': 90,
    'market.price_moves_with_volume': true,
    'market.price_floor': 6.0,
    'finance.interest_compounds_hourly': true,
    'finance.overdraft_allowed': false,
    'staff.overtime_multiplier': 1.5,
    'power.brownout_at_cap': true,
    'contracts.penalty_caps_at_contract_value': true,
  },
  unresolved: [
    {
      flag: 'inventory.spoilage_applies_to_gizmo',
      options: ['no spoilage', '2%/hour', '5%/hour'],
      experiment: {
        instruction:
          'Stockpile exactly 100 gizmos, then idle the assembler for one full in-game hour without selling.',
        observable: 'The gizmo count in the inventory panel at the top of hour 2.',
        discriminator: '100 = no spoilage; 98 = 2%/hour; 95 = 5%/hour.',
        estimated_seconds: 75,
      },
    },
    {
      flag: 'market.demand_resets_on_new_contract',
      options: ['resets to base', 'carries over'],
      experiment: {
        instruction:
          'With widget demand visibly depressed below 200/h, sign the Halvern contract and watch the demand readout on the next tick.',
        observable: 'market.demand_widget_per_hour immediately after signing.',
        discriminator: 'Jumps back to 240 = resets to base; stays under 200 = carries over.',
        estimated_seconds: 40,
      },
    },
    {
      flag: 'power.brownout_scales_or_halts',
      options: ['throughput scales linearly', 'line halts entirely'],
      experiment: {
        instruction: 'Switch on a fifth machine so draw exceeds the 500 kW cap by ~10%, hold for 30s.',
        observable: 'Units produced during the overdraw window vs the preceding window.',
        discriminator: '~90% of nominal = scales; 0 units = halts.',
        estimated_seconds: 55,
      },
    },
  ],
};

/* ------------------------------------------------------------------ */
/* Money curves                                                        */
/* ------------------------------------------------------------------ */

const HOURS = 12;

function curve(peak: number, shape: number, noise: number): MoneyPoint[] {
  const pts: MoneyPoint[] = [];
  let acc = 0;
  for (let h = 0; h <= HOURS; h++) {
    const t = h / HOURS;
    const rate = peak * (1 - Math.exp(-shape * t));
    acc += rate;
    const wobble = Math.sin(h * 1.7) * noise;
    pts.push({ hour: h, value: Math.round((acc + wobble) * 100) / 100 });
  }
  return pts;
}

/** The config currently loaded in the UI. */
export const mockCurrentCurve: MoneyPoint[] = curve(980, 2.4, 55);
/** What the game actually paid out, observed from captures — stops at "now". */
export const mockObservedCurve: MoneyPoint[] = curve(910, 2.1, 140).slice(0, 8);
/** Optimizer best-so-far; improves across the replay. */
export function mockOptimizerCurve(step: number): MoneyPoint[] {
  return curve(980 + step * 46, 2.4 + step * 0.06, 30);
}

export const MOCK_LP_BOUND = 14260.0;

/* ------------------------------------------------------------------ */
/* Leaderboard                                                         */
/* ------------------------------------------------------------------ */

export const mockLeaderboard: LeaderboardEntry[] = [
  {
    id: 'lb_006',
    label: 'gizmo pivot @ h4',
    score: 12480.75,
    tested_at: '2026-08-16T10:02:40Z',
    config_summary: '2 press / 2 lathe / 1 asm, gizmo from h4, 6 ops',
    is_high_water: true,
  },
  {
    id: 'lb_005',
    label: 'all-widget baseline',
    score: 11930.2,
    tested_at: '2026-08-16T09:51:12Z',
    config_summary: '3 press / 1 lathe / 1 asm, widgets only, 6 ops',
    is_high_water: false,
  },
  {
    id: 'lb_004',
    label: 'overtime push',
    score: 11902.55,
    tested_at: '2026-08-16T09:44:03Z',
    config_summary: 'baseline + overtime from h6, 8 ops',
    is_high_water: false,
  },
  {
    id: 'lb_003',
    label: 'debt-funded 4th press',
    score: 10744.0,
    tested_at: '2026-08-16T09:31:58Z',
    config_summary: '4 press / 1 asm, +$1200 debt at h1',
    is_high_water: false,
  },
  {
    id: 'lb_002',
    label: 'contract-heavy',
    score: 9812.4,
    tested_at: '2026-08-16T09:22:19Z',
    config_summary: 'Northgate + Halvern both signed at h0',
    is_high_water: false,
  },
  {
    id: 'lb_001',
    label: 'first read',
    score: 7104.9,
    tested_at: '2026-08-16T09:12:44Z',
    config_summary: 'as-found configuration, no changes',
    is_high_water: false,
  },
];

/* ------------------------------------------------------------------ */
/* Models & settings                                                   */
/* ------------------------------------------------------------------ */

export const mockModels: ModelsResponse = {
  catalog: [
    {
      slug: 'anthropic/claude-sonnet-4.5',
      family: 'anthropic',
      vision: true,
      structured: true,
      tools: true,
      price_in: 3.0,
      price_out: 15.0,
    },
    {
      slug: 'anthropic/claude-haiku-4.5',
      family: 'anthropic',
      vision: true,
      structured: true,
      tools: true,
      price_in: 1.0,
      price_out: 5.0,
    },
    {
      slug: 'openai/gpt-5',
      family: 'openai',
      vision: true,
      structured: true,
      tools: true,
      price_in: 1.25,
      price_out: 10.0,
    },
    {
      slug: 'openai/gpt-5-mini',
      family: 'openai',
      vision: true,
      structured: true,
      tools: true,
      price_in: 0.25,
      price_out: 2.0,
    },
    {
      slug: 'google/gemini-2.5-pro',
      family: 'google',
      vision: true,
      structured: true,
      tools: true,
      price_in: 1.25,
      price_out: 10.0,
    },
    {
      slug: 'google/gemini-2.5-flash',
      family: 'google',
      vision: true,
      structured: true,
      tools: true,
      price_in: 0.3,
      price_out: 2.5,
    },
    {
      slug: 'meta-llama/llama-4-maverick',
      family: 'meta',
      vision: true,
      structured: false,
      tools: true,
      price_in: 0.27,
      price_out: 0.85,
    },
    {
      slug: 'qwen/qwen3-vl-235b',
      family: 'qwen',
      vision: true,
      structured: true,
      tools: false,
      price_in: 0.35,
      price_out: 1.4,
    },
    {
      slug: 'deepseek/deepseek-v3.2',
      family: 'deepseek',
      vision: false,
      structured: true,
      tools: true,
      price_in: 0.28,
      price_out: 0.42,
    },
  ],
  roles: {
    extractor_jury: [
      'anthropic/claude-sonnet-4.5',
      'openai/gpt-5-mini',
      'google/gemini-2.5-flash',
    ],
    rule_compiler: 'anthropic/claude-sonnet-4.5',
    strategist: 'openai/gpt-5',
    tie_breaker: 'google/gemini-2.5-pro',
  },
  latency: {
    extractor_jury: 3420,
    rule_compiler: 5180,
    strategist: 7940,
    tie_breaker: 2260,
  },
};

export const mockKeyResult = {
  valid: true,
  label: 'sk-or-v1-…8f3a (Puzzle Copilot)',
  remaining_credit: 42.87,
  backend: 'keyring' as const,
};
