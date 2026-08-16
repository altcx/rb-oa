/**
 * Types for every payload in the Puzzle Copilot API contract.
 * Kept in one file so the REST surface and the websocket surface can be
 * diffed against the backend contract at a glance.
 */

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export type JsonObject = { [key: string]: JsonValue };

/* ------------------------------------------------------------------ */
/* Sessions                                                            */
/* ------------------------------------------------------------------ */

/** GET /api/sessions -> {sessions:[...]} */
export interface SessionSummary {
  id: string;
  created_at: string;
  puzzle_type: string;
}
export interface SessionsResponse {
  sessions: SessionSummary[];
}

/** POST /api/sessions {puzzle_type} -> {id} */
export interface CreateSessionRequest {
  puzzle_type: string;
}
export interface CreateSessionResponse {
  id: string;
}

/* ------------------------------------------------------------------ */
/* Captures                                                            */
/* ------------------------------------------------------------------ */

/** Pixel rectangle in capture-image coordinates. */
export interface Region {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** GET /api/sessions/{id}/captures -> {captures:[...]} */
export interface Capture {
  id: string;
  created_at: string;
  monitor: number;
  /** null for full-monitor captures. */
  region: Region | null;
  thumb_url: string;
  puzzle_type: string;
  latency_ms: number;
}
export interface CapturesResponse {
  captures: Capture[];
}

/** POST /api/captures/monitor {session_id,monitor} -> {capture_id} */
export interface CaptureMonitorRequest {
  session_id: string;
  monitor: number;
}
/** POST /api/captures/region {session_id,x,y,w,h} -> {capture_id} */
export interface CaptureRegionRequest {
  session_id: string;
  x: number;
  y: number;
  w: number;
  h: number;
}
export interface CaptureIdResponse {
  capture_id: string;
}

/* ------------------------------------------------------------------ */
/* Extracted state + jury verdicts                                     */
/* ------------------------------------------------------------------ */

export type VerdictStatus = 'unanimous' | 'majority' | 'split';

/** Where in a capture this field was read from. box = [x,y,w,h] px. */
export interface Crop {
  capture_id: string;
  box: [number, number, number, number];
}

export interface Verdict {
  path: string;
  value: JsonValue;
  status: VerdictStatus;
  /** model slug -> the value that model voted for. */
  votes: Record<string, JsonValue>;
  alternatives: JsonValue[];
  crop: Crop;
}

/**
 * A money-over-time curve as the backend emits it: a plain array of dollars
 * with length `horizon_hours + 1`.
 *
 * **Index 0 is hour 0** — the money on hand *before* hour 1 runs — and index h
 * is the money at the end of hour h. Plotting index 0 as hour 1 shifts every
 * line by an hour and misreports when the first sale lands, which is the exact
 * thing this chart exists to show. Use `hourlyToPoints` in `lib/money.ts`.
 */
export type MoneyByHour = number[];

/** POST /api/extract */
export type PuzzleType = 'factory' | 'builder';
export interface ExtractRequest {
  session_id: string;
  capture_ids: string[];
  puzzle_type?: PuzzleType;
}
export interface ExtractResponse {
  ok: boolean;
  /** Field paths the jury disagreed on. */
  disputed: string[];
  /** How many fields were settled without asking the operator. */
  auto_confirmed: number;
  unresolved: string[];
  elapsed_ms: number;
  /** True when this was a fast delta re-read rather than a full extraction. */
  used_delta: boolean;
}

/**
 * GET /api/sessions/{id}/chart — the three money lines and the LP ceiling.
 * A line the session does not have is `null`, never zeros: a flat zero line
 * reads as a real and catastrophic run.
 */
export interface ChartResponse {
  current: MoneyByHour | null;
  best: MoneyByHour | null;
  observed: MoneyByHour | null;
  bound: number | null;
  horizon_hours: number | null;
}

/** GET /api/sessions/{id}/state */
export interface StatePayload {
  state: JsonValue;
  verdicts: Verdict[];
  /** field paths with no confident reading at all. */
  unresolved: string[];
}

/** POST /api/sessions/{id}/state/confirm {patch:{path:value}} */
export interface ConfirmRequest {
  patch: Record<string, JsonValue>;
}
export interface OkResponse {
  ok: true;
}

/* ------------------------------------------------------------------ */
/* Rules                                                               */
/* ------------------------------------------------------------------ */

export interface Experiment {
  instruction: string;
  observable: string;
  discriminator: string;
  estimated_seconds: number;
}
export interface UnresolvedRuleFlag {
  flag: string;
  options: string[];
  experiment: Experiment;
}
/** GET /api/sessions/{id}/rules */
export interface RulesResponse {
  rules: JsonObject;
  unresolved: UnresolvedRuleFlag[];
}

/* ------------------------------------------------------------------ */
/* Solvers                                                             */
/* ------------------------------------------------------------------ */

/** POST /api/solve/optimize {session_id,seconds} -> {job_id} */
export interface OptimizeRequest {
  session_id: string;
  seconds: number;
}
/** POST /api/solve/builder {session_id,objective} -> {job_id} */
export interface BuilderRequest {
  session_id: string;
  objective: string;
}
export interface JobResponse {
  job_id: string;
}

/* ------------------------------------------------------------------ */
/* Leaderboard                                                         */
/* ------------------------------------------------------------------ */

export interface LeaderboardEntry {
  id: string;
  label: string;
  score: number;
  tested_at: string;
  config_summary: string;
  is_high_water: boolean;
}
export interface LeaderboardResponse {
  entries: LeaderboardEntry[];
}
/** POST /api/sessions/{id}/leaderboard {label,score,config_summary} */
export interface LeaderboardPostRequest {
  label: string;
  score: number;
  config_summary: string;
}

/* ------------------------------------------------------------------ */
/* Settings                                                            */
/* ------------------------------------------------------------------ */

export interface ModelInfo {
  slug: string;
  family: string;
  vision: boolean;
  structured: boolean;
  tools: boolean;
  /** USD per 1M input tokens. */
  price_in: number;
  /** USD per 1M output tokens. */
  price_out: number;
}

export type RoleName = 'extractor_jury' | 'rule_compiler' | 'strategist' | 'tie_breaker';
export type SingleRoleName = Exclude<RoleName, 'extractor_jury'>;

export interface Roles {
  extractor_jury: [string, string, string];
  rule_compiler: string;
  strategist: string;
  tie_breaker: string;
}

/** GET /api/settings/models */
export interface ModelsResponse {
  catalog: ModelInfo[];
  roles: Roles;
  /** role name -> measured latency in ms. */
  latency: Record<string, number>;
}
/** POST /api/settings/models {roles} */
export interface ModelsPostRequest {
  roles: Roles;
}

/** POST /api/settings/key {key} */
export interface KeyRequest {
  key: string;
}
export interface KeyResponse {
  valid: boolean;
  label: string;
  remaining_credit: number;
  backend: 'keyring' | 'file';
}

/* ------------------------------------------------------------------ */
/* WebSocket /ws/{session_id}                                          */
/* ------------------------------------------------------------------ */

/** One point of the money-over-time curve. */
export interface MoneyPoint {
  hour: number;
  value: number;
}

export interface OptimizerAction {
  target: string;
  setting: string;
  value: JsonValue;
  reason: string;
}

export interface CaptureEvent {
  type: 'capture';
  capture: Capture;
}
export interface ExtractionProgressEvent {
  type: 'extraction_progress';
  stage: string;
  ms: number;
  /** Counts, present on the terminal stage: how many fields the jury split on. */
  disputed?: number;
  /** How many fields were settled without the operator. */
  auto_confirmed?: number;
}
export interface StateEvent extends StatePayload {
  type: 'state';
}
export interface AgentTokenEvent {
  type: 'agent_token';
  text: string;
}
export interface AgentToolCallEvent {
  type: 'agent_tool_call';
  id: string;
  name: string;
  input: JsonValue;
}
export interface AgentToolResultEvent {
  type: 'agent_tool_result';
  id: string;
  name: string;
  output: JsonValue;
  ms: number;
}
export interface AgentDoneEvent {
  type: 'agent_done';
  message: string;
}
export interface OptimizerProgressEvent {
  type: 'optimizer_progress';
  job_id: string;
  /** True on the terminal event: this is the answer, not another tick. */
  final: boolean;
  best_value: number;
  bound: number;
  iterations: number;
  elapsed_s: number;
  /** Streaming curve; null until the worker has one. Index 0 = hour 0. */
  money_by_hour: MoneyByHour | null;
  actions: OptimizerAction[];
  top_warning: string | null;
}
export interface WarningEvent {
  type: 'warning';
  text: string;
}

/** Emitted after a calibration run; the observed line only exists after this. */
export interface CalibrationResult {
  matched?: boolean;
  resolved_flags?: JsonObject;
  [key: string]: JsonValue | undefined;
}
export interface CalibrationEvent {
  type: 'calibration';
  result: CalibrationResult;
}

/**
 * The agent stated a number that appears in no tool result from this
 * conversation. A correctness alarm, not a toast.
 */
export interface ProvenanceWarningEvent {
  type: 'provenance_warning';
  /** Numeric tokens exactly as they appeared in the message text. */
  numbers: string[];
  text: string;
}

export type ServerEvent =
  | CaptureEvent
  | ExtractionProgressEvent
  | StateEvent
  | AgentTokenEvent
  | AgentToolCallEvent
  | AgentToolResultEvent
  | AgentDoneEvent
  | OptimizerProgressEvent
  | WarningEvent
  | CalibrationEvent
  | ProvenanceWarningEvent;

export type ServerEventType = ServerEvent['type'];

export interface ChatClientEvent {
  type: 'chat';
  text: string;
}
export interface CancelClientEvent {
  type: 'cancel';
  job_id: string;
}
export type ClientEvent = ChatClientEvent | CancelClientEvent;
