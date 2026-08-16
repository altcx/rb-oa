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
  best_value: number;
  bound: number;
  iterations: number;
  elapsed_s: number;
  money_by_hour: MoneyPoint[];
  actions: OptimizerAction[];
  top_warning: string | null;
}
export interface WarningEvent {
  type: 'warning';
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
  | WarningEvent;

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
