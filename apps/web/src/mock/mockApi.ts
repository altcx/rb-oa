import type {
  CaptureIdResponse,
  CapturesResponse,
  ChartResponse,
  CreateSessionRequest,
  ExtractRequest,
  ExtractResponse,
  CreateSessionResponse,
  JobResponse,
  KeyRequest,
  KeyResponse,
  LeaderboardEntry,
  LeaderboardPostRequest,
  LeaderboardResponse,
  ModelsPostRequest,
  ModelsResponse,
  OkResponse,
  RulesResponse,
  SessionsResponse,
  StatePayload,
} from '../api/types';
import { emitMockCapture } from './bus';
import {
  MOCK_SESSION_ID,
  mockCaptures,
  mockChart,
  mockChartAfterCalibration,
  mockKeyResult,
  mockLeaderboard,
  mockModels,
  mockRules,
  mockSessions,
  mockState,
} from './fixtures';

/** Mock latency, small enough not to make the UI feel fake-slow. */
const delay = (ms = 90) => new Promise<void>((r) => setTimeout(r, ms));

async function respond<T>(value: T, ms?: number): Promise<T> {
  await delay(ms);
  return value;
}

/* Mutable copies so POSTs in mock mode have a visible effect. */
let leaderboard: LeaderboardEntry[] = [...mockLeaderboard];
let roles = { ...mockModels.roles };
let captureSeq = 100;

export function listSessions(): Promise<SessionsResponse> {
  return respond({ sessions: mockSessions });
}

export function createSession(body: CreateSessionRequest): Promise<CreateSessionResponse> {
  return respond({ id: `sess_new_${body.puzzle_type}` });
}

export function listCaptures(): Promise<CapturesResponse> {
  return respond({ captures: mockCaptures });
}

export function captureMonitor(): Promise<CaptureIdResponse> {
  captureSeq += 1;
  emitMockCapture();
  return respond({ capture_id: `cap_${captureSeq}` }, 40);
}

export function captureRegion(): Promise<CaptureIdResponse> {
  captureSeq += 1;
  emitMockCapture();
  return respond({ capture_id: `cap_${captureSeq}` }, 40);
}

export function getState(): Promise<StatePayload> {
  return respond(mockState);
}

/**
 * Before calibration the observed line does not exist and the optimizer has not
 * run, so both come back null. The fake socket flips this once it replays the
 * calibration event, which is how mock mode demos the absent-line states.
 */
let calibrated = false;

export function markMockCalibrated(): void {
  calibrated = true;
}

/** Reset the mutable mock session, so tests do not leak state into each other. */
export function resetMockState(): void {
  calibrated = false;
  extractedCaptures.clear();
  leaderboard = [...mockLeaderboard];
  roles = { ...mockModels.roles };
}

export function getChart(): Promise<ChartResponse> {
  return respond(calibrated ? mockChartAfterCalibration : mockChart);
}

export function extract(body: ExtractRequest): Promise<ExtractResponse> {
  // A re-read of a capture already extracted is a fast delta pass.
  const isDelta = extractedCaptures.size > 0;
  for (const id of body.capture_ids) extractedCaptures.add(id);
  return respond(
    {
      ok: true,
      disputed: mockState.verdicts.filter((v) => v.status !== 'unanimous').map((v) => v.path),
      auto_confirmed: mockState.verdicts.filter((v) => v.status === 'unanimous').length,
      unresolved: mockState.unresolved,
      elapsed_ms: isDelta ? 1180 : 3420,
      used_delta: isDelta,
    },
    isDelta ? 300 : 900,
  );
}

const extractedCaptures = new Set<string>();

export function ok(): Promise<OkResponse> {
  return respond({ ok: true as const }, 40);
}

export function getRules(): Promise<RulesResponse> {
  return respond(mockRules);
}

export function job(prefix: string): Promise<JobResponse> {
  return respond({ job_id: `${prefix}_${Date.now().toString(36)}` }, 40);
}

export function getLeaderboard(): Promise<LeaderboardResponse> {
  return respond({ entries: leaderboard });
}

export function postLeaderboard(body: LeaderboardPostRequest): Promise<OkResponse> {
  const entry: LeaderboardEntry = {
    id: `lb_${String(leaderboard.length + 1).padStart(3, '0')}`,
    label: body.label,
    score: body.score,
    tested_at: new Date().toISOString(),
    config_summary: body.config_summary,
    is_high_water: false,
  };
  const next = [entry, ...leaderboard];
  let high = next[0] as LeaderboardEntry;
  for (const e of next) if (e.score > high.score) high = e;
  leaderboard = next.map((e) => ({ ...e, is_high_water: e.id === high.id }));
  return ok();
}

export function getModels(): Promise<ModelsResponse> {
  return respond({ ...mockModels, roles });
}

export function setModels(body: ModelsPostRequest): Promise<OkResponse> {
  roles = { ...body.roles };
  return ok();
}

export function setKey(body: KeyRequest): Promise<KeyResponse> {
  const looksReal = body.key.trim().startsWith('sk-or-') && body.key.trim().length >= 20;
  if (!looksReal) {
    return respond(
      {
        valid: false,
        label: 'rejected — expected an sk-or-… OpenRouter key',
        remaining_credit: 0,
        backend: 'file' as const,
      },
      320,
    );
  }
  return respond(mockKeyResult, 420);
}

export const MOCK_SESSION = MOCK_SESSION_ID;
