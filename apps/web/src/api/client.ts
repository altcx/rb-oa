import type {
  BuilderRequest,
  CaptureIdResponse,
  CaptureMonitorRequest,
  CaptureRegionRequest,
  CapturesResponse,
  ChartResponse,
  ConfirmRequest,
  ExtractRequest,
  ExtractResponse,
  CreateSessionRequest,
  CreateSessionResponse,
  JobResponse,
  KeyRequest,
  KeyResponse,
  LeaderboardPostRequest,
  LeaderboardResponse,
  ModelsPostRequest,
  ModelsResponse,
  OkResponse,
  OptimizeRequest,
  RulesResponse,
  SessionsResponse,
  StatePayload,
} from './types';
import * as mock from '../mock/mockApi';

/* ------------------------------------------------------------------ */
/* Mock mode                                                           */
/* ------------------------------------------------------------------ */

const MOCK_STORAGE_KEY = 'puzzle-copilot.mock';

function envMock(): boolean {
  return import.meta.env.VITE_MOCK === '1' || import.meta.env.VITE_MOCK === 'true';
}

/**
 * Mock mode is on when `VITE_MOCK=1` at build/dev time, or when the operator
 * flipped the in-UI toggle (persisted in localStorage so a reload keeps it).
 */
export function isMockMode(): boolean {
  try {
    const stored = window.localStorage.getItem(MOCK_STORAGE_KEY);
    if (stored === '1') return true;
    if (stored === '0') return false;
  } catch {
    /* localStorage unavailable (private mode, tests) — fall through to env */
  }
  return envMock();
}

export function setMockMode(on: boolean): void {
  try {
    window.localStorage.setItem(MOCK_STORAGE_KEY, on ? '1' : '0');
  } catch {
    /* ignore */
  }
}

/** True when mock mode was forced by the build/dev env rather than the toggle. */
export function mockForcedByEnv(): boolean {
  return envMock();
}

/* ------------------------------------------------------------------ */
/* HTTP plumbing                                                       */
/* ------------------------------------------------------------------ */

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.text()) || detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, `${path} -> ${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

const get = <T>(path: string) => request<T>(path);
const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) });

/* ------------------------------------------------------------------ */
/* API surface                                                         */
/* ------------------------------------------------------------------ */

export const api = {
  listSessions(): Promise<SessionsResponse> {
    return isMockMode() ? mock.listSessions() : get('/api/sessions');
  },

  createSession(body: CreateSessionRequest): Promise<CreateSessionResponse> {
    return isMockMode() ? mock.createSession(body) : post('/api/sessions', body);
  },

  listCaptures(sessionId: string): Promise<CapturesResponse> {
    return isMockMode()
      ? mock.listCaptures()
      : get(`/api/sessions/${encodeURIComponent(sessionId)}/captures`);
  },

  captureMonitor(body: CaptureMonitorRequest): Promise<CaptureIdResponse> {
    return isMockMode() ? mock.captureMonitor() : post('/api/captures/monitor', body);
  },

  captureRegion(body: CaptureRegionRequest): Promise<CaptureIdResponse> {
    return isMockMode() ? mock.captureRegion() : post('/api/captures/region', body);
  },

  getState(sessionId: string): Promise<StatePayload> {
    return isMockMode()
      ? mock.getState()
      : get(`/api/sessions/${encodeURIComponent(sessionId)}/state`);
  },

  confirmState(sessionId: string, body: ConfirmRequest): Promise<OkResponse> {
    return isMockMode()
      ? mock.ok()
      : post(`/api/sessions/${encodeURIComponent(sessionId)}/state/confirm`, body);
  },

  /** The three money lines + the LP ceiling. Absent lines come back null. */
  getChart(sessionId: string): Promise<ChartResponse> {
    return isMockMode()
      ? mock.getChart()
      : get(`/api/sessions/${encodeURIComponent(sessionId)}/chart`);
  },

  /** Re-run extraction over specific captures (recovery + deliberate re-read). */
  extract(body: ExtractRequest): Promise<ExtractResponse> {
    return isMockMode() ? mock.extract(body) : post('/api/extract', body);
  },

  getRules(sessionId: string): Promise<RulesResponse> {
    return isMockMode()
      ? mock.getRules()
      : get(`/api/sessions/${encodeURIComponent(sessionId)}/rules`);
  },

  solveOptimize(body: OptimizeRequest): Promise<JobResponse> {
    return isMockMode() ? mock.job('opt') : post('/api/solve/optimize', body);
  },

  solveBuilder(body: BuilderRequest): Promise<JobResponse> {
    return isMockMode() ? mock.job('bld') : post('/api/solve/builder', body);
  },

  getLeaderboard(sessionId: string): Promise<LeaderboardResponse> {
    return isMockMode()
      ? mock.getLeaderboard()
      : get(`/api/sessions/${encodeURIComponent(sessionId)}/leaderboard`);
  },

  postLeaderboard(sessionId: string, body: LeaderboardPostRequest): Promise<OkResponse> {
    return isMockMode()
      ? mock.postLeaderboard(body)
      : post(`/api/sessions/${encodeURIComponent(sessionId)}/leaderboard`, body);
  },

  getModels(): Promise<ModelsResponse> {
    return isMockMode() ? mock.getModels() : get('/api/settings/models');
  },

  setModels(body: ModelsPostRequest): Promise<OkResponse> {
    return isMockMode() ? mock.setModels(body) : post('/api/settings/models', body);
  },

  setKey(body: KeyRequest): Promise<KeyResponse> {
    return isMockMode() ? mock.setKey(body) : post('/api/settings/key', body);
  },
};

/** WebSocket URL for a session, derived from the page origin. */
export function socketUrl(sessionId: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/${encodeURIComponent(sessionId)}`;
}
