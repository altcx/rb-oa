import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, api, isMockMode } from '../api/client';
import type {
  ChartResponse,
  ExtractResponse,
  JsonValue,
  LeaderboardEntry,
  RulesResponse,
  ServerEvent,
} from '../api/types';
import { useStore } from '../state/store';
import { useSocket, type SocketStatus } from './useSocket';

export interface SessionHandle {
  sessionId: string | null;
  status: SocketStatus;
  reconnectAttempt: number;
  rules: RulesResponse | null;
  leaderboard: LeaderboardEntry[];
  chart: ChartResponse | null;
  /** Result of the last on-demand extraction, for the filmstrip readout. */
  lastExtract: ExtractResponse | null;
  extracting: boolean;
  /** Set when the backend could not be reached; mock mode is the way out. */
  error: string | null;
  /** The optimizer's calibration gate (HTTP 412), which is not a plain error. */
  calibrationGate: string | null;
  captureMonitor: (monitor: number) => Promise<void>;
  captureRegion: (x: number, y: number, w: number, h: number) => Promise<void>;
  confirm: (patch: Record<string, JsonValue>) => Promise<void>;
  extract: (captureIds: string[]) => Promise<void>;
  optimize: (seconds: number) => Promise<void>;
  cancel: (jobId: string) => void;
  chat: (text: string) => void;
  refreshLeaderboard: () => Promise<void>;
}

/**
 * Boots a session (or joins the newest one), hydrates the REST snapshots, and
 * keeps the websocket wired into the store.
 */
export function useSession(): SessionHandle {
  const dispatch = useStore((s) => s.dispatch);
  const sessionId = useStore((s) => s.sessionId);
  const [rules, setRules] = useState<RulesResponse | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [chart, setChart] = useState<ChartResponse | null>(null);
  const [lastExtract, setLastExtract] = useState<ExtractResponse | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [calibrationGate, setCalibrationGate] = useState<string | null>(null);
  const booted = useRef(false);
  const sessionIdRef = useRef<string | null>(null);
  sessionIdRef.current = sessionId;

  const refreshChart = useCallback(async () => {
    const id = sessionIdRef.current;
    if (!id) return;
    try {
      setChart(await api.getChart(id));
    } catch {
      // Keep the last good chart rather than blanking the most informative view.
    }
  }, []);

  const onEvent = useCallback(
    (event: ServerEvent) => {
      dispatch({ kind: 'socket', event });

      // The chart is REST-backed, so it has to be pulled when the things it is
      // derived from change. Non-final optimizer ticks are deliberately NOT a
      // trigger: they carry their own `money_by_hour` in band, so the best line
      // animates live without a refetch per tick.
      if (
        event.type === 'state' ||
        event.type === 'calibration' ||
        (event.type === 'optimizer_progress' && event.final)
      ) {
        void refreshChart();
      }
    },
    [dispatch, refreshChart],
  );

  const socket = useSocket(sessionId, onEvent);

  useEffect(() => {
    // Guarded by a ref rather than an abort flag: StrictMode's double-invoke
    // would otherwise cancel the first pass and skip the second.
    if (booted.current) return;
    booted.current = true;

    void (async () => {
      try {
        const { sessions } = await api.listSessions();
        const existing = sessions[0];
        const id = existing ? existing.id : (await api.createSession({ puzzle_type: 'auto' })).id;
        dispatch({ kind: 'session', sessionId: id });

        const [captures, state, rulesRes, board, chartRes] = await Promise.all([
          api.listCaptures(id),
          api.getState(id).catch(() => null),
          api.getRules(id).catch(() => null),
          api.getLeaderboard(id).catch(() => null),
          api.getChart(id).catch(() => null),
        ]);
        dispatch({ kind: 'hydrate/captures', captures: captures.captures });
        if (state) dispatch({ kind: 'hydrate/state', payload: state });
        if (rulesRes) setRules(rulesRes);
        if (board) setLeaderboard(board.entries);
        if (chartRes) setChart(chartRes);
        setError(null);
      } catch (e) {
        setError(
          isMockMode()
            ? `Mock fixtures failed to load: ${String(e)}`
            : 'Backend unreachable. Turn on mock mode to work against fixtures.',
        );
      }
    })();
  }, [dispatch]);

  const refreshLeaderboard = useCallback(async () => {
    if (!sessionId) return;
    try {
      const board = await api.getLeaderboard(sessionId);
      setLeaderboard(board.entries);
    } catch {
      /* leave the last known board on screen rather than blanking it */
    }
  }, [sessionId]);

  const captureMonitor = useCallback(
    async (monitor: number) => {
      if (!sessionId) return;
      try {
        await api.captureMonitor({ session_id: sessionId, monitor });
      } catch (e) {
        setError(`Capture failed: ${String(e)}`);
      }
    },
    [sessionId],
  );

  const captureRegion = useCallback(
    async (x: number, y: number, w: number, h: number) => {
      if (!sessionId) return;
      try {
        await api.captureRegion({ session_id: sessionId, x, y, w, h });
      } catch (e) {
        setError(`Region capture failed: ${String(e)}`);
      }
    },
    [sessionId],
  );

  const confirm = useCallback(
    async (patch: Record<string, JsonValue>) => {
      if (!sessionId) return;
      try {
        await api.confirmState(sessionId, { patch });
      } catch (e) {
        setError(`Confirm failed: ${String(e)}`);
      }
    },
    [sessionId],
  );

  const extract = useCallback(
    async (captureIds: string[]) => {
      if (!sessionId || captureIds.length === 0) return;
      setExtracting(true);
      try {
        const res = await api.extract({ session_id: sessionId, capture_ids: captureIds });
        setLastExtract(res);
      } catch (e) {
        setError(`Re-extraction failed: ${String(e)}`);
      } finally {
        setExtracting(false);
      }
    },
    [sessionId],
  );

  const optimize = useCallback(
    async (seconds: number) => {
      if (!sessionId) return;
      try {
        await api.solveOptimize({ session_id: sessionId, seconds });
        setCalibrationGate(null);
      } catch (e) {
        // 412 is the calibration gate: no recorded run has matched the
        // simulator yet. That is a precondition to explain, not a failure.
        if (e instanceof ApiError && e.status === 412) {
          setCalibrationGate(
            'Calibration gate: no recorded run has matched the simulator yet, so optimizer ' +
              'output would be unverified. Run a calibration first.',
          );
          return;
        }
        setError(`Optimizer refused the job: ${String(e)}`);
      }
    },
    [sessionId],
  );

  const cancel = useCallback(
    (jobId: string) => {
      socket.send({ type: 'cancel', job_id: jobId });
    },
    [socket],
  );

  const chat = useCallback(
    (text: string) => {
      socket.send({ type: 'chat', text });
    },
    [socket],
  );

  return {
    sessionId,
    status: socket.status,
    reconnectAttempt: socket.attempt,
    rules,
    leaderboard,
    chart,
    lastExtract,
    extracting,
    error,
    calibrationGate,
    captureMonitor,
    captureRegion,
    confirm,
    extract,
    optimize,
    cancel,
    chat,
    refreshLeaderboard,
  };
}
