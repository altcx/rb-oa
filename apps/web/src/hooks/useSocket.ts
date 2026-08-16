import { useCallback, useEffect, useRef, useState } from 'react';
import type { ClientEvent, ServerEvent } from '../api/types';
import { isMockMode, socketUrl } from '../api/client';
import { FakeSocket, type SocketLike, type SocketStatus } from '../mock/fakeSocket';

export type { SocketStatus } from '../mock/fakeSocket';

/** Reconnect backoff in ms, capped so a dead backend does not spin. */
const BACKOFF = [500, 1000, 2000, 4000, 8000, 8000] as const;

function parseEvent(raw: string): ServerEvent | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      typeof (parsed as { type?: unknown }).type === 'string'
    ) {
      return parsed as ServerEvent;
    }
  } catch {
    /* fall through */
  }
  return null;
}

export interface SocketHandle {
  status: SocketStatus;
  /** Number of reconnect attempts since the last clean open. */
  attempt: number;
  send: (event: ClientEvent) => void;
}

/**
 * Typed websocket with auto-reconnect. In mock mode it returns a FakeSocket
 * that replays a scripted session instead of dialing the network.
 */
export function useSocket(
  sessionId: string | null,
  onEvent: (event: ServerEvent) => void,
): SocketHandle {
  const [status, setStatus] = useState<SocketStatus>('connecting');
  const [attempt, setAttempt] = useState(0);
  const socketRef = useRef<SocketLike | null>(null);
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;

  useEffect(() => {
    if (!sessionId) {
      setStatus('closed');
      return;
    }

    let disposed = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const dispatch = (event: ServerEvent) => {
      if (!disposed) handlerRef.current(event);
    };

    if (isMockMode()) {
      const fake = new FakeSocket(dispatch, (s) => {
        if (!disposed) setStatus(s);
      });
      socketRef.current = fake;
      return () => {
        disposed = true;
        fake.close();
        socketRef.current = null;
      };
    }

    const connect = () => {
      if (disposed) return;
      setStatus('connecting');
      let ws: WebSocket;
      try {
        ws = new WebSocket(socketUrl(sessionId));
      } catch {
        schedule();
        return;
      }

      const wrapper: SocketLike = {
        send: (event) => {
          if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(event));
        },
        close: () => ws.close(),
      };
      socketRef.current = wrapper;

      ws.onopen = () => {
        if (disposed) return;
        retry = 0;
        setAttempt(0);
        setStatus('open');
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return;
        const event = parseEvent(ev.data);
        if (event) dispatch(event);
      };
      ws.onerror = () => {
        /* onclose follows; reconnect is handled there. */
      };
      ws.onclose = () => {
        if (disposed) return;
        setStatus('closed');
        schedule();
      };
    };

    const schedule = () => {
      if (disposed) return;
      const wait = BACKOFF[Math.min(retry, BACKOFF.length - 1)] ?? 8000;
      retry += 1;
      setAttempt(retry);
      timer = setTimeout(connect, wait);
    };

    connect();

    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [sessionId]);

  const send = useCallback((event: ClientEvent) => {
    socketRef.current?.send(event);
  }, []);

  return { status, attempt, send };
}
