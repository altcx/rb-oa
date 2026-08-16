import type {
  Capture,
  JsonValue,
  OptimizerProgressEvent,
  ServerEvent,
  StatePayload,
  Verdict,
} from '../api/types';

/* ------------------------------------------------------------------ */
/* Chat transcript items                                               */
/* ------------------------------------------------------------------ */

export interface ChatUserItem {
  kind: 'user';
  id: string;
  text: string;
}
export interface ChatAgentItem {
  kind: 'agent';
  id: string;
  text: string;
  /** false while tokens are still arriving into this bubble. */
  complete: boolean;
}
export interface ChatToolItem {
  kind: 'tool';
  id: string;
  name: string;
  input: JsonValue;
  output: JsonValue | null;
  ms: number | null;
}
export interface ChatNoticeItem {
  kind: 'notice';
  id: string;
  text: string;
}
export type ChatItem = ChatUserItem | ChatAgentItem | ChatToolItem | ChatNoticeItem;

export interface WarningItem {
  id: string;
  text: string;
}

export interface ExtractionProgress {
  stage: string;
  ms: number;
}

/* ------------------------------------------------------------------ */
/* App state                                                           */
/* ------------------------------------------------------------------ */

export interface AppState {
  sessionId: string | null;
  captures: Capture[];
  selectedCaptureId: string | null;
  extraction: ExtractionProgress | null;
  extractionLog: ExtractionProgress[];
  statePayload: StatePayload | null;
  /** Paths the operator has confirmed locally, with the value they chose. */
  accepted: Record<string, JsonValue>;
  chat: ChatItem[];
  optimizer: OptimizerProgressEvent | null;
  /** best_value / bound over time, so the user can see the gap closing. */
  optimizerHistory: Array<{ elapsed_s: number; best_value: number; bound: number }>;
  warnings: WarningItem[];
  /** Monotonic counter used to mint ids without touching Math.random in tests. */
  seq: number;
}

export const initialAppState: AppState = {
  sessionId: null,
  captures: [],
  selectedCaptureId: null,
  extraction: null,
  extractionLog: [],
  statePayload: null,
  accepted: {},
  chat: [],
  optimizer: null,
  optimizerHistory: [],
  warnings: [],
  seq: 0,
};

/* ------------------------------------------------------------------ */
/* Actions                                                             */
/* ------------------------------------------------------------------ */

export type Action =
  | { kind: 'socket'; event: ServerEvent }
  | { kind: 'session'; sessionId: string }
  | { kind: 'hydrate/captures'; captures: Capture[] }
  | { kind: 'hydrate/state'; payload: StatePayload }
  | { kind: 'select-capture'; captureId: string }
  | { kind: 'accept'; path: string; value: JsonValue }
  | { kind: 'accept-all-majority' }
  | { kind: 'chat/send'; text: string }
  | { kind: 'warning/dismiss'; id: string }
  | { kind: 'reset' };

/* ------------------------------------------------------------------ */
/* Reducer                                                             */
/* ------------------------------------------------------------------ */

function nextId(state: AppState, prefix: string): string {
  return `${prefix}_${state.seq + 1}`;
}

export function appReducer(state: AppState, action: Action): AppState {
  switch (action.kind) {
    case 'reset':
      return { ...initialAppState, sessionId: state.sessionId };

    case 'session':
      return { ...state, sessionId: action.sessionId };

    case 'hydrate/captures': {
      const first = action.captures[0];
      return {
        ...state,
        captures: action.captures,
        selectedCaptureId: state.selectedCaptureId ?? (first ? first.id : null),
      };
    }

    case 'hydrate/state':
      return { ...state, statePayload: action.payload, accepted: {} };

    case 'select-capture':
      return { ...state, selectedCaptureId: action.captureId };

    case 'accept':
      return {
        ...state,
        accepted: { ...state.accepted, [action.path]: action.value },
      };

    case 'accept-all-majority': {
      if (!state.statePayload) return state;
      const patch: Record<string, JsonValue> = {};
      for (const v of state.statePayload.verdicts) {
        if (v.status === 'majority' && !(v.path in state.accepted)) {
          patch[v.path] = v.value;
        }
      }
      if (Object.keys(patch).length === 0) return state;
      return { ...state, accepted: { ...state.accepted, ...patch } };
    }

    case 'chat/send':
      return {
        ...state,
        seq: state.seq + 1,
        chat: [...state.chat, { kind: 'user', id: nextId(state, 'user'), text: action.text }],
      };

    case 'warning/dismiss':
      return { ...state, warnings: state.warnings.filter((w) => w.id !== action.id) };

    case 'socket':
      return applyServerEvent(state, action.event);
  }
}

/** The typed websocket dispatcher: one branch per `type` in the contract. */
export function applyServerEvent(state: AppState, event: ServerEvent): AppState {
  switch (event.type) {
    case 'capture': {
      const without = state.captures.filter((c) => c.id !== event.capture.id);
      return {
        ...state,
        // newest first
        captures: [event.capture, ...without],
        selectedCaptureId: event.capture.id,
        extraction: null,
        extractionLog: [],
      };
    }

    case 'extraction_progress': {
      const entry: ExtractionProgress = { stage: event.stage, ms: event.ms };
      return {
        ...state,
        extraction: entry,
        extractionLog: [...state.extractionLog, entry],
      };
    }

    case 'state': {
      const payload: StatePayload = {
        state: event.state,
        verdicts: event.verdicts,
        unresolved: event.unresolved,
      };
      return { ...state, statePayload: payload, accepted: {}, extraction: null };
    }

    case 'agent_token': {
      const last = state.chat[state.chat.length - 1];
      if (last && last.kind === 'agent' && !last.complete) {
        const updated: ChatAgentItem = { ...last, text: last.text + event.text };
        return { ...state, chat: [...state.chat.slice(0, -1), updated] };
      }
      return {
        ...state,
        seq: state.seq + 1,
        chat: [
          ...state.chat,
          { kind: 'agent', id: nextId(state, 'agent'), text: event.text, complete: false },
        ],
      };
    }

    case 'agent_tool_call': {
      // Close any open agent bubble so tokens after the tool start a new one.
      const chat = closeOpenAgentBubble(state.chat);
      const item: ChatToolItem = {
        kind: 'tool',
        id: event.id,
        name: event.name,
        input: event.input,
        output: null,
        ms: null,
      };
      return { ...state, chat: [...chat, item] };
    }

    case 'agent_tool_result': {
      let matched = false;
      const chat = state.chat.map((item) => {
        if (item.kind === 'tool' && item.id === event.id) {
          matched = true;
          return { ...item, name: event.name, output: event.output, ms: event.ms };
        }
        return item;
      });
      if (matched) return { ...state, chat };
      // Result with no preceding call: still surface it rather than dropping it.
      const orphan: ChatToolItem = {
        kind: 'tool',
        id: event.id,
        name: event.name,
        input: null,
        output: event.output,
        ms: event.ms,
      };
      return { ...state, chat: [...closeOpenAgentBubble(state.chat), orphan] };
    }

    case 'agent_done': {
      const chat = closeOpenAgentBubble(state.chat);
      if (!event.message) return { ...state, chat };
      return {
        ...state,
        seq: state.seq + 1,
        chat: [...chat, { kind: 'notice', id: nextId(state, 'notice'), text: event.message }],
      };
    }

    case 'optimizer_progress': {
      const history = [
        ...state.optimizerHistory,
        { elapsed_s: event.elapsed_s, best_value: event.best_value, bound: event.bound },
      ];
      return { ...state, optimizer: event, optimizerHistory: history };
    }

    case 'warning':
      return {
        ...state,
        seq: state.seq + 1,
        warnings: [...state.warnings, { id: nextId(state, 'warn'), text: event.text }],
      };
  }
}

function closeOpenAgentBubble(chat: ChatItem[]): ChatItem[] {
  const last = chat[chat.length - 1];
  if (last && last.kind === 'agent' && !last.complete) {
    return [...chat.slice(0, -1), { ...last, complete: true }];
  }
  return chat;
}

/* ------------------------------------------------------------------ */
/* Selectors                                                           */
/* ------------------------------------------------------------------ */

export function allVerdicts(state: AppState): Verdict[] {
  return state.statePayload?.verdicts ?? [];
}

/** Every field the jury did not agree on, in the original payload order. */
export function contestedVerdicts(state: AppState): Verdict[] {
  return allVerdicts(state).filter((v) => v.status !== 'unanimous');
}

/**
 * Disputed fields still awaiting the operator, hardest first: `split` outranks
 * `majority`, because a split has no safe default to fall back on.
 */
export function openDisputes(state: AppState): Verdict[] {
  const rank = (v: Verdict) => (v.status === 'split' ? 0 : 1);
  return contestedVerdicts(state)
    .filter((v) => !(v.path in state.accepted))
    .sort((a, b) => rank(a) - rank(b) || a.path.localeCompare(b.path));
}

/** Unanimous fields, plus disputes the operator has already settled. */
export function settledVerdicts(state: AppState): Verdict[] {
  return allVerdicts(state).filter(
    (v) => v.status === 'unanimous' || v.path in state.accepted,
  );
}

export interface ReviewCounts {
  fields: number;
  disputed: number;
  remaining: number;
  agreed: number;
  majorityRemaining: number;
}

export function reviewCounts(state: AppState): ReviewCounts {
  const verdicts = allVerdicts(state);
  const contested = verdicts.filter((v) => v.status !== 'unanimous');
  const remaining = contested.filter((v) => !(v.path in state.accepted));
  return {
    fields: verdicts.length,
    disputed: contested.length,
    remaining: remaining.length,
    agreed: verdicts.length - remaining.length,
    majorityRemaining: remaining.filter((v) => v.status === 'majority').length,
  };
}

/** The value shown for a field: the operator's choice wins over the jury's. */
export function effectiveValue(state: AppState, verdict: Verdict): JsonValue {
  return verdict.path in state.accepted
    ? (state.accepted[verdict.path] as JsonValue)
    : verdict.value;
}
