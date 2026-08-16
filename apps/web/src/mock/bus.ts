/**
 * The mock REST layer and the fake socket are separate modules, but in a real
 * session a capture POST is what causes a `capture` event to arrive on the
 * socket. This one-line bus reproduces that causality so the capture button and
 * the `C` hotkey actually do something in mock mode.
 */
type Listener = () => void;

const listeners = new Set<Listener>();

export function emitMockCapture(): void {
  for (const l of [...listeners]) l();
}

export function onMockCapture(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
