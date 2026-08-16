import type { MoneyPoint } from '../api/types';
import type { ChatItem } from '../state/reducer';

/**
 * The contract streams the optimizer's curve directly, but the "current config"
 * curve only exists inside a `simulate_config` tool result. Pull the newest one
 * out of the transcript rather than showing an empty line.
 */
export function currentCurveFromChat(chat: ChatItem[]): MoneyPoint[] {
  for (let i = chat.length - 1; i >= 0; i--) {
    const item = chat[i];
    if (!item || item.kind !== 'tool' || item.output === null) continue;
    if (typeof item.output !== 'object' || Array.isArray(item.output)) continue;
    const byHour = (item.output as Record<string, unknown>)['by_hour'];
    if (!Array.isArray(byHour)) continue;
    const points: MoneyPoint[] = [];
    byHour.forEach((v, hour) => {
      if (typeof v === 'number') points.push({ hour, value: v });
    });
    if (points.length > 0) return points;
  }
  return [];
}
