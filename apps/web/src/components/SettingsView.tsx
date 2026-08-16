import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, isMockMode, mockForcedByEnv, setMockMode } from '../api/client';
import type {
  KeyResponse,
  ModelInfo,
  ModelsResponse,
  Roles,
  SingleRoleName,
} from '../api/types';
import { formatMoney, formatMs, latencyColorClass } from '../lib/format';
import { Link } from '../router';
import { Badge, Button, Panel, StatusDot } from './ui';

const SINGLE_ROLES: Array<{ role: SingleRoleName; label: string; blurb: string }> = [
  { role: 'rule_compiler', label: 'Rule compiler', blurb: 'turns observations into rule flags' },
  { role: 'strategist', label: 'Strategist', blurb: 'plans the run and answers chat' },
  { role: 'tie_breaker', label: 'Tie breaker', blurb: 'called only when the jury splits' },
];

function ModelSelect({
  value,
  catalog,
  onChange,
  id,
  requireVision,
}: {
  value: string;
  catalog: ModelInfo[];
  onChange: (slug: string) => void;
  id: string;
  requireVision?: boolean;
}) {
  return (
    <select
      id={id}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="num w-full rounded border border-ink-600 bg-ink-900 px-1.5 py-1 text-xs text-ink-100"
    >
      {catalog.map((m) => (
        <option key={m.slug} value={m.slug} disabled={requireVision && !m.vision}>
          {m.slug}
          {requireVision && !m.vision ? ' (no vision)' : ''}
        </option>
      ))}
    </select>
  );
}

export function SettingsView() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [roles, setRoles] = useState<Roles | null>(null);
  const [saved, setSaved] = useState<'idle' | 'saving' | 'saved'>('idle');
  const [error, setError] = useState<string | null>(null);

  const [key, setKey] = useState('');
  const [keyResult, setKeyResult] = useState<KeyResponse | null>(null);
  const [validating, setValidating] = useState(false);

  const [mock, setMockState] = useState(isMockMode());

  useEffect(() => {
    let live = true;
    api
      .getModels()
      .then((res) => {
        if (!live) return;
        setModels(res);
        setRoles(res.roles);
      })
      .catch(() =>
        setError('Could not load the model catalog. Backend down, or mock mode is off.'),
      );
    return () => {
      live = false;
    };
  }, []);

  const familyOf = useCallback(
    (slug: string) => models?.catalog.find((m) => m.slug === slug)?.family ?? 'unknown',
    [models],
  );

  const juryFamilyWarning = useMemo(() => {
    if (!roles) return null;
    const families = roles.extractor_jury.map(familyOf);
    const dupes = families.filter((f, i) => families.indexOf(f) !== i);
    if (dupes.length === 0) return null;
    const unique = [...new Set(dupes)];
    return `Two or more jury slots are from the same family (${unique.join(
      ', ',
    )}). Same-family models fail the same way, so a 2-1 majority may just be one shared blind spot.`;
  }, [roles, familyOf]);

  const setJury = (index: 0 | 1 | 2, slug: string) => {
    if (!roles) return;
    const jury: [string, string, string] = [...roles.extractor_jury];
    jury[index] = slug;
    setRoles({ ...roles, extractor_jury: jury });
    setSaved('idle');
  };

  const save = async () => {
    if (!roles) return;
    setSaved('saving');
    try {
      await api.setModels({ roles });
      setSaved('saved');
    } catch {
      setError('Saving roles failed.');
      setSaved('idle');
    }
  };

  const validateKey = async () => {
    setValidating(true);
    try {
      setKeyResult(await api.setKey({ key }));
    } catch {
      setError('Key validation request failed.');
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="flex h-full flex-col bg-ink-950">
      <header className="flex shrink-0 items-center gap-3 border-b border-ink-700 bg-ink-900 px-3 py-1.5">
        <h1 className="text-xs font-semibold tracking-wide text-ink-100 uppercase">Settings</h1>
        <Link to="/" className="text-[11px] text-series-1 hover:underline">
          ← back to board
        </Link>
        <div className="ml-auto flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-[11px] text-ink-300">
            <input
              type="checkbox"
              checked={mock}
              disabled={mockForcedByEnv()}
              onChange={(e) => {
                setMockMode(e.target.checked);
                setMockState(e.target.checked);
                window.location.reload();
              }}
            />
            mock mode
          </label>
          {mockForcedByEnv() && <Badge tone="info">forced by VITE_MOCK</Badge>}
        </div>
      </header>

      {error && (
        <div className="border-b border-bad/40 bg-bad/10 px-3 py-1 text-[11px] text-bad">
          {error}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-2 overflow-y-auto p-2 lg:grid-cols-2">
        {/* ---------------- OpenRouter key ---------------- */}
        <Panel title="OpenRouter key" bodyClassName="p-2 space-y-2">
          <div className="flex gap-1">
            <input
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="sk-or-v1-…"
              aria-label="OpenRouter API key"
              className="num min-w-0 flex-1 rounded border border-ink-600 bg-ink-900 px-1.5 py-1 text-xs text-ink-100 placeholder:text-ink-500"
            />
            <Button variant="primary" onClick={() => void validateKey()} disabled={validating}>
              {validating ? 'Checking…' : 'Validate & store'}
            </Button>
          </div>

          {keyResult && (
            <div
              className={`rounded border p-2 ${
                keyResult.valid ? 'border-good/40 bg-good/10' : 'border-bad/40 bg-bad/10'
              }`}
            >
              <div className="flex items-center gap-1.5">
                <StatusDot tone={keyResult.valid ? 'good' : 'bad'} />
                <span className={`text-xs ${keyResult.valid ? 'text-good' : 'text-bad'}`}>
                  {keyResult.valid ? 'Key accepted' : 'Key rejected'}
                </span>
                <span className="num truncate text-[11px] text-ink-300">{keyResult.label}</span>
              </div>
              {keyResult.valid && (
                <div className="num mt-1 text-xs text-ink-200">
                  remaining credit{' '}
                  <span className="text-good">{formatMoney(keyResult.remaining_credit)}</span>
                </div>
              )}
              <div className="mt-1 text-[11px] text-ink-400">
                {keyResult.backend === 'keyring'
                  ? 'Stored in the OS keyring.'
                  : 'No OS keyring available — stored in a 0600 file under the app data directory.'}
              </div>
            </div>
          )}

          {!keyResult && (
            <p className="text-[11px] text-ink-400">
              The key is validated against OpenRouter before it is written. Storage falls back to a
              0600 file when no OS keyring is present; the result above says which one was used.
            </p>
          )}
        </Panel>

        {/* ---------------- Roles ---------------- */}
        <Panel
          title="Role assignment"
          right={
            <div className="flex items-center gap-2">
              {saved === 'saved' && <span className="text-[10px] text-good">saved</span>}
              <Button size="sm" variant="primary" onClick={() => void save()} disabled={!roles}>
                {saved === 'saving' ? 'Saving…' : 'Save roles'}
              </Button>
            </div>
          }
          bodyClassName="p-2 space-y-2"
        >
          {!roles ? (
            <p className="text-[11px] text-ink-500">Loading catalog…</p>
          ) : (
            <>
              <div>
                <div className="mb-1 flex items-center gap-2">
                  <h3 className="text-[11px] font-semibold text-ink-200">
                    Extractor jury (3 slots)
                  </h3>
                  <span className="text-[10px] text-ink-500">
                    vision required · disagreement drives the inspector
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-1.5">
                  {([0, 1, 2] as const).map((i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <label
                        className="num w-12 shrink-0 text-[10px] text-ink-500"
                        htmlFor={`jury-${i}`}
                      >
                        slot {i + 1}
                      </label>
                      <ModelSelect
                        id={`jury-${i}`}
                        value={roles.extractor_jury[i]}
                        catalog={models?.catalog ?? []}
                        onChange={(slug) => setJury(i, slug)}
                        requireVision
                      />
                      <Badge>{familyOf(roles.extractor_jury[i])}</Badge>
                    </div>
                  ))}
                </div>
                {juryFamilyWarning && (
                  <div
                    role="alert"
                    className="mt-1.5 rounded border border-warn/40 bg-warn/10 px-1.5 py-1 text-[11px] text-warn"
                  >
                    {juryFamilyWarning}
                  </div>
                )}
              </div>

              <div className="space-y-1.5 border-t border-ink-750 pt-2">
                {SINGLE_ROLES.map(({ role, label, blurb }) => (
                  <div key={role} className="flex items-center gap-1.5">
                    <label className="w-24 shrink-0 text-[10px] text-ink-400" htmlFor={role}>
                      {label}
                      <span className="block text-[9px] text-ink-500">{blurb}</span>
                    </label>
                    <ModelSelect
                      id={role}
                      value={roles[role]}
                      catalog={models?.catalog ?? []}
                      onChange={(slug) => {
                        setRoles({ ...roles, [role]: slug });
                        setSaved('idle');
                      }}
                    />
                    <Badge>{familyOf(roles[role])}</Badge>
                  </div>
                ))}
              </div>
            </>
          )}
        </Panel>

        {/* ---------------- Latency ---------------- */}
        <Panel title="Measured latency per role" bodyClassName="p-2">
          {!models ? (
            <p className="text-[11px] text-ink-500">—</p>
          ) : (
            <table className="w-full">
              <tbody>
                {Object.entries(models.latency).map(([role, ms]) => (
                  <tr key={role} className="border-t border-ink-800">
                    <td className="num py-0.5 text-[11px] text-ink-300">{role}</td>
                    <td className={`num py-0.5 text-right text-[11px] ${latencyColorClass(ms)}`}>
                      {formatMs(ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-1.5 text-[10px] text-ink-500">
            Measured on the last real call, not advertised. The extraction budget is 4,000 ms end to
            end.
          </p>
        </Panel>

        {/* ---------------- Catalog ---------------- */}
        <Panel title="Model catalog" bodyClassName="overflow-auto">
          <table className="w-full">
            <thead className="sticky top-0 bg-ink-850">
              <tr className="text-[10px] tracking-wide text-ink-400 uppercase">
                <th className="px-1.5 py-1 text-left">slug</th>
                <th className="px-1 py-1 text-left">family</th>
                <th className="px-1 py-1 text-center">vis</th>
                <th className="px-1 py-1 text-center">struct</th>
                <th className="px-1 py-1 text-center">tools</th>
                <th className="px-1 py-1 text-right">$/Mtok in</th>
                <th className="px-1.5 py-1 text-right">$/Mtok out</th>
              </tr>
            </thead>
            <tbody>
              {(models?.catalog ?? []).map((m) => (
                <tr key={m.slug} className="border-t border-ink-800">
                  <td className="num px-1.5 py-0.5 text-[11px] text-ink-100">{m.slug}</td>
                  <td className="px-1 py-0.5 text-[11px] text-ink-400">{m.family}</td>
                  <td className="px-1 py-0.5 text-center text-[11px]">
                    {m.vision ? <span className="text-good">✓</span> : <span className="text-ink-600">—</span>}
                  </td>
                  <td className="px-1 py-0.5 text-center text-[11px]">
                    {m.structured ? <span className="text-good">✓</span> : <span className="text-ink-600">—</span>}
                  </td>
                  <td className="px-1 py-0.5 text-center text-[11px]">
                    {m.tools ? <span className="text-good">✓</span> : <span className="text-ink-600">—</span>}
                  </td>
                  <td className="num px-1 py-0.5 text-right text-[11px] text-ink-200">
                    {formatMoney(m.price_in)}
                  </td>
                  <td className="num px-1.5 py-0.5 text-right text-[11px] text-ink-200">
                    {formatMoney(m.price_out)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
    </div>
  );
}
