import { Suspense, lazy } from 'react';
import { Hud } from './components/Hud';
import { useRoute } from './router';

// The HUD is always-on-top on monitor 1 and must start instantly, so the heavy
// monitor-2 views (recharts lives in Dashboard) are split out of its payload.
const Dashboard = lazy(() =>
  import('./components/Dashboard').then((m) => ({ default: m.Dashboard })),
);
const SettingsView = lazy(() =>
  import('./components/SettingsView').then((m) => ({ default: m.SettingsView })),
);

function Loading() {
  return (
    <div className="flex h-full items-center justify-center bg-ink-950 text-[11px] text-ink-500">
      loading view…
    </div>
  );
}

export function App() {
  const route = useRoute();

  if (route === '/hud') return <Hud />;

  return (
    <Suspense fallback={<Loading />}>
      {route === '/settings' ? <SettingsView /> : <Dashboard />}
    </Suspense>
  );
}
