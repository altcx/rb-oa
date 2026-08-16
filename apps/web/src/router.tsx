import { useEffect, useState, type ReactNode } from 'react';

/**
 * A ~50 line router. Two reasons not to pull in react-router here: the app has
 * three routes, and the HUD window may be opened as a `file://`-ish or
 * hash-addressed URL by the desktop shell, so both `/hud` and `/#/hud` have to
 * resolve to the same view without server rewrite rules.
 */

export type Route = '/' | '/hud' | '/settings';

const ROUTES: Route[] = ['/', '/hud', '/settings'];

function normalize(raw: string): Route {
  const path = ('/' + raw.replace(/^[/#]+/, '').replace(/\/+$/, '')) as string;
  const found = ROUTES.find((r) => r === path);
  return found ?? '/';
}

export function currentRoute(): Route {
  const hash = window.location.hash;
  if (hash.startsWith('#/')) return normalize(hash.slice(1));
  return normalize(window.location.pathname);
}

export function navigate(route: Route): void {
  if (window.location.hash.startsWith('#/')) {
    window.location.hash = `#${route}`;
    return;
  }
  window.history.pushState({}, '', route);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => currentRoute());
  useEffect(() => {
    const update = () => setRoute(currentRoute());
    window.addEventListener('popstate', update);
    window.addEventListener('hashchange', update);
    return () => {
      window.removeEventListener('popstate', update);
      window.removeEventListener('hashchange', update);
    };
  }, []);
  return route;
}

export function Link({
  to,
  children,
  className,
}: {
  to: Route;
  children: ReactNode;
  className?: string;
}) {
  return (
    <a
      href={to}
      className={className}
      onClick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
        e.preventDefault();
        navigate(to);
      }}
    >
      {children}
    </a>
  );
}
