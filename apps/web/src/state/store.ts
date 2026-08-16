import { create } from 'zustand';
import { appReducer, initialAppState, type Action, type AppState } from './reducer';

export interface Store extends AppState {
  dispatch: (action: Action) => void;
}

/**
 * One flat store driven by the pure reducer in `reducer.ts`. Components read the
 * whole state (`useStore()`) and derive with `useMemo`; zustand v5 compares
 * selector results with Object.is, so selectors that build new arrays belong in
 * the component, not here.
 */
export const useStore = create<Store>((set) => ({
  ...initialAppState,
  dispatch: (action) => set((state) => appReducer(state, action)),
}));

export const dispatch = (action: Action): void => useStore.getState().dispatch(action);
