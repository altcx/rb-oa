import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StateInspector } from '../components/StateInspector';
import { useStore } from '../state/store';
import { initialAppState } from '../state/reducer';
import { mockCaptures, mockState } from '../mock/fixtures';

function seed() {
  useStore.setState({ ...initialAppState });
  const { dispatch } = useStore.getState();
  dispatch({ kind: 'hydrate/captures', captures: mockCaptures });
  dispatch({ kind: 'hydrate/state', payload: mockState });
}

const counter = () => screen.getByTestId('review-counter').textContent;
const rows = () => screen.getAllByRole('option');

beforeEach(() => {
  seed();
});

describe('disputed-field keyboard flow', () => {
  it('shows the live N/M/K counter', () => {
    render(<StateInspector />);
    expect(counter()).toBe('40 fields, 3 disputed, 3 left to review');
  });

  it('puts disputed fields at the top, hardest first, with their source crop', () => {
    render(<StateInspector />);
    const list = rows();
    expect(list).toHaveLength(3);
    expect(list[0]).toHaveAttribute('data-status', 'split');
    expect(list[1]).toHaveAttribute('data-status', 'majority');
    expect(screen.getAllByTestId('crop-zoom')).toHaveLength(3);
  });

  it('Tab moves forward between disputed fields and wraps', () => {
    render(<StateInspector />);
    const list = rows();
    list[0]!.focus();

    fireEvent.keyDown(list[0]!, { key: 'Tab' });
    expect(rows()[1]).toHaveFocus();
    expect(rows()[1]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(rows()[1]!, { key: 'Tab' });
    expect(rows()[2]).toHaveFocus();

    fireEvent.keyDown(rows()[2]!, { key: 'Tab' });
    expect(rows()[0]).toHaveFocus();
  });

  it('Shift+Tab moves backward', () => {
    render(<StateInspector />);
    const list = rows();
    list[0]!.focus();
    fireEvent.keyDown(list[0]!, { key: 'Tab', shiftKey: true });
    expect(rows()[2]).toHaveFocus();
  });

  it('Enter accepts the shown value and removes the field from review', () => {
    const onConfirm = vi.fn();
    render(<StateInspector onConfirm={onConfirm} />);
    const list = rows();
    list[0]!.focus();

    // Move to the first majority field, then accept it.
    fireEvent.keyDown(list[0]!, { key: 'Tab' });
    const target = rows()[1]!;
    const path = target.getAttribute('data-path');
    fireEvent.keyDown(target, { key: 'Enter' });

    expect(counter()).toBe('40 fields, 3 disputed, 2 left to review');
    expect(rows()).toHaveLength(2);
    expect(screen.queryByText(path!)).not.toBeInTheDocument();
    expect(onConfirm).toHaveBeenCalledWith({ 'machines[2].throughput_per_hour': 118 });
  });

  it('A accepts every majority value at once and leaves the split for a human', () => {
    const onConfirm = vi.fn();
    render(<StateInspector onConfirm={onConfirm} />);

    fireEvent.keyDown(window, { key: 'a' });

    expect(counter()).toBe('40 fields, 3 disputed, 1 left to review');
    const remaining = rows();
    expect(remaining).toHaveLength(1);
    expect(remaining[0]).toHaveAttribute('data-status', 'split');
    expect(onConfirm).toHaveBeenCalledWith({
      'machines[2].throughput_per_hour': 118,
      'market.demand_curve_slope': -0.42,
    });
  });

  it('A is inert while typing in a text field', async () => {
    const user = userEvent.setup();
    render(
      <>
        <input aria-label="chat" />
        <StateInspector />
      </>,
    );
    await user.click(screen.getByLabelText('chat'));
    await user.keyboard('a');
    expect(counter()).toBe('40 fields, 3 disputed, 3 left to review');
  });

  it('a number key accepts the nth alternative instead of the majority', () => {
    const onConfirm = vi.fn();
    render(<StateInspector onConfirm={onConfirm} />);
    const first = rows()[0]!;
    first.focus();
    fireEvent.keyDown(first, { key: '1' });
    // The split field's first alternative is 1.25, not the leading 12.5.
    expect(onConfirm).toHaveBeenCalledWith({ 'contracts[1].penalty_per_late_unit': 1.25 });
    expect(counter()).toBe('40 fields, 3 disputed, 2 left to review');
  });

  it('verifies the whole board without the mouse', () => {
    render(<StateInspector />);
    fireEvent.keyDown(window, { key: 'A' });
    const last = rows()[0]!;
    last.focus();
    fireEvent.keyDown(last, { key: 'Enter' });

    expect(counter()).toBe('40 fields, 3 disputed, 0 left to review');
    expect(screen.queryAllByRole('option')).toHaveLength(0);
    expect(screen.getByText(/Board verified/)).toBeInTheDocument();
  });
});

describe('agreed-fields collapse', () => {
  it('collapses every unanimous field into one row', () => {
    render(<StateInspector />);
    const toggle = screen.getByRole('button', { name: /37 fields agreed/ });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId('agreed-list')).not.toBeInTheDocument();
    expect(screen.queryByText('cash.on_hand')).not.toBeInTheDocument();
  });

  it('expands on click and lists the fields', async () => {
    const user = userEvent.setup();
    render(<StateInspector />);
    await user.click(screen.getByRole('button', { name: /37 fields agreed/ }));

    const list = screen.getByTestId('agreed-list');
    expect(within(list).getByText('cash.on_hand')).toBeInTheDocument();
    expect(within(list).getAllByRole('listitem')).toHaveLength(37);
    expect(screen.getByRole('button', { name: /37 fields agreed/ })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
  });

  it('grows as disputes are settled', async () => {
    const user = userEvent.setup();
    render(<StateInspector />);
    fireEvent.keyDown(window, { key: 'a' });

    const toggle = screen.getByRole('button', { name: /39 fields agreed/ });
    await user.click(toggle);
    const list = screen.getByTestId('agreed-list');
    expect(within(list).getAllByRole('listitem')).toHaveLength(39);
    // A settled dispute is marked so it is not mistaken for a jury agreement.
    expect(within(list).getAllByText('confirmed')).toHaveLength(2);
  });

  it('collapses again on a second click', async () => {
    const user = userEvent.setup();
    render(<StateInspector />);
    const toggle = screen.getByRole('button', { name: /37 fields agreed/ });
    await user.click(toggle);
    await user.click(screen.getByRole('button', { name: /37 fields agreed/ }));
    expect(screen.queryByTestId('agreed-list')).not.toBeInTheDocument();
  });
});
