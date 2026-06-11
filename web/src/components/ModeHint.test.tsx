import { render, screen, act } from '@testing-library/react';
import { vi } from 'vitest';
import { ModeHint } from './ModeHint';

describe('ModeHint Component', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders the correct icon and message', () => {
        render(<ModeHint mode="draw" icon="✏️" message="Drawing Mode" />);
        
        expect(screen.getByText('✏️')).toBeInTheDocument();
        expect(screen.getByText('Drawing Mode')).toBeInTheDocument();
    });

    it('starts hidden and becomes visible', () => {
        const { container } = render(<ModeHint mode="draw" icon="✏️" message="Drawing Mode" />);
        
        const hintElement = container.firstChild as HTMLElement;
        expect(hintElement).toHaveClass('hidden');

        act(() => {
            vi.advanceTimersByTime(10);
        });

        expect(hintElement).toHaveClass('visible');
    });

    it('becomes hidden after 1500ms and unmounts after 2000ms', () => {
        const { container } = render(<ModeHint mode="draw" icon="✏️" message="Drawing Mode" />);
        
        act(() => {
            vi.advanceTimersByTime(10);
        });
        
        const hintElement = container.firstChild as HTMLElement;
        expect(hintElement).toHaveClass('visible');

        act(() => {
            vi.advanceTimersByTime(1490); // 1500ms total
        });

        expect(hintElement).toHaveClass('hidden');
        expect(document.body.contains(hintElement)).toBe(true);

        act(() => {
            vi.advanceTimersByTime(500); // 2000ms total
        });

        expect(document.body.contains(hintElement)).toBe(false);
    });
});
