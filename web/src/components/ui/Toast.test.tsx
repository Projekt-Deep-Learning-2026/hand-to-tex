import { render, screen, act } from '@testing-library/react';
import { vi } from 'vitest';
import { Toast } from './Toast';

describe('Toast Component', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders the message and default icon correctly', () => {
        const handleClose = vi.fn();
        render(<Toast message="Test Message" onClose={handleClose} />);

        expect(screen.getByText('Test Message')).toBeInTheDocument();
        expect(screen.getByText('✨')).toBeInTheDocument();
    });

    it('renders a custom icon when provided', () => {
        const handleClose = vi.fn();
        render(<Toast message="Success" icon="✅" onClose={handleClose} />);

        expect(screen.getByText('✅')).toBeInTheDocument();
    });

    it('starts hidden and becomes visible after mount', () => {
        const handleClose = vi.fn();
        const { container } = render(<Toast message="Test Message" onClose={handleClose} />);

        const toastElement = container.firstChild as HTMLElement;
        expect(toastElement).toHaveClass('hidden');
        expect(toastElement).not.toHaveClass('visible');

        act(() => {
            vi.advanceTimersByTime(10);
        });

        expect(toastElement).toHaveClass('visible');
        expect(toastElement).not.toHaveClass('hidden');
    });

    it('fades out and calls onClose after duration', () => {
        const handleClose = vi.fn();
        const { container } = render(<Toast message="Test Message" duration={1000} onClose={handleClose} />);
        
        const toastElement = container.firstChild as HTMLElement;
        
        act(() => {
            vi.advanceTimersByTime(10);
        });
        expect(toastElement).toHaveClass('visible');

        act(() => {
            vi.advanceTimersByTime(1000);
        });

        expect(toastElement).toHaveClass('hidden');
        expect(handleClose).not.toHaveBeenCalled();

        act(() => {
            vi.advanceTimersByTime(300);
        });

        expect(handleClose).toHaveBeenCalledTimes(1);
    });
});
