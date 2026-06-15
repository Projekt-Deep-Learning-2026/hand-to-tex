import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { TutorialModal } from './TutorialModal';

describe('TutorialModal', () => {
    it('renders the tutorial content correctly', () => {
        render(<TutorialModal onClose={() => {}} />);
        expect(screen.getByText('How to use Hand-to-TeX')).toBeInTheDocument();
        expect(screen.getByText(/Use the Pencil tool/i)).toBeInTheDocument();
    });

    it('calls onClose when clicking the close button', () => {
        const handleClose = vi.fn();
        render(<TutorialModal onClose={handleClose} />);
        
        fireEvent.click(screen.getByRole('button', { name: 'Close' }));
        expect(handleClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when clicking the Get Started button', () => {
        const handleClose = vi.fn();
        render(<TutorialModal onClose={handleClose} />);
        
        fireEvent.click(screen.getByText('Get Started'));
        expect(handleClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when clicking the overlay', () => {
        const handleClose = vi.fn();
        render(<TutorialModal onClose={handleClose} />);
        
        fireEvent.click(screen.getByTestId('tutorial-overlay'));
        expect(handleClose).toHaveBeenCalledTimes(1);
    });
});
