import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { EditModal } from './EditModal';

// Mock katex to avoid actual rendering during tests
vi.mock('katex', () => ({
    default: {
        render: vi.fn()
    }
}));

describe('EditModal Component', () => {
    const mockOnSave = vi.fn();
    const mockOnCancel = vi.fn();

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('renders correctly with initial latex', () => {
        render(<EditModal initialLatex="a^2 + b^2 = c^2" onSave={mockOnSave} onCancel={mockOnCancel} />);
        
        expect(screen.getByText('Edit LaTeX')).toBeInTheDocument();
        
        const textarea = screen.getByPlaceholderText('Enter LaTeX code here...') as HTMLTextAreaElement;
        expect(textarea.value).toBe('a^2 + b^2 = c^2');
    });

    it('updates latex when typed', () => {
        render(<EditModal initialLatex="a^2" onSave={mockOnSave} onCancel={mockOnCancel} />);
        
        const textarea = screen.getByPlaceholderText('Enter LaTeX code here...');
        fireEvent.change(textarea, { target: { value: 'a^2 + b^2' } });
        
        expect((textarea as HTMLTextAreaElement).value).toBe('a^2 + b^2');
    });

    it('calls onSave with current latex', () => {
        render(<EditModal initialLatex="x = y" onSave={mockOnSave} onCancel={mockOnCancel} />);
        
        const saveBtn = screen.getByRole('button', { name: 'Save Changes' });
        fireEvent.click(saveBtn);
        
        expect(mockOnSave).toHaveBeenCalledWith('x = y');
        
        // Update and save
        const textarea = screen.getByPlaceholderText('Enter LaTeX code here...');
        fireEvent.change(textarea, { target: { value: 'x = y + 1' } });
        fireEvent.click(saveBtn);
        
        expect(mockOnSave).toHaveBeenCalledWith('x = y + 1');
    });

    it('calls onCancel when cancel button or close button is clicked', () => {
        render(<EditModal initialLatex="x = y" onSave={mockOnSave} onCancel={mockOnCancel} />);
        
        const cancelBtn = screen.getByRole('button', { name: 'Cancel' });
        fireEvent.click(cancelBtn);
        expect(mockOnCancel).toHaveBeenCalledTimes(1);

        const closeBtn = screen.getByRole('button', { name: '×' });
        fireEvent.click(closeBtn);
        expect(mockOnCancel).toHaveBeenCalledTimes(2);
    });

    it('calls onCancel when clicking outside modal content', () => {
        render(<EditModal initialLatex="x = y" onSave={mockOnSave} onCancel={mockOnCancel} />);
        
        // The overlay is the first element with onClick
        const overlay = document.querySelector('.modal-overlay');
        expect(overlay).not.toBeNull();
        if (overlay) {
            fireEvent.click(overlay);
        }
        
        expect(mockOnCancel).toHaveBeenCalledTimes(1);
    });

    it('does not call onCancel when clicking inside modal content', () => {
        render(<EditModal initialLatex="x = y" onSave={mockOnSave} onCancel={mockOnCancel} />);
        
        const content = document.querySelector('.modal-content');
        expect(content).not.toBeNull();
        if (content) {
            fireEvent.click(content);
        }
        
        expect(mockOnCancel).not.toHaveBeenCalled();
    });
});
