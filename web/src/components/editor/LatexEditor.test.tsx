import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { LatexEditor } from './LatexEditor';

// Mock katex
vi.mock('katex', () => ({
    default: {
        render: vi.fn()
    }
}));

describe('LatexEditor Component', () => {
    const mockOnChange = vi.fn();

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('renders the preview section and textarea', () => {
        render(<LatexEditor latex="a^2 + b^2 = c^2" onChange={mockOnChange} />);
        
        expect(screen.getByText('Preview')).toBeInTheDocument();
        expect(screen.getByText('LaTeX Code')).toBeInTheDocument();
        
        const textarea = screen.getByPlaceholderText('Enter LaTeX code here...') as HTMLTextAreaElement;
        expect(textarea.value).toBe('a^2 + b^2 = c^2');
    });

    it('calls onChange when the text is edited', () => {
        render(<LatexEditor latex="a^2" onChange={mockOnChange} />);
        
        const textarea = screen.getByPlaceholderText('Enter LaTeX code here...');
        fireEvent.change(textarea, { target: { value: 'a^2 + b^2' } });
        
        expect(mockOnChange).toHaveBeenCalledWith('a^2 + b^2');
    });
});
