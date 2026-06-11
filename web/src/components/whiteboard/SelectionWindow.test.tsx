import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { SelectionWindow } from './SelectionWindow';

// Mock katex
vi.mock('katex', () => ({
    default: {
        render: vi.fn()
    }
}));

describe('SelectionWindow Component', () => {
    const mockOnReplace = vi.fn();
    const mockOnClose = vi.fn();

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('renders the header and close button', () => {
        render(
            <SelectionWindow 
                latex={null} 
                isProcessing={false} 
                isModelReady={true} 
                onReplace={mockOnReplace} 
                onClose={mockOnClose} 
                numSelectedTraces={0} 
            />
        );
        
        expect(screen.getByText('Selection Recognition')).toBeInTheDocument();
        
        const closeBtn = screen.getByRole('button', { name: '×' });
        fireEvent.click(closeBtn);
        expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('renders processing state', () => {
        render(
            <SelectionWindow 
                latex={null} 
                isProcessing={true} 
                isModelReady={true} 
                onReplace={mockOnReplace} 
                onClose={mockOnClose} 
                numSelectedTraces={5} 
            />
        );
        
        expect(screen.getByText('Recognizing 5 traces...')).toBeInTheDocument();
        expect(screen.queryByText('Replace Selected Traces')).not.toBeInTheDocument();
    });

    it('renders latex result and replace button', () => {
        render(
            <SelectionWindow 
                latex="\\int x dx" 
                isProcessing={false} 
                isModelReady={true} 
                onReplace={mockOnReplace} 
                onClose={mockOnClose} 
                numSelectedTraces={5} 
            />
        );
        
        expect(screen.getByText('\\\\int x dx')).toBeInTheDocument();
        
        const replaceBtn = screen.getByRole('button', { name: 'Replace Selected Traces' });
        expect(replaceBtn).toBeEnabled();
        
        fireEvent.click(replaceBtn);
        expect(mockOnReplace).toHaveBeenCalledTimes(1);
    });

    it('disables replace button if model is not ready or latex is error', () => {
        const { rerender } = render(
            <SelectionWindow 
                latex="\\int x dx" 
                isProcessing={false} 
                isModelReady={false} 
                onReplace={mockOnReplace} 
                onClose={mockOnClose} 
                numSelectedTraces={5} 
            />
        );
        
        expect(screen.getByRole('button', { name: 'Replace Selected Traces' })).toBeDisabled();

        rerender(
            <SelectionWindow 
                latex="Error" 
                isProcessing={false} 
                isModelReady={true} 
                onReplace={mockOnReplace} 
                onClose={mockOnClose} 
                numSelectedTraces={5} 
            />
        );

        expect(screen.getByRole('button', { name: 'Replace Selected Traces' })).toBeDisabled();
    });

    it('renders empty selection state', () => {
        render(
            <SelectionWindow 
                latex={null} 
                isProcessing={false} 
                isModelReady={true} 
                onReplace={mockOnReplace} 
                onClose={mockOnClose} 
                numSelectedTraces={2} 
            />
        );
        
        expect(screen.getByText('2 traces selected. Release to recognize.')).toBeInTheDocument();
    });
});
