import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { Toolbar } from './Toolbar';

describe('Toolbar Component', () => {
    const defaultProps = {
        canvasMode: 'draw' as const,
        penOnlyMode: false,
        onChangeMode: vi.fn(),
        onTogglePenOnly: vi.fn(),
        onUndo: vi.fn(),
        onRedo: vi.fn(),
        onClear: vi.fn(),
        mini: false
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders all tools with correct initial active state', () => {
        render(<Toolbar {...defaultProps} />);
        
        const drawBtn = screen.getByTitle('Draw (Pencil)');
        expect(drawBtn).toHaveClass('active');
        
        const selectBtn = screen.getByTitle('Select (Lasso)');
        expect(selectBtn).not.toHaveClass('active');
    });

    it('calls onChangeMode when mode buttons are clicked', () => {
        render(<Toolbar {...defaultProps} />);
        
        fireEvent.click(screen.getByTitle('Select (Lasso)'));
        expect(defaultProps.onChangeMode).toHaveBeenCalledWith('select');
        
        fireEvent.click(screen.getByTitle('Erase'));
        expect(defaultProps.onChangeMode).toHaveBeenCalledWith('erase');
        
        fireEvent.click(screen.getByTitle('Pointer (Move/Resize)'));
        expect(defaultProps.onChangeMode).toHaveBeenCalledWith('pointer');
    });

    it('calls onTogglePenOnly when pen only button is clicked', () => {
        render(<Toolbar {...defaultProps} />);
        
        fireEvent.click(screen.getByTitle('Lock to Pen only (for mobile)'));
        expect(defaultProps.onTogglePenOnly).toHaveBeenCalledTimes(1);
    });

    it('calls action callbacks correctly (undo, redo, clear)', () => {
        render(<Toolbar {...defaultProps} />);
        
        fireEvent.click(screen.getByTitle('Undo (Ctrl + Z)'));
        expect(defaultProps.onUndo).toHaveBeenCalledTimes(1);
        
        fireEvent.click(screen.getByTitle('Redo (Ctrl + Y)'));
        expect(defaultProps.onRedo).toHaveBeenCalledTimes(1);
        
        fireEvent.click(screen.getByTitle('Clear Canvas'));
        expect(defaultProps.onClear).toHaveBeenCalledTimes(1);
    });

    it('renders in mini mode without spans', () => {
        render(<Toolbar {...defaultProps} mini={true} />);
        
        // Ensure no <span> elements like <span>Draw</span> are rendered
        expect(screen.queryByText('Draw')).not.toBeInTheDocument();
        expect(screen.queryByText('Select')).not.toBeInTheDocument();
        expect(screen.queryByText('Erase')).not.toBeInTheDocument();
        
        // The container should have the mini class
        const container = screen.getByTitle('Draw (Pencil)').closest('.mode-toggle');
        expect(container).toHaveClass('mini');
    });
});
