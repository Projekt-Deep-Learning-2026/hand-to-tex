import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { WhiteboardView } from './WhiteboardView';

// Mock sub-components
vi.mock('./DrawingCanvas', async (importOriginal) => {
    const actual = await importOriginal() as any;
    const React = await import('react');
    return {
        ...actual,
        DrawingCanvas: React.forwardRef((props: any, ref: any) => {
            React.useImperativeHandle(ref, () => ({
                undo: vi.fn(),
                redo: vi.fn(),
                clear: vi.fn(),
                setTraces: vi.fn(),
                setLatexObjects: vi.fn(),
                updateLatexObject: vi.fn(),
                clearSelection: vi.fn()
            }));
            return <div data-testid="drawing-canvas" className={props.className} />;
        })
    };
});

vi.mock('./Toolbar', () => ({
    Toolbar: (props: any) => (
        <div data-testid="toolbar">
            <button onClick={() => props.onChangeMode('select')}>Select</button>
            <button onClick={props.onUndo}>Undo</button>
            <button onClick={props.onClear}>Clear</button>
        </div>
    )
}));

vi.mock('./SelectionWindow', () => ({
    SelectionWindow: (props: any) => (
        <div data-testid="selection-window">
            <span>{props.latex}</span>
            <button onClick={props.onReplace}>Replace</button>
            <button onClick={props.onClose}>Close</button>
        </div>
    )
}));

vi.mock('../editor/EditModal', () => ({
    EditModal: (props: any) => (
        <div data-testid="edit-modal">
            <button onClick={() => props.onSave('new latex')}>Save</button>
        </div>
    )
}));

// Mock hooks
vi.mock('../../hooks/useProjectIO', () => ({
    useProjectIO: () => ({
        handleSaveToFile: vi.fn(),
        handleLoadProject: vi.fn(),
        handleExportPDF: vi.fn(),
        isExporting: false
    })
}));

describe('WhiteboardView Component', () => {
    const defaultProps = {
        initialProjectData: null,
        onClearInitialData: vi.fn(),
        onNavigateHome: vi.fn(),
        onToast: vi.fn(),
        modelStatus: 'success',
        modelProgress: 'Ready',
        vocab: { PAD_IDX: 0, SOS_IDX: 1, id2token: ['pad', 'sos', 'x', '+'] },
        recognize: vi.fn().mockResolvedValue([2, 3, 2]) // x + x
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders the header and canvas', () => {
        render(<WhiteboardView {...defaultProps} />);
        
        expect(screen.getByText('Whiteboard')).toBeInTheDocument();
        expect(screen.getByTestId('drawing-canvas')).toBeInTheDocument();
        expect(screen.getByTestId('toolbar')).toBeInTheDocument();
    });

    it('navigates home when back button is clicked', () => {
        render(<WhiteboardView {...defaultProps} />);
        
        const backBtn = document.querySelector('.back-button');
        if (backBtn) fireEvent.click(backBtn);
        
        expect(defaultProps.onNavigateHome).toHaveBeenCalledTimes(1);
    });

    it('changes canvas mode and shows toast', () => {
        render(<WhiteboardView {...defaultProps} />);
        
        fireEvent.click(screen.getByText('Select'));
        
        expect(defaultProps.onToast).toHaveBeenCalledWith(expect.objectContaining({
            message: 'Select Mode Active'
        }));
    });

    it('clears initial data if provided', async () => {
        const initialData = { traces: [[[1, 2, 3]]], latexObjects: [] };
        render(<WhiteboardView {...defaultProps} initialProjectData={initialData} />);
        
        await waitFor(() => {
            expect(defaultProps.onClearInitialData).toHaveBeenCalledTimes(1);
        });
    });

    it('handles clear action', () => {
        render(<WhiteboardView {...defaultProps} />);
        
        fireEvent.click(screen.getByText('Clear'));
        
        expect(defaultProps.onToast).toHaveBeenCalledWith(expect.objectContaining({
            message: 'Whiteboard cleared'
        }));
    });
});
