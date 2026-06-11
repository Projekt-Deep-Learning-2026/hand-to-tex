import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { DrawingCanvas } from './DrawingCanvas';
import React from 'react';

// Mock CanvasDrawing to prevent DOM issues during testing
const mockCanvasDrawingInstance = {
    setMode: vi.fn(),
    setPenOnlyMode: vi.fn(),
    setOnSelectionComplete: vi.fn(),
    setOnSelectionChange: vi.fn(),
    setOnObjectsChange: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
    clear: vi.fn(),
    getTraces: vi.fn().mockReturnValue([]),
    getLatexObjects: vi.fn().mockReturnValue([]),
    hasStrokes: vi.fn().mockReturnValue(false),
    clearSelection: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    canRedo: vi.fn().mockReturnValue(false),
    replaceSelectedWithLatex: vi.fn(),
    replaceAllWithLatex: vi.fn(),
    setTraces: vi.fn(),
    setLatexObjects: vi.fn(),
    setBackgroundImage: vi.fn(),
    updateLatexObject: vi.fn(),
    deleteLatexObject: vi.fn().mockReturnValue(true),
    handlePointerDown: vi.fn(),
    handlePointerMove: vi.fn(),
    handlePointerUp: vi.fn(),
};

vi.mock('../logic/canvas', () => {
    return {
        CanvasDrawing: vi.fn().mockImplementation(function() { return mockCanvasDrawingInstance; })
    };
});

// Mock katex
vi.mock('katex', () => ({
    default: {
        render: vi.fn()
    }
}));

describe('DrawingCanvas Component', () => {
    afterEach(() => {
        vi.clearAllMocks();
    });

    it('renders a canvas element', () => {
        const { container } = render(<DrawingCanvas className="test-class" />);
        const divContainer = container.querySelector('.test-class');
        expect(divContainer).toBeInTheDocument();
        
        const canvas = divContainer?.querySelector('canvas');
        expect(canvas).toBeInTheDocument();
    });

    it('initializes CanvasDrawing with correct props', () => {
        render(<DrawingCanvas mode="erase" penOnlyMode={true} />);
        
        expect(mockCanvasDrawingInstance.setMode).toHaveBeenCalledWith('erase');
        expect(mockCanvasDrawingInstance.setPenOnlyMode).toHaveBeenCalledWith(true);
    });

    it('updates mode and penOnlyMode when props change', () => {
        const { rerender } = render(<DrawingCanvas mode="draw" penOnlyMode={false} />);
        
        expect(mockCanvasDrawingInstance.setMode).toHaveBeenCalledWith('draw');
        expect(mockCanvasDrawingInstance.setPenOnlyMode).toHaveBeenCalledWith(false);

        rerender(<DrawingCanvas mode="select" penOnlyMode={true} />);
        
        expect(mockCanvasDrawingInstance.setMode).toHaveBeenCalledWith('select');
        expect(mockCanvasDrawingInstance.setPenOnlyMode).toHaveBeenCalledWith(true);
    });

    it('exposes methods via ref', () => {
        const ref = React.createRef<any>();
        render(<DrawingCanvas ref={ref} />);
        
        expect(ref.current).toBeDefined();
        ref.current.clear();
        expect(mockCanvasDrawingInstance.clear).toHaveBeenCalledTimes(1);

        ref.current.setMode('erase');
        expect(mockCanvasDrawingInstance.setMode).toHaveBeenCalledWith('erase');
        
        ref.current.getTraces();
        expect(mockCanvasDrawingInstance.getTraces).toHaveBeenCalled();
    });
});
