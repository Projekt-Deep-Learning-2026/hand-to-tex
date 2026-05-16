import { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { CanvasDrawing } from '../logic/canvas';
import type { CanvasMode } from '../logic/canvas';

interface DrawingCanvasProps {
    className?: string;
    mode?: CanvasMode;
    onSelectionComplete?: (traces: number[][][]) => void;
}

export interface DrawingCanvasHandle {
    clear: () => void;
    getTraces: () => number[][][];
    hasStrokes: () => boolean;
    setMode: (mode: CanvasMode) => void;
    clearSelection: () => void;
}

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(({ 
    className, 
    mode = 'draw',
    onSelectionComplete 
}, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const drawingRef = useRef<CanvasDrawing | null>(null);

    useEffect(() => {
        if (canvasRef.current) {
            drawingRef.current = new CanvasDrawing(canvasRef.current);
            drawingRef.current.setMode(mode);
            if (onSelectionComplete) {
                drawingRef.current.setOnSelectionComplete(onSelectionComplete);
            }
            
            const handleResize = () => {
                drawingRef.current?.resize();
            };
            window.addEventListener('resize', handleResize);
            return () => window.removeEventListener('resize', handleResize);
        }
    }, []);

    useEffect(() => {
        drawingRef.current?.setMode(mode);
    }, [mode]);

    useEffect(() => {
        if (onSelectionComplete) {
            drawingRef.current?.setOnSelectionComplete(onSelectionComplete);
        }
    }, [onSelectionComplete]);

    useImperativeHandle(ref, () => ({
        clear: () => drawingRef.current?.clear(),
        getTraces: () => drawingRef.current?.getTraces() || [],
        hasStrokes: () => drawingRef.current?.hasStrokes() || false,
        setMode: (m: CanvasMode) => drawingRef.current?.setMode(m),
        clearSelection: () => drawingRef.current?.clearSelection()
    }));

    return (
        <canvas
            ref={canvasRef}
            className={className}
            style={{ touchAction: 'none' }}
            onMouseDown={(e) => drawingRef.current?.handleMouseDown(e)}
            onMouseMove={(e) => drawingRef.current?.handleMouseMove(e)}
            onMouseUp={() => drawingRef.current?.handleMouseUp()}
            onMouseLeave={() => drawingRef.current?.handleMouseUp()}
            onTouchStart={(e) => drawingRef.current?.handleTouchStart(e)}
            onTouchMove={(e) => drawingRef.current?.handleTouchMove(e)}
            onTouchEnd={() => drawingRef.current?.handleTouchEnd()}
        />
    );
});
