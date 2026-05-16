import { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { CanvasDrawing } from '../logic/canvas';

interface DrawingCanvasProps {
    className?: string;
}

export interface DrawingCanvasHandle {
    clear: () => void;
    getTraces: () => number[][][];
    hasStrokes: () => boolean;
}

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(({ className }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const drawingRef = useRef<CanvasDrawing | null>(null);

    useEffect(() => {
        if (canvasRef.current) {
            drawingRef.current = new CanvasDrawing(canvasRef.current);
            
            const handleResize = () => {
                drawingRef.current?.resize();
            };
            window.addEventListener('resize', handleResize);
            return () => window.removeEventListener('resize', handleResize);
        }
    }, []);

    useImperativeHandle(ref, () => ({
        clear: () => drawingRef.current?.clear(),
        getTraces: () => drawingRef.current?.getTraces() || [],
        hasStrokes: () => drawingRef.current?.hasStrokes() || false
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
