import { useRef, useEffect, useImperativeHandle, forwardRef, useState } from 'react';
import { CanvasDrawing } from '../logic/canvas';
import type { CanvasMode, LatexObject } from '../logic/canvas';
import katex from 'katex';

interface DrawingCanvasProps {
    className?: string;
    mode?: CanvasMode;
    penOnlyMode?: boolean;
    onSelectionComplete?: (traces: number[][][]) => void;
    onSelectionChange?: (count: number) => void;
}

export interface DrawingCanvasHandle {
    clear: () => void;
    getTraces: () => number[][][];
    getLatexObjects: () => LatexObject[];
    hasStrokes: () => boolean;
    setMode: (mode: CanvasMode) => void;
    setPenOnlyMode: (enabled: boolean) => void;
    clearSelection: () => void;
    undo: () => void;
    redo: () => void;
    canRedo: () => boolean;
    replaceSelectedWithLatex: (latex: string) => void;
    replaceAllWithLatex: (latex: string) => void;
    setTraces: (traces: number[][][]) => void;
    setLatexObjects: (objects: LatexObject[]) => void;
    setBackgroundImage: (image: HTMLImageElement | null) => void;
}

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(({ 
    className, 
    mode = 'draw',
    penOnlyMode = false,
    onSelectionComplete,
    onSelectionChange
}, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const drawingRef = useRef<CanvasDrawing | null>(null);
    const [latexObjects, setLatexObjects] = useState<LatexObject[]>([]);

    useEffect(() => {
        if (canvasRef.current) {
            drawingRef.current = new CanvasDrawing(canvasRef.current);
            drawingRef.current.setMode(mode);
            drawingRef.current.setPenOnlyMode(penOnlyMode);
            if (onSelectionComplete) drawingRef.current.setOnSelectionComplete(onSelectionComplete);
            if (onSelectionChange) drawingRef.current.setOnSelectionChange(onSelectionChange);
            
            const handleResize = () => drawingRef.current?.resize();
            window.addEventListener('resize', handleResize);
            return () => window.removeEventListener('resize', handleResize);
        }
        // These are intentionally empty to run once on mount. 
        // Subsequent changes are handled by other effects.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        drawingRef.current?.setMode(mode);
    }, [mode]);

    useEffect(() => {
        drawingRef.current?.setPenOnlyMode(penOnlyMode);
    }, [penOnlyMode]);

    useEffect(() => {
        if (onSelectionComplete) drawingRef.current?.setOnSelectionComplete(onSelectionComplete);
    }, [onSelectionComplete]);

    useEffect(() => {
        if (onSelectionChange) drawingRef.current?.setOnSelectionChange(onSelectionChange);
    }, [onSelectionChange]);

    const syncLatexObjects = () => {
        if (drawingRef.current) {
            setLatexObjects([...drawingRef.current.getLatexObjects()]);
        }
    };

    useImperativeHandle(ref, () => ({
        clear: () => {
            drawingRef.current?.clear();
            setLatexObjects([]);
        },
        getTraces: () => drawingRef.current?.getTraces() || [],
        getLatexObjects: () => drawingRef.current?.getLatexObjects() || [],
        hasStrokes: () => drawingRef.current?.hasStrokes() || false,
        setMode: (m: CanvasMode) => drawingRef.current?.setMode(m),
        setPenOnlyMode: (e: boolean) => drawingRef.current?.setPenOnlyMode(e),
        clearSelection: () => drawingRef.current?.clearSelection(),
        undo: () => drawingRef.current?.undo(),
        redo: () => drawingRef.current?.redo(),
        canRedo: () => drawingRef.current?.canRedo() || false,
        replaceSelectedWithLatex: (latex: string) => {
            drawingRef.current?.replaceSelectedWithLatex(latex);
            syncLatexObjects();
        },
        replaceAllWithLatex: (latex: string) => {
            drawingRef.current?.replaceAllWithLatex(latex);
            syncLatexObjects();
        },
        setTraces: (t: number[][][]) => drawingRef.current?.setTraces(t),
        setLatexObjects: (o: LatexObject[]) => {
            drawingRef.current?.setLatexObjects(o);
            setLatexObjects(o);
        },
        setBackgroundImage: (i: HTMLImageElement | null) => drawingRef.current?.setBackgroundImage(i)
    }));

    return (
        <div className={className} style={{ position: 'relative', overflow: 'hidden' }}>
            <canvas
                ref={canvasRef}
                draggable={false}
                onContextMenu={(e) => e.preventDefault()}
                style={{ touchAction: 'none', width: '100%', height: '100%', display: 'block' }}
                onPointerDown={(e) => {
                    drawingRef.current?.handlePointerDown(e.nativeEvent);
                    syncLatexObjects();
                }}
                onPointerMove={(e) => {
                    drawingRef.current?.handlePointerMove(e.nativeEvent);
                    if (mode === 'pointer') syncLatexObjects();
                }}
                onPointerUp={() => {
                    drawingRef.current?.handlePointerUp();
                    syncLatexObjects();
                }}
                onPointerLeave={() => {
                    drawingRef.current?.handlePointerUp();
                    syncLatexObjects();
                }}
            />
            {latexObjects.map(obj => (
                <div 
                    key={obj.id}
                    style={{
                        position: 'absolute',
                        left: obj.x,
                        top: obj.y,
                        width: obj.width,
                        height: obj.height,
                        pointerEvents: 'none',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        userSelect: 'none',
                        color: 'black'
                    }}
                    ref={(el) => {
                        if (el) {
                            katex.render(obj.latex, el, { 
                                throwOnError: false,
                                displayMode: true
                            });
                            const mathEl = el.querySelector('.katex-display');
                            if (mathEl) {
                                (mathEl as HTMLElement).style.margin = '0';
                                (mathEl as HTMLElement).style.color = 'black';
                                (mathEl as HTMLElement).style.fontSize = `${Math.min(obj.width, obj.height) * 0.8}px`;
                            }
                        }
                    }}
                />
            ))}
        </div>
    );
});
