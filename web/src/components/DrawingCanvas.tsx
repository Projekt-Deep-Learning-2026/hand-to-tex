import { useRef, useEffect, useImperativeHandle, forwardRef, useState } from 'react';
import { CanvasDrawing } from '../logic/canvas';
import type { CanvasMode, LatexObject } from '../logic/canvas';
import katex from 'katex';

interface DrawingCanvasProps {
    className?: string;
    mode?: CanvasMode;
    penOnlyMode?: boolean;
    defaultFontSize?: number;
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
    updateLatexObject: (id: string, latex: string) => void;
}

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(({ 
    className, 
    mode = 'draw',
    penOnlyMode = false,
    defaultFontSize = 1.0,
    onSelectionComplete,
    onSelectionChange
}, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const drawingRef = useRef<CanvasDrawing | null>(null);
    const [latexObjects, setLatexObjects] = useState<LatexObject[]>([]);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editValue, setEditValue] = useState("");

    useEffect(() => {
        if (canvasRef.current) {
            drawingRef.current = new CanvasDrawing(canvasRef.current);
            drawingRef.current.setMode(mode);
            drawingRef.current.setPenOnlyMode(penOnlyMode);
            if (onSelectionComplete) drawingRef.current.setOnSelectionComplete(onSelectionComplete);
            if (onSelectionChange) drawingRef.current.setOnSelectionChange(onSelectionChange);
            
            drawingRef.current.setOnObjectsChange((objects) => {
                setLatexObjects([...objects]);
            });

            const handleResize = () => drawingRef.current?.resize();
            window.addEventListener('resize', handleResize);
            
            return () => {
                window.removeEventListener('resize', handleResize);
                drawingRef.current?.dispose(); // Clean up canvas resources
                drawingRef.current = null;
            };
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleCopy = (latex: string) => {
        navigator.clipboard.writeText(latex).then(() => {
            alert("Copied to clipboard!");
        });
    };

    const startEditing = (obj: LatexObject) => {
        setEditingId(obj.id);
        setEditValue(obj.latex);
    };

    const saveEdit = () => {
        if (editingId && drawingRef.current) {
            drawingRef.current.updateLatexObject(editingId, editValue);
            setEditingId(null);
        }
    };

    useImperativeHandle(ref, () => ({
        clear: () => {
            drawingRef.current?.clear();
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
        },
        replaceAllWithLatex: (latex: string) => {
            drawingRef.current?.replaceAllWithLatex(latex);
        },
        setTraces: (t: number[][][]) => drawingRef.current?.setTraces(t),
        setLatexObjects: (o: LatexObject[]) => {
            drawingRef.current?.setLatexObjects(o);
        },
        setBackgroundImage: (i: HTMLImageElement | null) => drawingRef.current?.setBackgroundImage(i),
        updateLatexObject: (id: string, latex: string) => {
            drawingRef.current?.updateLatexObject(id, latex);
        }
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
                }}
                onPointerMove={(e) => {
                    drawingRef.current?.handlePointerMove(e.nativeEvent);
                }}
                onPointerUp={() => {
                    drawingRef.current?.handlePointerUp();
                }}
                onPointerLeave={() => {
                    drawingRef.current?.handlePointerUp();
                }}
            />
            {latexObjects.map(obj => (
                <div 
                    key={obj.id}
                    className="latex-object-container"
                    style={{
                        position: 'absolute',
                        left: obj.x,
                        top: obj.y,
                        width: obj.width,
                        height: obj.height,
                        pointerEvents: mode === 'pointer' ? 'auto' : 'none',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        userSelect: 'none',
                        color: 'black'
                    }}
                >
                    {editingId === obj.id ? (
                        <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', background: 'white', border: '1px solid #aa3bff', zIndex: 10 }}>
                            <textarea 
                                value={editValue} 
                                onChange={(e) => setEditValue(e.target.value)}
                                style={{ flexGrow: 1, border: 'none', padding: '5px', fontSize: '12px', resize: 'none' }}
                            />
                            <div style={{ display: 'flex', gap: '2px' }}>
                                <button onClick={saveEdit} style={{ flexGrow: 1, fontSize: '10px', padding: '2px' }}>Save</button>
                                <button onClick={() => setEditingId(null)} style={{ flexGrow: 1, fontSize: '10px', padding: '2px' }}>Cancel</button>
                            </div>
                        </div>
                    ) : (
                        <>
                            <div 
                                style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
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
                                            (mathEl as HTMLElement).style.fontSize = `${Math.min(obj.width, obj.height) * 0.8 * defaultFontSize}px`;
                                        }
                                    }
                                }}
                            />
                            {mode === 'pointer' && (
                                <div className="object-actions" style={{ position: 'absolute', top: '-25px', right: 0, display: 'flex', gap: '5px' }}>
                                    <button onClick={() => handleCopy(obj.latex)} title="Copy TeX" style={{ padding: '2px 5px', fontSize: '10px', background: '#333', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}>Copy</button>
                                    <button onClick={() => startEditing(obj)} title="Edit TeX" style={{ padding: '2px 5px', fontSize: '10px', background: '#333', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}>Edit</button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            ))}
        </div>
    );
});
