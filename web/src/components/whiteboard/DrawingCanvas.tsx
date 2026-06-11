import React, { useRef, useEffect, useImperativeHandle, forwardRef, useState } from 'react';
import { CanvasDrawing } from '../../logic/canvas';
import type { ModeDetails } from './WhiteboardView';
import type { CanvasMode, LatexObject } from '../../logic/canvas';
import katex from 'katex';

interface DrawingCanvasProps {
    className?: string;
    mode?: CanvasMode;
    penOnlyMode?: boolean;
    defaultFontSize?: number;
    onSelectionComplete?: (traces: number[][][]) => void;
    onSelectionChange?: (count: number) => void;
    onToast?: (details: ModeDetails) => void;
    onEdit?: (obj: LatexObject) => void;
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

interface LatexRendererProps {
    obj: LatexObject;
    x: number;
    y: number;
    width: number;
    height: number;
    mode: CanvasMode;
    onCopy: (latex: string) => void;
    onEdit?: (obj: LatexObject) => void;
    onDelete: (id: string) => void;
}

const LatexRenderer = React.memo(({ obj, x, y, width, height, mode, onCopy, onEdit, onDelete }: LatexRendererProps) => {
    const elRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (elRef.current) {
            katex.render(obj.latex, elRef.current, { 
                throwOnError: false,
                displayMode: true,
                output: 'mathml'
            });
            const mathEl = elRef.current.querySelector('.katex-display, .katex, math') || elRef.current;
            if (mathEl) {
                (mathEl as HTMLElement).style.margin = '0';
                (mathEl as HTMLElement).style.color = 'black';
                (mathEl as HTMLElement).style.fontSize = `${Math.min(width, height) * 0.8}px`;
            }
        }
    }, [obj.latex, width, height]);

    return (
        <div 
            className="latex-object-container"
            style={{
                position: 'absolute',
                left: x,
                top: y,
                width: width,
                height: height,
                pointerEvents: 'none',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                userSelect: 'none',
                color: 'black',
                border: mode === 'pointer' ? '1px dashed rgba(170, 59, 255, 0.3)' : 'none'
            }}
        >
            <div 
                style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                ref={elRef}
            />
            {mode === 'pointer' && (
                <div className="object-actions" style={{ position: 'absolute', top: '-25px', right: 0, display: 'flex', gap: '5px', pointerEvents: 'auto' }}>
                    <button onClick={() => onCopy(obj.latex)} title="Copy TeX" style={{ padding: '2px 5px', fontSize: '10px', background: '#333', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}>Copy</button>
                    <button onClick={() => onEdit?.(obj)} title="Edit TeX" style={{ padding: '2px 5px', fontSize: '10px', background: '#333', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}>Edit</button>
                    <button onClick={() => onDelete(obj.id)} title="Delete Object" style={{ padding: '2px 5px', fontSize: '10px', background: '#d32f2f', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}>Delete</button>
                </div>
            )}
        </div>
    );
});

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(({ 
    className, 
    mode = 'draw',
    penOnlyMode = false,
    onSelectionComplete,
    onSelectionChange,
    onToast,
    onEdit
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

    const handleCopy = (latex: string) => {
        const textToCopy = latex;
        
        // Modern approach
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(textToCopy).then(() => {
                onToast?.({message: "Copied to clipboard!"});
            }).catch(err => {
                console.error('Clipboard error:', err);
                fallbackCopy(textToCopy);
            });
        } else {
            fallbackCopy(textToCopy);
        }
    };

    const fallbackCopy = (text: string) => {
        try {
            const textArea = document.createElement("textarea");
            textArea.value = text;
            
            // Ensure the textarea is not visible
            textArea.style.position = "fixed";
            textArea.style.left = "-9999px";
            textArea.style.top = "0";
            document.body.appendChild(textArea);
            
            textArea.focus();
            textArea.select();
            
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (successful) {
                onToast?.({message: "Copied to clipboard!"});
            }
        } catch (err) {
            console.error('Fallback copy error:', err);
        }
    };

    const handleDelete = (id: string) => {
        if (drawingRef.current?.deleteLatexObject(id)) {
            onToast?.({message: "Object deleted"});
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
        },
        deleteLatexObject: (id: string) => {
            drawingRef.current?.deleteLatexObject(id);
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
                <LatexRenderer 
                    key={obj.id} 
                    obj={obj} 
                    x={obj.x}
                    y={obj.y}
                    width={obj.width}
                    height={obj.height}
                    mode={mode} 
                    onCopy={handleCopy} 
                    onEdit={onEdit} 
                    onDelete={handleDelete} 
                />
            ))}
        </div>
    );
});
