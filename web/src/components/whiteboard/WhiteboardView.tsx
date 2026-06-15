import React, { useState, useRef, useEffect, useCallback } from 'react';

import { DrawingCanvas } from './DrawingCanvas';
import type { DrawingCanvasHandle } from './DrawingCanvas';
import { SelectionWindow } from './SelectionWindow';
import { Toolbar } from './Toolbar';
import { EditModal } from '../editor/EditModal';
import { TutorialModal } from '../ui/TutorialModal';
import { useProjectIO } from '../../hooks/useProjectIO';
import type { ProjectData } from '../../hooks/useProjectIO';
import type { CanvasMode, LatexObject } from '../../logic/canvas';
import { extractFeatures } from '../../logic/inference';

interface WhiteboardViewProps {
    initialProjectData: ProjectData | null;
    onClearInitialData: () => void;
    onNavigateHome: () => void;
    onToast: (details: ModeDetails) => void;
    modelStatus: string;
    modelProgress: string;
    vocab: any;
    recognize: (flatData: Float32Array, numPoints: number, numFeatures: number) => Promise<number[]>;
}

export type ModeDetails = { icon?: string, message: string };

const MODE_DETAILS: Record<CanvasMode, ModeDetails> = {
    draw: { icon: '✏️', message: 'Draw Mode Active' },
    select: { icon: '🔍', message: 'Select Mode Active' },
    erase: { icon: '🧹', message: 'Erase Mode Active' },
    pointer: { icon: '🎯', message: 'Pointer Mode Active' }
};

export const WhiteboardView: React.FC<WhiteboardViewProps> = ({
    initialProjectData,
    onClearInitialData,
    onNavigateHome,
    onToast,
    modelStatus,
    modelProgress,
    vocab,
    recognize
}) => {
    const [canvasMode, setCanvasMode] = useState<CanvasMode>('draw');
    const [penOnlyMode, setPenOnlyMode] = useState<boolean>(false);
    const [selectedLatex, setSelectedLatex] = useState<string | null>(null);
    const [isSelectionProcessing, setIsSelectionProcessing] = useState(false);
    const [isSelectionWindowVisible, setIsSelectionWindowVisible] = useState(false);
    const [numSelectedTraces, setNumSelectedTraces] = useState(0);
    const [editingObject, setEditingObject] = useState<LatexObject | null>(null);
    const [showTutorial, setShowTutorial] = useState(false);

    const canvasRef = useRef<DrawingCanvasHandle>(null);
    const whiteboardWrapperRef = useRef<HTMLDivElement>(null);

    const { handleSaveToFile, handleLoadProject, handleExportPDF, isExporting } = useProjectIO(
        canvasRef, 
        whiteboardWrapperRef, 
        onToast
    );

    useEffect(() => {
        if (initialProjectData && canvasRef.current) {
            const timer = setTimeout(() => {
                if (initialProjectData.traces) canvasRef.current?.setTraces(initialProjectData.traces);
                if (initialProjectData.latexObjects) canvasRef.current?.setLatexObjects(initialProjectData.latexObjects);
                onClearInitialData();
            }, 50);
            return () => clearTimeout(timer);
        }
    }, [initialProjectData, onClearInitialData]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                canvasRef.current?.undo();
            } else if ((e.ctrlKey && e.key === 'y') || (e.ctrlKey && e.shiftKey && e.key === 'Z')) {
                e.preventDefault();
                canvasRef.current?.redo();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const changeCanvasMode = (mode: CanvasMode, showToast = true) => {
        setCanvasMode(mode);
        setSelectedLatex(null);
        setNumSelectedTraces(0);
        setIsSelectionWindowVisible(false);
        canvasRef.current?.clearSelection();
        if (showToast) {
            onToast(MODE_DETAILS[mode]);
        }
    };

    const performInference = useCallback(async (traces: number[][][]) => {
        if (modelStatus !== 'success' || !vocab) return "Models not loaded.";
        const { flatData, numPoints, numFeatures } = extractFeatures(traces);
        if (numPoints === 0) throw new Error("No valid drawing data captured.");

        const tokenIds = await recognize(flatData, numPoints, numFeatures);

        return tokenIds
            .filter((id: number) => id !== vocab.PAD_IDX && id !== vocab.SOS_IDX)
            .map((id: number) => vocab.id2token[id])
            .join(" ");
    }, [modelStatus, vocab, recognize]);

    const handleSelectionRecognize = useCallback(async (traces: number[][][]) => {
        setIsSelectionProcessing(true);
        setIsSelectionWindowVisible(true);
        setSelectedLatex("..."); 
        try {
            const result = await performInference(traces);
            setSelectedLatex(result || " ");
        } catch {
            setSelectedLatex("Error");
        } finally {
            setIsSelectionProcessing(false);
        }
    }, [performInference]);

    const handleReplace = () => {
        if (selectedLatex && !['...', 'Error'].includes(selectedLatex)) {
            canvasRef.current?.replaceSelectedWithLatex(selectedLatex);
            setSelectedLatex(null);
            setIsSelectionWindowVisible(false);
        }
    };

    const handleClear = () => {
        canvasRef.current?.clear();
        setSelectedLatex(null);
        setNumSelectedTraces(0);
        setIsSelectionWindowVisible(false);
        onToast({message: "Whiteboard cleared"});
    };

    return (
        <div className="whiteboard-container">
            <div className="whiteboard-header">
                <button aria-label="Back to home" className="back-button" onClick={() => {
                    onNavigateHome();
                }} style={{ marginBottom: 0 }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
                </button>
                <div className="whiteboard-title">Whiteboard</div>
                <div className={`model-status-indicator ${modelStatus}`} title={modelProgress}>
                    <span className="dot"></span>
                    <span className="label">
                        {
                        modelStatus === 'success' ? 'Ready' : 
                        modelStatus === 'loading' ? modelProgress.replace('Loading ', '') : 
                        modelStatus === 'error' ? 'Error' : 'Offline'
                        }
                    </span>
                </div>
                <div style={{ flexGrow: 1 }}></div>
                
                <Toolbar 
                    canvasMode={canvasMode}
                    penOnlyMode={penOnlyMode}
                    onChangeMode={changeCanvasMode}
                    onTogglePenOnly={() => setPenOnlyMode(!penOnlyMode)}
                    onUndo={() => canvasRef.current?.undo()}
                    onRedo={() => canvasRef.current?.redo()}
                    onClear={handleClear}
                    mini={true}
                />

                <div style={{ flexGrow: 1 }}></div>
                
                <div className="header-actions">
                    <button onClick={() => setShowTutorial(true)} className="mini help-btn" title="How to use">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                    </button>
                    <button onClick={handleSaveToFile} className="mini" title="Save Project as JSON">Save JSON</button>
                    <label className="button mini" title="Load Project from JSON">
                        Load JSON
                        <input type="file" accept=".json" onChange={handleLoadProject} style={{ display: 'none' }} />
                    </label>
                    <button onClick={() => handleExportPDF(canvasMode, setCanvasMode)} className="mini" disabled={isExporting} title="Export to PDF">
                        {isExporting ? 'Exporting...' : 'Export PDF'}
                    </button>
                </div>
            </div>
            <div className="whiteboard-scroll-area">
                <div className="whiteboard-canvas-wrapper" ref={whiteboardWrapperRef}>
                    <DrawingCanvas 
                        ref={canvasRef} 
                        className="whiteboard-canvas" 
                        mode={canvasMode} 
                        penOnlyMode={penOnlyMode}
                        onSelectionComplete={handleSelectionRecognize} 
                        onSelectionChange={setNumSelectedTraces}
                        onToast={onToast}
                        onEdit={setEditingObject}
                    />
                </div>
            </div>
            
            {canvasMode === 'select' && isSelectionWindowVisible && (
                <SelectionWindow 
                    latex={selectedLatex}
                    isProcessing={isSelectionProcessing}
                    isModelReady={modelStatus === 'success'}
                    numSelectedTraces={numSelectedTraces}
                    onReplace={handleReplace}
                    onClose={() => { 
                        setSelectedLatex(null); 
                        setIsSelectionWindowVisible(false);
                        canvasRef.current?.clearSelection(); 
                    }}
                />
            )}

            {editingObject && (
                <EditModal 
                    initialLatex={editingObject.latex}
                    onSave={(newLatex: string) => {
                        if (canvasRef.current) {
                            canvasRef.current.updateLatexObject(editingObject.id, newLatex);
                            setEditingObject(null);
                            onToast({message: "LaTeX updated"});
                        }
                    }}
                    onCancel={() => setEditingObject(null)}
                />
            )}

            {showTutorial && <TutorialModal onClose={() => setShowTutorial(false)} />}
        </div>
    );
};
