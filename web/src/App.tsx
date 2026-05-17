import React, { useState, useRef, useEffect, useCallback } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import './App.css';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

import { useModel } from './hooks/useModel';
import { DrawingCanvas } from './components/DrawingCanvas';
import type { DrawingCanvasHandle } from './components/DrawingCanvas';
import type { CanvasMode } from './logic/canvas';
import { extractFeatures } from './logic/inference';
import { Home } from './components/Home';
import { SelectionWindow } from './components/SelectionWindow';
import { ModeHint } from './components/ModeHint';

type View = 'home' | 'demo' | 'whiteboard';

const MODE_DETAILS: Record<CanvasMode, { icon: string, message: string }> = {
    draw: { icon: '✏️', message: 'Draw with your mouse or pen' },
    select: { icon: '🔍', message: 'Drag to select traces for recognition' },
    erase: { icon: '🧹', message: 'Click or drag over traces to erase' },
    pointer: { icon: '🎯', message: 'Click and drag objects to move or resize' }
};

function App() {
    const [view, setView] = useState<View>('home');
    const [canvasMode, setCanvasMode] = useState<CanvasMode>('draw');
    const [penOnlyMode, setPenOnlyMode] = useState<boolean>(false);
    const [selectedLatex, setSelectedLatex] = useState<string | null>(null);
    const [isSelectionProcessing, setIsSelectionProcessing] = useState(false);
    const [isSelectionWindowVisible, setIsSelectionWindowVisible] = useState(false);
    const [numSelectedTraces, setNumSelectedTraces] = useState(0);
    const [isProcessing, setIsProcessing] = useState<boolean>(false);
    const [initialProjectData, setInitialProjectData] = useState<{traces?: number[][][], latexObjects?: unknown[]} | null>(null);
    const [showTutorial, setShowTutorial] = useState(false);

    const canvasRef = useRef<DrawingCanvasHandle>(null);
    const selectionPreviewRef = useRef<HTMLDivElement>(null);
    const whiteboardWrapperRef = useRef<HTMLDivElement>(null);

    const { 
        status: modelStatus, 
        progress: modelProgress, 
        load: loadModel, 
        vocab,
        recognize
    } = useModel();

    useEffect(() => {
        const root = document.getElementById('root');
        if (!root) return;
        root.classList.toggle('full-width', view === 'whiteboard');
    }, [view]);

    useEffect(() => {
        if (view === 'whiteboard' && initialProjectData && canvasRef.current) {
            const timer = setTimeout(() => {
                if (initialProjectData.traces) canvasRef.current?.setTraces(initialProjectData.traces);
                if (initialProjectData.latexObjects) canvasRef.current?.setLatexObjects(initialProjectData.latexObjects);
                setInitialProjectData(null);
            }, 50);
            return () => clearTimeout(timer);
        }
    }, [view, initialProjectData]);

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

    const performInference = useCallback(async (traces: number[][][]) => {
        if (modelStatus !== 'success' || !vocab) return "Models not loaded.";
        const { flatData, numPoints, numFeatures } = extractFeatures(traces);
        if (numPoints === 0) throw new Error("No valid drawing data captured.");

        const tokenIds = await recognize(flatData, numPoints, numFeatures);

        return tokenIds
            .filter(id => id !== vocab.PAD_IDX && id !== vocab.SOS_IDX)
            .map(id => vocab.id2token[id])
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

    useEffect(() => {
        if (selectionPreviewRef.current && selectedLatex && !['...', 'Error'].includes(selectedLatex)) {
            katex.render(selectedLatex, selectionPreviewRef.current, { displayMode: true, throwOnError: false });
        }
    }, [selectedLatex, isSelectionProcessing]);

    const handleClear = () => {
        canvasRef.current?.clear();
        setSelectedLatex(null);
        setNumSelectedTraces(0);
        setIsSelectionWindowVisible(false);
    };

    const navigateToView = (v: View) => {
        setView(v);
        changeCanvasMode('draw');
        if (modelStatus !== 'success') loadModel();
    };

    const changeCanvasMode = (mode: CanvasMode) => {
        setCanvasMode(mode);
        setSelectedLatex(null);
        setNumSelectedTraces(0);
        setIsSelectionWindowVisible(false);
        canvasRef.current?.clearSelection();
    };

    const handleSaveToFile = () => {
        if (!canvasRef.current) return;
        const traces = canvasRef.current.getTraces();
        const latexObjects = canvasRef.current.getLatexObjects();
        
        const data = {
            version: "1.0",
            timestamp: new Date().toISOString(),
            traces,
            latexObjects
        };

        const defaultName = `hand-to-tex-project-${new Date().getTime()}`;
        const fileName = prompt("Enter project filename:", defaultName) || defaultName;
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName.endsWith('.json') ? fileName : `${fileName}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleLoadProject = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const data = JSON.parse(event.target?.result as string);
                
                if (!data || typeof data !== 'object') throw new Error("Invalid project format");
                if (!Array.isArray(data.traces) && !Array.isArray(data.latexObjects)) {
                    throw new Error("File does not contain valid whiteboard data");
                }

                if (view !== 'whiteboard') {
                    setInitialProjectData(data);
                    navigateToView('whiteboard');
                } else {
                    if (data.traces) canvasRef.current?.setTraces(data.traces);
                    if (data.latexObjects) canvasRef.current?.setLatexObjects(data.latexObjects);
                    changeCanvasMode('draw');
                }
            } catch (err) {
                alert("Error loading project: " + (err as Error).message);
            }
        };
        reader.readAsText(file);
        e.target.value = '';
    };

    const handleExportPDF = async () => {
        if (!whiteboardWrapperRef.current) return;
        setIsProcessing(true);
        const originalMode = canvasMode;
        try {
            if (canvasMode === 'pointer' || canvasMode === 'select') {
                canvasRef.current?.setMode('draw');
            }

            const defaultName = `hand-to-tex-export-${new Date().getTime()}`;
            const fileName = prompt("Enter PDF filename:", defaultName) || defaultName;

            const canvas = await html2canvas(whiteboardWrapperRef.current, {
                useCORS: true, scale: 2, backgroundColor: "#ffffff"
            });
            const imgData = canvas.toDataURL('image/jpeg', 0.95);
            const pdf = new jsPDF({
                orientation: canvas.width > canvas.height ? 'l' : 'p',
                unit: 'px', format: [canvas.width, canvas.height]
            });
            pdf.addImage(imgData, 'JPEG', 0, 0, canvas.width, canvas.height);
            pdf.save(fileName.endsWith('.pdf') ? fileName : `${fileName}.pdf`);
        } catch (err) {
            alert("Error exporting PDF: " + (err as Error).message);
        } finally {
            canvasRef.current?.setMode(originalMode);
            setIsProcessing(false);
        }
    };

    const renderDemo = () => (
        <div className="view-container">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <button className="back-button" onClick={() => navigateToView('home')} style={{ marginBottom: 0 }}>← Back to Home</button>
                <button className="mini" onClick={() => setShowTutorial(true)} style={{ borderRadius: '50%', width: '32px', height: '32px', padding: 0, fontSize: '18px' }}>ℹ️</button>
            </div>
                        <h1>Interactive Demo</h1>
            <section className="card">
                <h3>Model Status</h3>
                <p className="status" style={{ color: modelStatus === 'success' ? 'green' : modelStatus === 'error' ? 'red' : 'inherit' }}>{modelProgress}</p>
                {modelStatus !== 'success' && <button onClick={loadModel} disabled={modelStatus === 'loading'}>{modelStatus === 'loading' ? 'Loading...' : 'Load Models'}</button>}
            </section>
            <section className="card">
                <div className="section-header">
                    <h3>Draw Expression</h3>
                    {renderModeToggle()}
                </div>
                <DrawingCanvas 
                    ref={canvasRef} 
                    className="drawing-canvas" 
                    mode={canvasMode} 
                    penOnlyMode={penOnlyMode}
                    onSelectionComplete={handleSelectionRecognize}
                    onSelectionChange={setNumSelectedTraces}
                />
            </section>
            {canvasMode === 'select' && isSelectionWindowVisible && (
                <SelectionWindow 
                    latex={selectedLatex}
                    isProcessing={isSelectionProcessing}
                    numSelectedTraces={numSelectedTraces}
                    onReplace={handleReplace}
                    onClose={() => { 
                        setSelectedLatex(null); 
                        setIsSelectionWindowVisible(false);
                        canvasRef.current?.clearSelection(); 
                    }}
                    selectionPreviewRef={selectionPreviewRef}
                />
            )}
            <ModeHint 
                key={`${canvasMode}-${penOnlyMode}`} 
                mode={canvasMode} 
                icon={MODE_DETAILS[canvasMode].icon} 
                message={penOnlyMode && ['draw', 'select', 'erase'].includes(canvasMode) 
                    ? `Pen Only Mode Active: ${MODE_DETAILS[canvasMode].message}` 
                    : MODE_DETAILS[canvasMode].message} 
            />
        </div>
    );

    const renderWhiteboard = () => (
        <div className="whiteboard-container">
            <div className="whiteboard-header">
                <button className="back-button" onClick={() => navigateToView('home')} style={{ marginBottom: 0 }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
                </button>
                <div className="whiteboard-title">Whiteboard</div>
                <div style={{ flexGrow: 1 }}></div>
                {renderModeToggle(true)}
                <div style={{ flexGrow: 1 }}></div>
                
                <div className="header-actions">
                    <button onClick={handleSaveToFile} className="mini" title="Save Project as JSON">Save JSON</button>
                    <label className="button mini" title="Load Project from JSON">
                        Load JSON
                        <input type="file" accept=".json" onChange={handleLoadProject} style={{ display: 'none' }} />
                    </label>
                    <button onClick={handleExportPDF} className="mini" disabled={isProcessing} title="Export to PDF">
                        {isProcessing ? 'Exporting...' : 'Export PDF'}
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
                    />
                </div>
            </div>
            
            {canvasMode === 'select' && isSelectionWindowVisible && (
                <SelectionWindow 
                    latex={selectedLatex}
                    isProcessing={isSelectionProcessing}
                    numSelectedTraces={numSelectedTraces}
                    onReplace={handleReplace}
                    onClose={() => { 
                        setSelectedLatex(null); 
                        setIsSelectionWindowVisible(false);
                        canvasRef.current?.clearSelection(); 
                    }}
                    selectionPreviewRef={selectionPreviewRef}
                />
            )}
            <ModeHint 
                key={`${canvasMode}-${penOnlyMode}`} 
                mode={canvasMode} 
                icon={MODE_DETAILS[canvasMode].icon} 
                message={penOnlyMode && ['draw', 'select', 'erase'].includes(canvasMode) 
                    ? `Pen Only Mode Active: ${MODE_DETAILS[canvasMode].message}` 
                    : MODE_DETAILS[canvasMode].message} 
            />
        </div>
    );

    const renderModeToggle = (mini = false) => (
        <div className={`mode-toggle ${mini ? 'mini' : ''}`}>
            <button 
                className={canvasMode === 'draw' ? 'active' : ''} 
                onClick={() => changeCanvasMode('draw')}
                title="Draw (Pencil)"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"></path><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path><path d="M2 2l5 5"></path><path d="M9.5 14.5L16 8"></path></svg>
                {!mini && <span>Draw</span>}
            </button>
            <button 
                className={canvasMode === 'select' ? 'active' : ''} 
                onClick={() => changeCanvasMode('select')}
                title="Select (Lasso)"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                {!mini && <span>Select</span>}
            </button>
            <button 
                className={canvasMode === 'erase' ? 'active' : ''} 
                onClick={() => changeCanvasMode('erase')}
                title="Erase"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 20H7L3 16C2 15 2 13 3 12L13 2C14 1 16 1 17 2L21 6C22 7 22 9 21 10L11 20"></path><line x1="17" y1="6" x2="7" y2="16"></line></svg>
                {!mini && <span>Erase</span>}
            </button>
            <button 
                className={canvasMode === 'pointer' ? 'active' : ''} 
                onClick={() => changeCanvasMode('pointer')}
                title="Pointer (Move/Resize)"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 3 10 21 13 13 21 10 3 3"></polyline><line x1="13" y1="13" x2="21" y2="21"></line></svg>
                {!mini && <span>Pointer</span>}
            </button>
            <div className="separator"></div>
            <button 
                className={penOnlyMode ? 'active' : ''} 
                onClick={() => setPenOnlyMode(!penOnlyMode)}
                title="Lock to Pen only (for mobile)"
                style={{ color: penOnlyMode ? '#aa3bff' : '#aaa' }}
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"></path><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path></svg>
                {!mini && <span>Pen Only</span>}
            </button>
            <div className="separator"></div>
            <button onClick={() => canvasRef.current?.undo()} title="Undo (Ctrl + Z)">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7v6h6"></path><path d="M21 17a9 9 0 00-9-9 9 9 0 00-6 2.3L3 13"></path></svg>
                {!mini && <span>Undo</span>}
            </button>
            <button onClick={() => canvasRef.current?.redo()} title="Redo (Ctrl + Y)">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 7v6h-6"></path><path d="M3 17a9 9 0 019-9 9 9 0 016 2.3L21 13"></path></svg>
                {!mini && <span>Redo</span>}
            </button>
            <div className="separator"></div>
            <button onClick={handleClear} title="Clear Canvas">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                {!mini && <span>Clear</span>}
            </button>
        </div>
    );

    return (
        <div className="app-container">
            {view === 'home' && <Home onSelectView={navigateToView} onLoadProject={handleLoadProject} />}
            {view === 'demo' && renderDemo()}
            {view === 'whiteboard' && renderWhiteboard()}

            {showTutorial && (
                <div className="tutorial-overlay" onClick={() => setShowTutorial(false)}>
                    <div className="tutorial-content" onClick={(e) => e.stopPropagation()}>
                        <h2>How to use Hand-to-TeX</h2>
                        <ol>
                            <li><strong>Draw:</strong> Use the Pencil tool to write any mathematical expression.</li>
                            <li><strong>Selective Recognition:</strong> Use the Select tool (🔍) to highlight a specific area for targeted recognition and conversion.</li>
                            <li><strong>Pointer Tool:</strong> Move or resize digitized math objects on your canvas.</li>
                            <li><strong>Erase Tool:</strong> Remove specific strokes or objects from the canvas.</li>
                        </ol>
                        <button className="primary close-tutorial" onClick={() => setShowTutorial(false)}>Got it!</button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;
