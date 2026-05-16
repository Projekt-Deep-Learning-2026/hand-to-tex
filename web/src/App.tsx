import React, { useState, useRef, useEffect, useCallback } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import './App.css';

import { useModel } from './hooks/useModel';
import { DrawingCanvas } from './components/DrawingCanvas';
import type { DrawingCanvasHandle } from './components/DrawingCanvas';
import type { CanvasMode } from './logic/canvas';
import { extractFeatures, runInference, parseInkml } from './logic/inference';
import { Home } from './components/Home';

type View = 'home' | 'demo' | 'whiteboard';

function App() {
    const [view, setView] = useState<View>('home');
    const [canvasMode, setCanvasMode] = useState<CanvasMode>('draw');
    const [selectedLatex, setSelectedLatex] = useState<string | null>(null);
    const [isSelectionProcessing, setIsSelectionProcessing] = useState(false);
    const [latex, setLatex] = useState<string>('');
    const [isProcessing, setIsProcessing] = useState<boolean>(false);

    const canvasRef = useRef<DrawingCanvasHandle>(null);
    const previewRef = useRef<HTMLDivElement>(null);
    const selectionPreviewRef = useRef<HTMLDivElement>(null);

    const { status, progress, load, encoderSession, decoderSession, vocab, error } = useModel();

    useEffect(() => {
        const root = document.getElementById('root');
        if (!root) return;
        view === 'whiteboard' ? root.classList.add('full-width') : root.classList.remove('full-width');
    }, [view]);

    const performInference = useCallback(async (traces: number[][][]) => {
        if (!encoderSession || !decoderSession || !vocab) return "Models not loaded.";
        
        const { flatData, numPoints, numFeatures } = extractFeatures(traces);
        if (numPoints === 0) throw new Error("No valid drawing data captured.");

        const tokenIds = await runInference(
            encoderSession, decoderSession, flatData, numPoints, numFeatures,
            vocab.SOS_IDX, vocab.EOS_IDX
        );

        return tokenIds
            .filter(id => id !== vocab.PAD_IDX && id !== vocab.SOS_IDX)
            .map(id => vocab.id2token[id])
            .join(" ");
    }, [encoderSession, decoderSession, vocab]);

    const handleRecognize = async () => {
        if (!canvasRef.current?.hasStrokes()) return setLatex("Please draw something first.");
        
        setIsProcessing(true);
        setLatex("Processing drawing...");

        try {
            const result = await performInference(canvasRef.current.getTraces());
            setLatex(result);
        } catch (err) {
            setLatex("Recognition error: " + (err as Error).message);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleSelectionRecognize = async (traces: number[][][]) => {
        setIsSelectionProcessing(true);
        setSelectedLatex("...");

        try {
            const result = await performInference(traces);
            setSelectedLatex(result);
        } catch (err) {
            setSelectedLatex("Error");
        } finally {
            setIsSelectionProcessing(false);
        }
    };

    useEffect(() => {
        if (!previewRef.current) return;
        if (latex) {
            katex.render(latex, previewRef.current, { displayMode: true, throwOnError: false });
        } else {
            previewRef.current.innerHTML = 'Rendered preview';
        }
    }, [latex, view]);

    useEffect(() => {
        if (selectionPreviewRef.current && selectedLatex && !['...', 'Error'].includes(selectedLatex)) {
            katex.render(selectedLatex, selectionPreviewRef.current, { displayMode: false, throwOnError: false });
        }
    }, [selectedLatex]);

    const handleClear = () => {
        canvasRef.current?.clear();
        setLatex('');
        setSelectedLatex(null);
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !encoderSession || !decoderSession || !vocab) return;

        setIsProcessing(true);
        setLatex("Processing file...");

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const traces = parseInkml(e.target?.result as string);
                if (traces.length === 0) throw new Error("No strokes found in file.");
                const result = await performInference(traces);
                setLatex(result);
            } catch (err) {
                setLatex("Recognition error: " + (err as Error).message);
            } finally {
                setIsProcessing(false);
            }
        };
        reader.readAsText(file);
    };

    const navigateToView = (v: View) => {
        setView(v);
        setLatex('');
        setSelectedLatex(null);
        if (status !== 'success') load();
    };

    const renderHome = () => <Home onSelectView={navigateToView} />;

    const renderDemo = () => (
        <div className="view-container">
            <button className="back-button" onClick={() => navigateToView('home')}>← Back to Home</button>
            <h1>Interactive Demo</h1>

            <section className="card">
                <h3>Model Status</h3>
                <p className="status" style={{ color: status === 'success' ? 'green' : status === 'error' ? 'red' : 'inherit' }}>{progress}</p>
                {status !== 'success' && (
                    <button onClick={load} disabled={status === 'loading'}>{status === 'loading' ? 'Loading...' : 'Load Models'}</button>
                )}
                {error && <p style={{ color: 'red', fontSize: '12px' }}>{error}</p>}
            </section>

            <section className="card">
                <div className="section-header">
                    <h3>Draw Expression</h3>
                    <div className="mode-toggle">
                        <button className={canvasMode === 'draw' ? 'active' : ''} onClick={() => setCanvasMode('draw')}>✏️ Draw</button>
                        <button className={canvasMode === 'select' ? 'active' : ''} onClick={() => setCanvasMode('select')}>🔍 Select</button>
                    </div>
                </div>
                <DrawingCanvas ref={canvasRef} className="drawing-canvas" mode={canvasMode} onSelectionComplete={handleSelectionRecognize} />
                <div className="button-group">
                    <button onClick={handleRecognize} disabled={status !== 'success' || isProcessing}>{isProcessing ? 'Recognizing...' : 'Recognize All'}</button>
                    <button onClick={handleClear} disabled={status !== 'success' || isProcessing}>Clear</button>
                </div>
            </section>

            {selectedLatex && (
                <section className="card selection-card">
                    <h3>Selection Result</h3>
                    <div className="selection-result">
                        {isSelectionProcessing ? <span>Processing selection...</span> : (
                            <>
                                <div ref={selectionPreviewRef} className="selection-preview"></div>
                                <code className="selection-latex">{selectedLatex}</code>
                                <button className="close-selection" onClick={() => { setSelectedLatex(null); canvasRef.current?.clearSelection(); }}>×</button>
                            </>
                        )}
                    </div>
                </section>
            )}

            <section className="card">
                <h3>Upload InkML</h3>
                <input type="file" accept=".inkml" onChange={handleFileUpload} disabled={status !== 'success' || isProcessing} />
            </section>

            <section className="card">
                <h3>Result:</h3>
                <div className="result-container">
                    <div className="preview-label">LaTeX Output:</div>
                    <div className="latex-output">{latex || 'No results yet.'}</div>
                </div>
                <div className="result-container">
                    <div className="preview-label">TeX Preview:</div>
                    <div ref={previewRef} className="preview-container">Rendered preview</div>
                </div>
            </section>
        </div>
    );

    const renderWhiteboard = () => (
        <div className="whiteboard-container">
            <div className="whiteboard-header">
                <button className="back-button" onClick={() => navigateToView('home')}>← Back</button>
                <div className="status-mini" style={{ color: status === 'success' ? 'green' : 'orange' }}>{status === 'success' ? '● Ready' : '● Loading...'}</div>
                <div className="whiteboard-title">Whiteboard Mode</div>
                <div className="mode-toggle mini">
                    <button className={canvasMode === 'draw' ? 'active' : ''} onClick={() => setCanvasMode('draw')}>✏️ Draw</button>
                    <button className={canvasMode === 'select' ? 'active' : ''} onClick={() => setCanvasMode('select')}>🔍 Select</button>
                </div>
            </div>

            <DrawingCanvas ref={canvasRef} className="whiteboard-canvas" mode={canvasMode} onSelectionComplete={handleSelectionRecognize} />

            {selectedLatex && (
                <div className="selection-popover">
                    <div className="popover-header">
                        <span>Selection Recognition</span>
                        <button onClick={() => { setSelectedLatex(null); canvasRef.current?.clearSelection(); }}>×</button>
                    </div>
                    <div className="popover-content">
                        {isSelectionProcessing ? "Recognizing..." : (
                            <>
                                <div ref={selectionPreviewRef} className="selection-preview-large"></div>
                                <div className="selection-latex-mini">{selectedLatex}</div>
                            </>
                        )}
                    </div>
                </div>
            )}

            <div className="whiteboard-result">
                <div ref={previewRef} className="preview-container">{latex ? '' : 'Draw and click Recognize'}</div>
                <div className="latex-mini">{latex}</div>
            </div>

            <div className="whiteboard-controls">
                <button onClick={handleRecognize} disabled={status !== 'success' || isProcessing} className="primary">{isProcessing ? 'Recognizing...' : 'Recognize All'}</button>
                <button onClick={handleClear} disabled={status !== 'success' || isProcessing}>Clear Canvas</button>
            </div>
        </div>
    );

    return (
        <div className="app-container">
            {view === 'home' && renderHome()}
            {view === 'demo' && renderDemo()}
            {view === 'whiteboard' && renderWhiteboard()}
        </div>
    );
}

export default App;
