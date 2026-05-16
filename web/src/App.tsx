import React, { useState, useRef, useEffect } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import './App.css';

import { useModel } from './hooks/useModel';
import { DrawingCanvas } from './components/DrawingCanvas';
import type { DrawingCanvasHandle } from './components/DrawingCanvas';
import { extractFeatures, runInference, parseInkml } from './logic/inference';

function App() {
    const { 
        status, 
        progress, 
        load, 
        encoderSession, 
        decoderSession, 
        vocab,
        error 
    } = useModel();

    const [latex, setLatex] = useState<string>('');
    const [isProcessing, setIsProcessing] = useState<boolean>(false);
    const canvasRef = useRef<DrawingCanvasHandle>(null);
    const previewRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (previewRef.current && latex) {
            try {
                katex.render(latex, previewRef.current, {
                    displayMode: true,
                    throwOnError: false
                });
            } catch (err) {
                console.error("KaTeX error:", err);
            }
        } else if (previewRef.current) {
            previewRef.current.innerHTML = 'Rendered preview will appear here';
        }
    }, [latex]);

    const handleRecognize = async () => {
        if (!canvasRef.current?.hasStrokes()) {
            setLatex("Please draw something first.");
            return;
        }

        if (!encoderSession || !decoderSession || !vocab) {
            setLatex("Models not loaded.");
            return;
        }

        setIsProcessing(true);
        setLatex("Processing drawing...");

        try {
            const traces = canvasRef.current.getTraces();
            const { flatData, numPoints, numFeatures } = extractFeatures(traces);

            if (numPoints === 0) {
                throw new Error("No valid drawing data captured.");
            }

            const tokenIds = await runInference(
                encoderSession,
                decoderSession,
                flatData,
                numPoints,
                numFeatures,
                vocab.SOS_IDX,
                vocab.EOS_IDX
            );

            const latexTokens = tokenIds
                .filter(id => id !== vocab.PAD_IDX && id !== vocab.SOS_IDX)
                .map(id => vocab.id2token[id]);

            setLatex(latexTokens.join(" "));
        } catch (err) {
            console.error("Inference error:", err);
            setLatex("Recognition error: " + (err as Error).message);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleClear = () => {
        canvasRef.current?.clear();
        setLatex('');
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !encoderSession || !decoderSession || !vocab) return;

        setIsProcessing(true);
        setLatex("Processing file...");

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const content = e.target?.result as string;
                const traces = parseInkml(content);
                
                if (traces.length === 0) throw new Error("No strokes found in file.");

                const { flatData, numPoints, numFeatures } = extractFeatures(traces);
                const tokenIds = await runInference(
                    encoderSession,
                    decoderSession,
                    flatData,
                    numPoints,
                    numFeatures,
                    vocab.SOS_IDX,
                    vocab.EOS_IDX
                );

                const latexTokens = tokenIds
                    .filter(id => id !== vocab.PAD_IDX && id !== vocab.SOS_IDX)
                    .map(id => vocab.id2token[id]);

                setLatex(latexTokens.join(" "));
            } catch (err) {
                console.error("Inference error:", err);
                setLatex("Recognition error: " + (err as Error).message);
            } finally {
                setIsProcessing(false);
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="app-container">
            <h1>Hand-to-TeX: Mathematical Expression Recognition</h1>

            <section className="card">
                <h3>Step 1: Initialize</h3>
                <p className="status" style={{ color: status === 'success' ? 'green' : status === 'error' ? 'red' : 'inherit' }}>
                    {progress}
                </p>
                {status !== 'success' && (
                    <button onClick={load} disabled={status === 'loading'}>
                        {status === 'loading' ? 'Loading...' : 'Load Models and Vocabulary'}
                    </button>
                )}
                {error && <p style={{ color: 'red', fontSize: '12px' }}>{error}</p>}
            </section>

            <section className="card">
                <h3>Step 2: Draw Expression</h3>
                <p>Draw your mathematical expression on the canvas below:</p>
                <DrawingCanvas ref={canvasRef} className="drawing-canvas" />
                <div className="button-group">
                    <button 
                        onClick={handleRecognize} 
                        disabled={status !== 'success' || isProcessing}
                    >
                        {isProcessing ? 'Recognizing...' : 'Recognize Drawing'}
                    </button>
                    <button 
                        onClick={handleClear} 
                        disabled={status !== 'success' || isProcessing}
                    >
                        Clear Canvas
                    </button>
                </div>
            </section>

            <section className="card">
                <h3>Step 3: Upload InkML File (Alternative)</h3>
                <input 
                    type="file" 
                    accept=".inkml" 
                    onChange={handleFileUpload} 
                    disabled={status !== 'success' || isProcessing} 
                />
                <p style={{ fontSize: '0.8em', color: '#666' }}>Or select an .inkml file to test alternative input method.</p>
            </section>

            <section className="card">
                <h3>Result:</h3>
                <div className="result-container">
                    <div className="preview-label">LaTeX Output:</div>
                    <div className="latex-output">{latex || 'No results yet.'}</div>
                </div>
                <div className="result-container">
                    <div className="preview-label">TeX Preview:</div>
                    <div ref={previewRef} className="preview-container">
                        Rendered preview will appear here
                    </div>
                </div>
            </section>
        </div>
    );
}

export default App;
