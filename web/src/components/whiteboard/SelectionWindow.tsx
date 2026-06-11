import React, { useRef, useEffect } from 'react';
import katex from 'katex';

interface SelectionWindowProps {
    latex: string | null;
    isProcessing: boolean;
    isModelReady: boolean;
    onReplace: () => void;
    onClose: () => void;
    numSelectedTraces: number;
}

export const SelectionWindow: React.FC<SelectionWindowProps> = ({
    latex,
    isProcessing,
    isModelReady,
    onReplace,
    onClose,
    numSelectedTraces
}) => {
    const previewRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (previewRef.current && latex && !['...', 'Error'].includes(latex)) {
            try {
                katex.render(latex, previewRef.current, {
                    displayMode: true,
                    throwOnError: false,
                    output: 'mathml'
                });
            } catch (err) {
                console.error("KaTeX error:", err);
            }
        }
    }, [latex, isProcessing]);

    return (
        <div className="selection-window">
            <div className="window-header">
                <span>Selection Recognition</span>
                <button className="close-btn" onClick={(e) => {
                    e.stopPropagation();
                    onClose();
                }}>×</button>
            </div>
            <div className="window-content">
                {isProcessing ? (
                    <div className="processing">
                        <div className="spinner"></div>
                        <span>Recognizing {numSelectedTraces} traces...</span>
                    </div>
                ) : latex ? (
                    <div className="result-area">
                        <div ref={previewRef} className="latex-preview-large"></div>
                        <code className="latex-code">{latex}</code>
                        <button 
                            className="replace-btn primary" 
                            onClick={onReplace}
                            disabled={!isModelReady || ['...', 'Error', 'Models not loaded.'].includes(latex)}
                        >
                            Replace Selected Traces
                        </button>
                    </div>
                ) : (
                    <div className="empty-selection">
                        {numSelectedTraces > 0 ? `${numSelectedTraces} traces selected. Release to recognize.` : "No traces selected."}
                    </div>
                )}
            </div>
        </div>
    );
};
