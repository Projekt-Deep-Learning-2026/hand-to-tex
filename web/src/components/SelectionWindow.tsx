import React from 'react';

interface SelectionWindowProps {
    latex: string | null;
    isProcessing: boolean;
    onReplace: () => void;
    onClose: () => void;
    selectionPreviewRef: React.RefObject<HTMLDivElement>;
    numSelectedTraces: number;
}

export const SelectionWindow: React.FC<SelectionWindowProps> = ({
    latex,
    isProcessing,
    onReplace,
    onClose,
    selectionPreviewRef,
    numSelectedTraces
}) => {
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
                        <div ref={selectionPreviewRef} className="latex-preview-large"></div>
                        <code className="latex-code">{latex}</code>
                        <button 
                            className="replace-btn primary" 
                            onClick={onReplace}
                            disabled={['...', 'Error'].includes(latex)}
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
