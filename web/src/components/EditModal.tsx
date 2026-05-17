import React, { useState, useEffect, useRef } from 'react';
import katex from 'katex';

interface EditModalProps {
    initialLatex: string;
    onSave: (newLatex: string) => void;
    onCancel: () => void;
}

export const EditModal: React.FC<EditModalProps> = ({ initialLatex, onSave, onCancel }) => {
    const [latex, setLatex] = useState(initialLatex);
    const previewRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (previewRef.current) {
            try {
                katex.render(latex || ' ', previewRef.current, {
                    displayMode: true,
                    throwOnError: false
                });
            } catch (err) {
                console.error("KaTeX error:", err);
            }
        }
    }, [latex]);

    return (
        <div className="modal-overlay" onClick={onCancel}>
            <div className="modal-content edit-modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Edit LaTeX</h3>
                    <button className="close-btn" onClick={onCancel}>×</button>
                </div>
                
                <div className="modal-body">
                    <div className="preview-section">
                        <label>Preview</label>
                        <div ref={previewRef} className="latex-preview-box"></div>
                    </div>
                    
                    <div className="input-section">
                        <label>LaTeX Code</label>
                        <textarea 
                            value={latex}
                            onChange={(e) => setLatex(e.target.value)}
                            placeholder="Enter LaTeX code here..."
                            spellCheck={false}
                            autoFocus
                        />
                    </div>
                </div>
                
                <div className="modal-footer">
                    <button className="secondary" onClick={onCancel}>Cancel</button>
                    <button className="primary" onClick={() => onSave(latex)}>Save Changes</button>
                </div>
            </div>
        </div>
    );
};
