import React, { useRef, useEffect } from 'react';
import katex from 'katex';

export interface LatexEditorProps {
    latex: string;
    onChange: (newLatex: string) => void;
    id?: string;
}

export const LatexEditor: React.FC<LatexEditorProps> = ({ latex, onChange, id }) => {
    const previewRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (previewRef.current) {
            try {
                katex.render(latex || ' ', previewRef.current, {
                    displayMode: true,
                    throwOnError: false,
                    output: 'mathml'
                });
            } catch (err) {
                console.error("KaTeX error:", err);
            }
        }
    }, [latex]);

    return (
        <div className="latex-editor" id={id}>
            <div className="preview-section">
                <label>Preview</label>
                <div ref={previewRef} className="latex-preview-box"></div>
            </div>
            
            <div className="input-section">
                <label>LaTeX Code</label>
                <textarea 
                    value={latex}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder="Enter LaTeX code here..."
                    spellCheck={false}
                    autoFocus
                    className="latex-code-input"
                />
            </div>
        </div>
    );
};
