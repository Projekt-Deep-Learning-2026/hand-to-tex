import React, { useState } from 'react';
import { LatexEditor } from './LatexEditor';

interface EditModalProps {
    initialLatex: string;
    onSave: (newLatex: string) => void;
    onCancel: () => void;
}

export const EditModal: React.FC<EditModalProps> = ({ initialLatex, onSave, onCancel }) => {
    const [latex, setLatex] = useState(initialLatex);

    return (
        <div className="modal-overlay" onClick={onCancel}>
            <div className="modal-content edit-modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Edit LaTeX</h3>
                    <button className="close-btn" onClick={onCancel}>×</button>
                </div>
                
                <div className="modal-body">
                    <LatexEditor latex={latex} onChange={setLatex} />
                </div>
                
                <div className="modal-footer">
                    <button className="secondary" onClick={onCancel}>Cancel</button>
                    <button className="primary" onClick={() => onSave(latex)}>Save Changes</button>
                </div>
            </div>
        </div>
    );
};
