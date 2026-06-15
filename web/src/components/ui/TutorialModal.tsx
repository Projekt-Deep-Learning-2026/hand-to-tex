import React from 'react';

export interface TutorialModalProps {
    onClose: () => void;
}

export const TutorialModal: React.FC<TutorialModalProps> = ({ onClose }) => {
    return (
        <div className="modal-overlay" onClick={onClose} data-testid="tutorial-overlay">
            <div className="modal-content tutorial-modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>How to use Hand-to-TeX</h3>
                    <button className="primary modal-close-btn" onClick={onClose} aria-label="Close">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
                <div className="tutorial-body">
                    <div className="tutorial-steps">
                        <div className="tutorial-step">
                            <div className="tutorial-icon">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"></path><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path><path d="M2 2l5 5"></path><path d="M9.5 14.5L16 8"></path></svg>
                            </div>
                            <div className="tutorial-text">
                                <strong>Draw</strong>
                                <p>Use the Pencil tool to write any mathematical expression.</p>
                            </div>
                        </div>
                        <div className="tutorial-step">
                            <div className="tutorial-icon">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                            </div>
                            <div className="tutorial-text">
                                <strong>Selective Recognition</strong>
                                <p>Use the Select tool to highlight a specific area for targeted recognition and conversion.</p>
                            </div>
                        </div>
                        <div className="tutorial-step">
                            <div className="tutorial-icon">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 3 10 21 13 13 21 10 3 3"></polyline><line x1="13" y1="13" x2="21" y2="21"></line></svg>
                            </div>
                            <div className="tutorial-text">
                                <strong>Pointer Tool</strong>
                                <p>Move or resize digitized math objects on your canvas.</p>
                            </div>
                        </div>
                        <div className="tutorial-step">
                            <div className="tutorial-icon">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 20H7L3 16C2 15 2 13 3 12L13 2C14 1 16 1 17 2L21 6C22 7 22 9 21 10L11 20"></path><line x1="17" y1="6" x2="7" y2="16"></line></svg>
                            </div>
                            <div className="tutorial-text">
                                <strong>Erase Tool</strong>
                                <p>Remove specific strokes or objects from the canvas.</p>
                            </div>
                        </div>
                    </div>
                    <button className="primary tutorial-exit-btn" onClick={onClose}>
                        <span>Get Started</span>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                    </button>
                </div>
            </div>
        </div>
    );
};
