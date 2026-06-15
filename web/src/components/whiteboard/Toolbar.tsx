import React from 'react';
import type { CanvasMode } from '../../logic/canvas';

interface ToolbarProps {
    canvasMode: CanvasMode;
    penOnlyMode: boolean;
    onChangeMode: (mode: CanvasMode) => void;
    onTogglePenOnly: () => void;
    onUndo: () => void;
    onRedo: () => void;
    onClear: () => void;
    mini?: boolean;
}

export const Toolbar: React.FC<ToolbarProps> = ({
    canvasMode,
    penOnlyMode,
    onChangeMode,
    onTogglePenOnly,
    onUndo,
    onRedo,
    onClear,
    mini = false
}) => {
    return (
        <div className={`mode-toggle ${mini ? 'mini' : ''}`}>
            <button 
                className={canvasMode === 'draw' ? 'active' : ''} 
                onClick={() => onChangeMode('draw')}
                title="Draw (Pencil)"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"></path><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path><path d="M2 2l5 5"></path><path d="M9.5 14.5L16 8"></path></svg>
                {!mini && <span>Draw</span>}
            </button>
            <button 
                className={canvasMode === 'select' ? 'active' : ''} 
                onClick={() => onChangeMode('select')}
                title="Select (Lasso)"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                {!mini && <span>Select</span>}
            </button>
            <button 
                className={canvasMode === 'erase' ? 'active' : ''} 
                onClick={() => onChangeMode('erase')}
                title="Erase"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 20H7L3 16C2 15 2 13 3 12L13 2C14 1 16 1 17 2L21 6C22 7 22 9 21 10L11 20"></path><line x1="17" y1="6" x2="7" y2="16"></line></svg>
                {!mini && <span>Erase</span>}
            </button>
            <button 
                className={canvasMode === 'pointer' ? 'active' : ''} 
                onClick={() => onChangeMode('pointer')}
                title="Pointer (Move/Resize)"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 3 10 21 13 13 21 10 3 3"></polyline><line x1="13" y1="13" x2="21" y2="21"></line></svg>
                {!mini && <span>Pointer</span>}
            </button>
            <div className="separator"></div>
            <button 
                className={penOnlyMode ? 'active' : ''} 
                onClick={onTogglePenOnly}
                title="Lock to Pen only (for mobile)"
                style={{ color: penOnlyMode ? '#aa3bff' : '#aaa' }}
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"></path><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path></svg>
                {!mini && <span>Pen Only</span>}
            </button>
            <div className="separator"></div>
            <button onClick={onUndo} title="Undo (Ctrl + Z)">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7v6h6"></path><path d="M21 17a9 9 0 00-9-9 9 9 0 00-6 2.3L3 13"></path></svg>
                {!mini && <span>Undo</span>}
            </button>
            <button onClick={onRedo} title="Redo (Ctrl + Y)">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 7v6h-6"></path><path d="M3 17a9 9 0 019-9 9 9 0 016 2.3L21 13"></path></svg>
                {!mini && <span>Redo</span>}
            </button>
            <div className="separator"></div>
            <button onClick={onClear} title="Clear Canvas">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                {!mini && <span>Clear</span>}
            </button>
        </div>
    );
};
