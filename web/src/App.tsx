import React, { useState, useEffect } from 'react';
import './App.css';

import { useModel } from './hooks/useModel';
import type { ProjectData } from './hooks/useProjectIO';
import { Home } from './components/home/Home';
import { Toast } from './components/ui/Toast';
import { WhiteboardView } from './components/whiteboard/WhiteboardView';

type View = 'home' | 'whiteboard';
type ModeDetails = { icon?: string, message: string };

function App() {
    const [view, setView] = useState<View>('home');
    const [initialProjectData, setInitialProjectData] = useState<ProjectData | null>(null);
    const [showTutorial, setShowTutorial] = useState(false);
    const [toast, setToast] = useState<ModeDetails | null>(null);

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

    const navigateToView = (v: View) => {
        setView(v);
        if (modelStatus !== 'success') loadModel();
    };

    const handleLoadProjectFromHome = (e: React.ChangeEvent<HTMLInputElement>) => {
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

                setInitialProjectData(data);
                navigateToView('whiteboard');
                setToast({message: "Project loaded"});
            } catch (err) {
                alert("Error loading project: " + (err as Error).message);
            }
        };
        reader.readAsText(file);
        e.target.value = '';
    };

    return (
        <div className="app-container">
            {view === 'home' && <Home onSelectView={navigateToView} onLoadProject={handleLoadProjectFromHome} />}
            
            {view === 'whiteboard' && (
                <WhiteboardView 
                    initialProjectData={initialProjectData}
                    onClearInitialData={() => setInitialProjectData(null)}
                    onNavigateHome={() => {
                        navigateToView('home');
                        setToast(null);
                    }}
                    onToast={details => setToast(details)}
                    modelStatus={modelStatus}
                    modelProgress={modelProgress}
                    vocab={vocab}
                    recognize={recognize}
                />
            )}

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

            {toast && <Toast message={toast.message} icon={toast?.icon} onClose={() => setToast(null)} />}
        </div>
    );
}

export default App;
