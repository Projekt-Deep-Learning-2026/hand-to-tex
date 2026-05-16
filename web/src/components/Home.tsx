import React from 'react';

interface HomeProps {
    onSelectView: (view: 'demo' | 'whiteboard') => void;
}

export const Home: React.FC<HomeProps> = ({ onSelectView }) => {
    return (
        <div className="home-container">
            <header className="hero">
                <h1>Hand-to-TeX</h1>
                <p className="subtitle">Transform your handwritten math into digital LaTeX effortlessly.</p>
            </header>
            
            <div className="cta-group">
                <button className="cta-button primary" onClick={() => onSelectView('demo')}>
                    <span className="icon">🚀</span>
                    <div className="text">
                        <strong>Try Interactive Demo</strong>
                        <span>Step-by-step recognition walkthrough</span>
                    </div>
                </button>
                
                <button className="cta-button secondary" onClick={() => onSelectView('whiteboard')}>
                    <span className="icon">🎨</span>
                    <div className="text">
                        <strong>Open Whiteboard</strong>
                        <span>Fullscreen distraction-free drawing</span>
                    </div>
                </button>
            </div>

            <section className="features">
                <div className="feature">
                    <h3>Fast Inference</h3>
                    <p>Runs directly in your browser using ONNX Runtime Web.</p>
                </div>
                <div className="feature">
                    <h3>Accurate</h3>
                    <p>Powered by a deep learning model trained on mathematical expressions.</p>
                </div>
                <div className="feature">
                    <h3>Private</h3>
                    <p>No data leaves your device. Everything is processed locally.</p>
                </div>
            </section>
        </div>
    );
};
