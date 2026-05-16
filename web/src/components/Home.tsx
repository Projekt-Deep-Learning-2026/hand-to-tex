import React from 'react';
import { CTAButton } from './CTAButton';

interface HomeProps {
    onSelectView: (view: 'demo' | 'whiteboard') => void;
    onLoadProject: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export const Home: React.FC<HomeProps> = ({ onSelectView, onLoadProject }) => {
    return (
        <div className="home-container">
            <header className="hero">
                <h1>Hand-to-TeX</h1>
                <p className="subtitle">Transform your handwritten math into digital LaTeX effortlessly.</p>
            </header>
            
            <div className="cta-group">
                <CTAButton 
                    type="primary"
                    icon="🚀"
                    title="Try Interactive Demo"
                    subtitle="Step-by-step recognition walkthrough"
                    onClick={() => onSelectView('demo')}
                />
                
                <CTAButton 
                    type="secondary"
                    icon="🎨"
                    title="Open Whiteboard"
                    subtitle="Fullscreen distraction-free drawing"
                    onClick={() => onSelectView('whiteboard')}
                />

                <CTAButton 
                    type="secondary"
                    icon="📂"
                    title="Load Project"
                    subtitle="Restore work from JSON"
                    isFileInput={true}
                    accept=".json"
                    onChange={onLoadProject}
                />
            </div>
        </div>
    );
};

