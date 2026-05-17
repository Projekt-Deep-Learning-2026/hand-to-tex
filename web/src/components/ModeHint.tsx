import React, { useEffect, useState } from 'react';

interface ModeHintProps {
    mode: string;
    icon: string;
    message: string;
}

export const ModeHint: React.FC<ModeHintProps> = ({ mode, icon, message }) => {
    const [visible, setVisible] = useState(false);
    const [shouldRender, setShouldRender] = useState(true);

    useEffect(() => {
        // Small delay to trigger entry animation
        const showTimer = setTimeout(() => setVisible(true), 10);
        
        const hideTimer = setTimeout(() => {
            setVisible(false);
        }, 5000);

        const removeTimer = setTimeout(() => {
            setShouldRender(false);
        }, 5500); // Wait for fade animation (0.5s)

        return () => {
            clearTimeout(showTimer);
            clearTimeout(hideTimer);
            clearTimeout(removeTimer);
        };
    }, [mode, message]); // Re-run when mode or message changes

    if (!shouldRender) return null;

    return (
        <div className={`mode-hint ${visible ? 'visible' : 'hidden'}`}>
            <div className="hint-icon">{icon}</div>
            <p>{message}</p>
        </div>
    );
};
