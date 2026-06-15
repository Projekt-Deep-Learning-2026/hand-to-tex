import React, { useEffect, useState } from 'react';

interface ToastProps {
    message: string;
    icon?: string;
    duration?: number;
    onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({ message, icon = '✨', duration = 2000, onClose }) => {
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        // Trigger entry animation
        const entryTimer = setTimeout(() => setIsVisible(true), 10);

        const exitTimer = setTimeout(() => {
            setIsVisible(false);
            setTimeout(onClose, 300); // Wait for fade-out animation
        }, duration);

        return () => {
            clearTimeout(entryTimer);
            clearTimeout(exitTimer);
        };
    }, [duration, onClose]);

    return (
        <div className={`toast-popup ${isVisible ? 'visible' : 'hidden'}`}>
            <span className="toast-icon">{icon}</span>
            <span className="toast-message">{message}</span>
        </div>
    );
};
