import React, { useEffect, useState } from 'react';

interface ToastProps {
    message: string;
    duration?: number;
    onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({ message, duration = 2000, onClose }) => {
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsVisible(false);
            setTimeout(onClose, 300); // Wait for fade-out animation
        }, duration);

        return () => clearTimeout(timer);
    }, [duration, onClose]);

    return (
        <div className={`toast-popup ${isVisible ? 'visible' : 'hidden'}`}>
            <span className="toast-icon">✨</span>
            <span className="toast-message">{message}</span>
        </div>
    );
};
