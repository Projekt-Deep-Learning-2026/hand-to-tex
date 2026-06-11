import React from 'react';

interface CTAButtonProps {
    icon: string;
    title: string;
    subtitle: string;
    onClick?: () => void;
    onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
    type?: 'primary' | 'secondary';
    isFileInput?: boolean;
    accept?: string;
}

export const CTAButton: React.FC<CTAButtonProps> = ({
    icon,
    title,
    subtitle,
    onClick,
    onChange,
    type = 'secondary',
    isFileInput = false,
    accept
}) => {
    const content = (
        <>
            <span className="icon">{icon}</span>
            <div className="text">
                <strong>{title}</strong>
                <span>{subtitle}</span>
            </div>
        </>
    );

    if (isFileInput) {
        return (
            <label className={`cta-button ${type}`} style={{ cursor: 'pointer' }}>
                {content}
                <input 
                    type="file" 
                    accept={accept} 
                    onChange={onChange} 
                    style={{ display: 'none' }} 
                />
            </label>
        );
    }

    return (
        <button className={`cta-button ${type}`} onClick={onClick}>
            {content}
        </button>
    );
};
