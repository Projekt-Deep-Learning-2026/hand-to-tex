import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { CTAButton } from './CTAButton';

describe('CTAButton Component', () => {
    const defaultProps = {
        icon: '🚀',
        title: 'Launch',
        subtitle: 'Start the mission'
    };

    it('renders title, subtitle and icon', () => {
        render(<CTAButton {...defaultProps} />);
        
        expect(screen.getByText('Launch')).toBeInTheDocument();
        expect(screen.getByText('Start the mission')).toBeInTheDocument();
        expect(screen.getByText('🚀')).toBeInTheDocument();
    });

    it('calls onClick when clicked', () => {
        const handleClick = vi.fn();
        render(<CTAButton {...defaultProps} onClick={handleClick} />);
        
        fireEvent.click(screen.getByRole('button'));
        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('renders as a label with input when isFileInput is true', () => {
        const handleChange = vi.fn();
        render(<CTAButton {...defaultProps} isFileInput={true} onChange={handleChange} accept=".json" />);
        
        const label = screen.getByText('Launch').closest('label');
        expect(label).toBeInTheDocument();
        
        const input = label?.querySelector('input[type="file"]') as HTMLInputElement;
        expect(input).toBeInTheDocument();
        expect(input.accept).toBe('.json');
        
        const file = new File(['{}'], 'test.json', { type: 'application/json' });
        fireEvent.change(input, { target: { files: [file] } });
        expect(handleChange).toHaveBeenCalledTimes(1);
    });

    it('applies the correct class based on type prop', () => {
        const { rerender } = render(<CTAButton {...defaultProps} type="primary" />);
        expect(screen.getByRole('button')).toHaveClass('primary');

        rerender(<CTAButton {...defaultProps} type="secondary" />);
        expect(screen.getByRole('button')).toHaveClass('secondary');
    });
});
