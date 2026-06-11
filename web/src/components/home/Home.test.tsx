import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { Home } from './Home';

describe('Home Component', () => {
    it('renders the title and subtitle', () => {
        const handleSelectView = vi.fn();
        const handleLoadProject = vi.fn();
        
        render(<Home onSelectView={handleSelectView} onLoadProject={handleLoadProject} />);
        
        expect(screen.getByText('Hand-to-TeX')).toBeInTheDocument();
        expect(screen.getByText('Transform your handwritten math into digital LaTeX effortlessly.')).toBeInTheDocument();
    });

    it('calls onSelectView when whiteboard button is clicked', () => {
        const handleSelectView = vi.fn();
        const handleLoadProject = vi.fn();
        
        render(<Home onSelectView={handleSelectView} onLoadProject={handleLoadProject} />);
        
        const whiteboardButton = screen.getByRole('button', { name: /🎨 Open Whiteboard/i });
        fireEvent.click(whiteboardButton);
        
        expect(handleSelectView).toHaveBeenCalledWith('whiteboard');
        expect(handleSelectView).toHaveBeenCalledTimes(1);
    });

    it('renders a file input for loading project and calls onLoadProject', () => {
        const handleSelectView = vi.fn();
        const handleLoadProject = vi.fn();
        
        render(<Home onSelectView={handleSelectView} onLoadProject={handleLoadProject} />);
        
        const loadLabel = screen.getByText('Load Project').closest('label') as HTMLLabelElement;
        const loadInput = loadLabel.querySelector('input') as HTMLInputElement;
        expect(loadInput).toHaveAttribute('type', 'file');
        expect(loadInput).toHaveAttribute('accept', '.json');

        const file = new File(['{}'], 'project.json', { type: 'application/json' });
        fireEvent.change(loadInput, { target: { files: [file] } });
        
        expect(handleLoadProject).toHaveBeenCalledTimes(1);
    });
});
