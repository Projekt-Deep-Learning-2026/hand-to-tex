import { useState } from 'react';
import type { RefObject } from 'react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import type { DrawingCanvasHandle } from '../components/whiteboard/DrawingCanvas';
import type { LatexObject } from '../logic/canvas';
import type { ModeDetails } from '../components/whiteboard/WhiteboardView';

export interface ProjectData {
    traces?: number[][][];
    latexObjects?: LatexObject[];
}

export function useProjectIO(
    canvasRef: RefObject<DrawingCanvasHandle | null>,
    whiteboardWrapperRef: RefObject<HTMLDivElement | null>,
    onToast?: (details: ModeDetails) => void,
    onProjectLoaded?: (data: ProjectData) => void
) {
    const [isExporting, setIsExporting] = useState(false);

    const handleSaveToFile = () => {
        if (!canvasRef.current) return;
        const traces = canvasRef.current.getTraces();
        const latexObjects = canvasRef.current.getLatexObjects();
        
        const data = {
            version: "1.0",
            timestamp: new Date().toISOString(),
            traces,
            latexObjects
        };

        const defaultName = `hand-to-tex-project-${new Date().getTime()}`;
        const fileName = prompt("Enter project filename:", defaultName) || defaultName;
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName.endsWith('.json') ? fileName : `${fileName}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        onToast?.({message: "Project saved"});
    };

    const handleLoadProject = (e: React.ChangeEvent<HTMLInputElement>) => {
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

                if (onProjectLoaded) {
                    onProjectLoaded(data);
                } else if (canvasRef.current) {
                    if (data.traces) canvasRef.current.setTraces(data.traces);
                    if (data.latexObjects) canvasRef.current.setLatexObjects(data.latexObjects);
                }
                
                onToast?.({message: "Project loaded"});
            } catch (err) {
                alert("Error loading project: " + (err as Error).message);
            }
        };
        reader.readAsText(file);
        e.target.value = '';
    };

    const handleExportPDF = async (originalMode: string, setMode: (mode: any) => void) => {
        if (!whiteboardWrapperRef.current) return;
        setIsExporting(true);
        try {
            if (originalMode === 'pointer' || originalMode === 'select') {
                setMode('draw');
            }

            const defaultName = `hand-to-tex-export-${new Date().getTime()}`;
            const fileName = prompt("Enter PDF filename:", defaultName) || defaultName;

            const canvas = await html2canvas(whiteboardWrapperRef.current, {
                useCORS: true, scale: 2, backgroundColor: "#ffffff"
            });
            const imgData = canvas.toDataURL('image/jpeg', 0.95);
            const pdf = new jsPDF({
                orientation: canvas.width > canvas.height ? 'l' : 'p',
                unit: 'px', format: [canvas.width, canvas.height]
            });
            pdf.addImage(imgData, 'JPEG', 0, 0, canvas.width, canvas.height);
            pdf.save(fileName.endsWith('.pdf') ? fileName : `${fileName}.pdf`);
            onToast?.({message: "PDF exported"});
        } catch (err) {
            alert("Error exporting PDF: " + (err as Error).message);
        } finally {
            setMode(originalMode);
            setIsExporting(false);
        }
    };

    return {
        isExporting,
        handleSaveToFile,
        handleLoadProject,
        handleExportPDF
    };
}
