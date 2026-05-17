export type CanvasMode = 'draw' | 'select' | 'erase' | 'pointer';

export interface LatexObject {
    id: string;
    latex: string;
    x: number;
    y: number;
    width: number;
    height: number;
}

export class CanvasDrawing {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private isActive = false;
    private traces: number[][][] = [];
    private redoStack: number[][][] = [];
    private currentTrace: number[][] = [];
    private startTime = 0;
    
    private mode: CanvasMode = 'draw';
    private penOnlyMode = false;
    private selectionStart: { x: number, y: number } | null = null;
    private selectionEnd: { x: number, y: number } | null = null;
    private selectedTraceIndices: Set<number> = new Set();
    
    private latexObjects: LatexObject[] = [];
    private backgroundImage: HTMLImageElement | null = null;
    private draggingObjectId: string | null = null;
    private resizingObjectId: string | null = null;
    private dragOffset = { x: 0, y: 0 };

    private onSelectionComplete?: (traces: number[][][]) => void;
    private onSelectionChange?: (count: number) => void;

    constructor(canvasElement: HTMLCanvasElement) {
        this.canvas = canvasElement;
        const context = this.canvas.getContext('2d');
        if (!context) throw new Error("Could not get 2D context");
        this.ctx = context;
        this.setupCanvas();
    }

    private setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#000000';
    }

    public setMode(mode: CanvasMode) {
        this.mode = mode;
        this.clearSelection();
        this.redraw();
    }

    public setPenOnlyMode(enabled: boolean) {
        this.penOnlyMode = enabled;
    }

    public setOnSelectionComplete(callback: (traces: number[][][]) => void) {
        this.onSelectionComplete = callback;
    }

    public setOnSelectionChange(callback: (count: number) => void) {
        this.onSelectionChange = callback;
    }

    private isEventAllowed(e: PointerEvent): boolean {
        if (!this.penOnlyMode) return true;
        
        // If penOnlyMode is active, only allow pen for draw, select, erase
        if (['draw', 'select', 'erase'].includes(this.mode)) {
            return e.pointerType === 'pen';
        }
        
        // pointer mode (move/resize) is always allowed (finger is okay)
        return true;
    }

    public handlePointerDown(e: PointerEvent) {
        if (!this.isEventAllowed(e)) return;
        
        // Ensure it's a real touch/press, not just hover proximity
        // e.buttons === 1 is primary (tip contact)
        // e.pressure > 0 is another check for contact
        if (e.pointerType === 'pen' && e.buttons === 0 && e.pressure === 0) return;

        const coords = this.getCoordinates(e);
        
        if (this.mode === 'pointer') {
            const handle = this.getResizeHandleAt(coords.x, coords.y);
            if (handle) {
                this.resizingObjectId = handle.id;
                this.isActive = true;
                this.redraw();
                return;
            }

            const clickedObj = this.getLatexObjectAt(coords.x, coords.y);
            if (clickedObj) {
                this.draggingObjectId = clickedObj.id;
                this.dragOffset = { x: coords.x - clickedObj.x, y: coords.y - clickedObj.y };
                this.isActive = true;
                this.redraw();
                return;
            }
        }

        if (this.mode === 'draw') this.startStroke(coords);
        else if (this.mode === 'select') this.startSelection(coords);
        else if (this.mode === 'erase') this.startErasing(coords);
    }

    public handlePointerMove(e: PointerEvent) {
        if (!this.isActive && !['erase'].includes(this.mode)) return;
        if (!this.isEventAllowed(e)) return;

        const events = (e as any).getCoalescedEvents?.() || [e];
        let needsRedraw = false;
        
        for (const ev of events) {
            const coords = this.getCoordinates(ev);
            
            if (this.resizingObjectId) {
                // Resize and Drag move their own objects and then call redraw
                this.updateObjectSize(coords);
            } else if (this.draggingObjectId) {
                this.updateObjectPosition(coords);
            } else if (this.mode === 'draw') {
                // Draw is optimized: it just draws a line segment on top, no clear/redraw
                this.draw(coords);
            } else if (this.mode === 'select') {
                // Select needs a redraw to show the marquee. 
                // We'll update the logic to only redraw once after the loop.
                this.updateSelectionNoRedraw(coords);
                needsRedraw = true;
            } else if (this.mode === 'erase') {
                // Erase needs redraw to show traces disappearing.
                if (this.eraseAtNoRedraw(coords)) needsRedraw = true;
            }
        }

        if (needsRedraw) {
            this.redraw();
        }
    }

    private updateSelectionNoRedraw({ x, y }: { x: number, y: number }) {
        if (!this.isActive || this.mode !== 'select') return;
        this.selectionEnd = { x, y };
        this.updateSelectedTraces();
    }

    private eraseAtNoRedraw({ x, y }: { x: number, y: number }): boolean {
        if (!this.isActive || this.mode !== 'erase') return false;
        const eraserSize = 10;
        const initialTracesCount = this.traces.length;
        this.traces = this.traces.filter(trace => 
            !trace.some(([px, py]) => 
                px >= x - eraserSize && px <= x + eraserSize && 
                py >= y - eraserSize && py <= y + eraserSize
            )
        );

        const initialObjectsCount = this.latexObjects.length;
        this.latexObjects = this.latexObjects.filter(obj => 
            !(x >= obj.x - 5 && x <= obj.x + obj.width + 5 && 
              y >= obj.y - 5 && y <= obj.y + obj.height + 5)
        );

        if (this.traces.length !== initialTracesCount || this.latexObjects.length !== initialObjectsCount) {
            this.redoStack = []; 
            return true;
        }
        return false;
    }

    public handlePointerUp() {
        this.draggingObjectId = null;
        this.resizingObjectId = null;

        if (this.mode === 'draw') this.endStroke();
        else if (this.mode === 'select') this.endSelection();
        else this.isActive = false;
    }

    private getCoordinates(event: PointerEvent): { x: number, y: number } {
        const rect = this.canvas.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    private startStroke({ x, y }: { x: number, y: number }) {
        this.isActive = true;
        this.currentTrace = [[x, y, 0]];
        this.startTime = Date.now();
    }

    private draw({ x, y }: { x: number, y: number }) {
        if (!this.isActive || this.mode !== 'draw' || this.currentTrace.length === 0) return;
        
        const lastPoint = this.currentTrace[this.currentTrace.length - 1];
        this.currentTrace.push([x, y, Date.now() - this.startTime]);
        
        // Only draw the NEW segment to keep performance constant regardless of stroke length
        this.ctx.beginPath();
        this.ctx.moveTo(lastPoint[0], lastPoint[1]);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    private endStroke() {
        if (!this.isActive || this.currentTrace.length < 2) {
            this.isActive = false;
            return;
        }
        this.isActive = false;
        this.traces.push([...this.currentTrace]);
        this.redoStack = []; // Clear redo stack on new stroke
        this.currentTrace = [];
        this.redraw();
    }

    private startSelection({ x, y }: { x: number, y: number }) {
        this.isActive = true;
        this.selectedTraceIndices.clear();
        this.onSelectionChange?.(0);
        this.selectionStart = { x, y };
        this.selectionEnd = { x, y };
    }

    private updateSelection({ x, y }: { x: number, y: number }) {
        if (!this.isActive || this.mode !== 'select') return;
        this.selectionEnd = { x, y };
        this.updateSelectedTraces();
        this.redraw();
    }

    private endSelection() {
        if (!this.isActive || !this.selectionStart || !this.selectionEnd) return;
        this.isActive = false;
        const selected = this.getSelectedTraces();
        console.log("Selection ended. Traces found:", selected.length);
        if (selected.length > 0) {
            console.log("Triggering onSelectionComplete callback");
            this.onSelectionComplete?.(selected);
        }
        
        this.selectionStart = null;
        this.selectionEnd = null;
        this.redraw();
    }

    private startErasing(coords: { x: number, y: number }) {
        this.isActive = true;
        this.eraseAt(coords);
    }

    private eraseAt({ x, y }: { x: number, y: number }) {
        if (!this.isActive || this.mode !== 'erase') return;
        const eraserSize = 10;
        const initialTracesCount = this.traces.length;
        this.traces = this.traces.filter(trace => 
            !trace.some(([px, py]) => 
                px >= x - eraserSize && px <= x + eraserSize && 
                py >= y - eraserSize && py <= y + eraserSize
            )
        );

        const initialObjectsCount = this.latexObjects.length;
        this.latexObjects = this.latexObjects.filter(obj => 
            !(x >= obj.x - 5 && x <= obj.x + obj.width + 5 && 
              y >= obj.y - 5 && y <= obj.y + obj.height + 5)
        );

        if (this.traces.length !== initialTracesCount || this.latexObjects.length !== initialObjectsCount) {
            this.redoStack = []; // Clear redo stack on erase
            this.redraw();
        }
    }

    private updateSelectedTraces() {
        if (!this.selectionStart || !this.selectionEnd) return;
        const x1 = Math.min(this.selectionStart.x, this.selectionEnd.x);
        const y1 = Math.min(this.selectionStart.y, this.selectionEnd.y);
        const x2 = Math.max(this.selectionStart.x, this.selectionEnd.x);
        const y2 = Math.max(this.selectionStart.y, this.selectionEnd.y);

        this.selectedTraceIndices.clear();
        this.traces.forEach((trace, index) => {
            if (trace.some(([px, py]) => px >= x1 && px <= x2 && py >= y1 && py <= y2)) {
                this.selectedTraceIndices.add(index);
            }
        });
        
        this.onSelectionChange?.(this.selectedTraceIndices.size);
    }

    private getSelectedTraces(): number[][][] {
        return this.traces.filter((_, index) => this.selectedTraceIndices.has(index));
    }

    public undo() {
        if (this.traces.length > 0) {
            const popped = this.traces.pop();
            if (popped) this.redoStack.push(popped);
            this.redraw();
        }
    }

    public redo() {
        if (this.redoStack.length > 0) {
            const popped = this.redoStack.pop();
            if (popped) this.traces.push(popped);
            this.redraw();
        }
    }

    public canRedo() {
        return this.redoStack.length > 0;
    }

    public redraw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (this.backgroundImage) {
            this.ctx.drawImage(this.backgroundImage, 0, 0);
        }

        this.traces.forEach((trace, index) => {
            if (trace.length < 2) return;
            this.ctx.beginPath();
            this.ctx.strokeStyle = this.selectedTraceIndices.has(index) ? '#aa3bff' : '#000000';
            this.ctx.lineWidth = this.selectedTraceIndices.has(index) ? 3 : 2;
            this.ctx.moveTo(trace[0][0], trace[0][1]);
            for (let i = 1; i < trace.length; i++) this.ctx.lineTo(trace[i][0], trace[i][1]);
            this.ctx.stroke();
        });

        if (this.mode === 'select' && this.selectionStart && this.selectionEnd) {
            const x1 = Math.min(this.selectionStart.x, this.selectionEnd.x);
            const y1 = Math.min(this.selectionStart.y, this.selectionEnd.y);
            const w = Math.abs(this.selectionStart.x - this.selectionEnd.x);
            const h = Math.abs(this.selectionStart.y - this.selectionEnd.y);
            this.ctx.setLineDash([5, 5]);
            this.ctx.strokeStyle = '#007BFF';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(x1, y1, w, h);
            this.ctx.fillStyle = 'rgba(0, 123, 255, 0.05)';
            this.ctx.fillRect(x1, y1, w, h);
            this.ctx.setLineDash([]);
        }

        if (this.mode === 'pointer') {
            this.latexObjects.forEach(obj => {
                this.ctx.strokeStyle = '#aa3bff';
                this.ctx.setLineDash([2, 2]);
                this.ctx.strokeRect(obj.x, obj.y, obj.width, obj.height);
                this.ctx.setLineDash([]);
                
                // Resize handle (bottom right)
                this.ctx.fillStyle = '#aa3bff';
                this.ctx.fillRect(obj.x + obj.width - 6, obj.y + obj.height - 6, 12, 12);
            });
        }
    }

    private getLatexObjectAt(x: number, y: number): LatexObject | null {
        return this.latexObjects.find(obj => 
            x >= obj.x && x <= obj.x + obj.width && 
            y >= obj.y && y <= obj.y + obj.height
        ) || null;
    }

    private getResizeHandleAt(x: number, y: number): { id: string } | null {
        const found = this.latexObjects.find(obj => {
            const handleX = obj.x + obj.width;
            const handleY = obj.y + obj.height;
            return x >= handleX - 10 && x <= handleX + 10 && y >= handleY - 10 && y <= handleY + 10;
        });
        return found ? { id: found.id } : null;
    }

    private updateObjectPosition({ x, y }: { x: number, y: number }) {
        const obj = this.latexObjects.find(o => o.id === this.draggingObjectId);
        if (obj) {
            obj.x = x - this.dragOffset.x;
            obj.y = y - this.dragOffset.y;
            this.redraw();
        }
    }

    private updateObjectSize({ x, y }: { x: number, y: number }) {
        const obj = this.latexObjects.find(o => o.id === this.resizingObjectId);
        if (obj) {
            obj.width = Math.max(20, x - obj.x);
            obj.height = Math.max(20, y - obj.y);
            this.redraw();
        }
    }

    public replaceSelectedWithLatex(latex: string) {
        if (this.selectedTraceIndices.size === 0) return;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.traces.forEach((trace, index) => {
            if (this.selectedTraceIndices.has(index)) {
                trace.forEach(([tx, ty]) => {
                    minX = Math.min(minX, tx);
                    minY = Math.min(minY, ty);
                    maxX = Math.max(maxX, tx);
                    maxY = Math.max(maxY, ty);
                });
            }
        });

        const padding = 10;
        const newObj: LatexObject = {
            id: Date.now().toString(),
            latex,
            x: minX - padding,
            y: minY - padding,
            width: (maxX - minX) + (padding * 2),
            height: (maxY - minY) + (padding * 2)
        };

        this.traces = this.traces.filter((_, i) => !this.selectedTraceIndices.has(i));
        this.latexObjects.push(newObj);
        this.redoStack = []; // Clear redo stack on replace
        this.clearSelection();
        this.redraw();
        return newObj;
    }

    public replaceAllWithLatex(latex: string) {
        if (this.traces.length === 0) return;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.traces.forEach(trace => {
            trace.forEach(([tx, ty]) => {
                minX = Math.min(minX, tx);
                minY = Math.min(minY, ty);
                maxX = Math.max(maxX, tx);
                maxY = Math.max(maxY, ty);
            });
        });

        const padding = 10;
        const newObj: LatexObject = {
            id: Date.now().toString(),
            latex,
            x: minX - padding,
            y: minY - padding,
            width: (maxX - minX) + (padding * 2),
            height: (maxY - minY) + (padding * 2)
        };

        this.traces = [];
        this.latexObjects.push(newObj);
        this.redoStack = []; // Clear redo stack on replace
        this.clearSelection();
        this.redraw();
        return newObj;
    }

    public getLatexObjects() { return this.latexObjects; }
    public clearSelection() {
        this.selectionStart = null;
        this.selectionEnd = null;
        this.selectedTraceIndices.clear();
        this.onSelectionChange?.(0);
        this.redraw();
    }
    public getTraces() { return this.traces; }
    public hasStrokes() { return this.traces.length > 0; }
    public clear() {
        this.traces = [];
        this.latexObjects = [];
        this.currentTrace = [];
        this.redoStack = []; // Clear redo stack on clear
        this.isActive = false;
        this.clearSelection();
    }
    public setTraces(traces: number[][][]) {
        this.traces = traces;
        this.redraw();
    }

    public setLatexObjects(objects: LatexObject[]) {
        this.latexObjects = objects;
        this.redraw();
    }

    public setBackgroundImage(image: HTMLImageElement | null) {
        this.backgroundImage = image;
        this.redraw();
    }

    public resize() {
        const rect = this.canvas.getBoundingClientRect();
        if (this.canvas.width !== rect.width || this.canvas.height !== rect.height) {
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
            this.setupCanvas();
            this.redraw();
        }
    }
}
