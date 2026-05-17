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
    
    // Optimization: Layered Rendering
    // We use an offscreen canvas to cache all "static" elements (previous traces, background)
    // This way, while drawing, we only draw the current segment on top of the cached image
    // instead of redrawing hundreds of existing traces every frame.
    private offscreenCanvas: HTMLCanvasElement;
    private offscreenCtx: CanvasRenderingContext2D;

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
    private onObjectsChange?: (objects: LatexObject[]) => void;

    constructor(canvasElement: HTMLCanvasElement) {
        this.canvas = canvasElement;
        const context = this.canvas.getContext('2d', { alpha: false }); // Optimization: disable alpha
        if (!context) throw new Error("Could not get 2D context");
        this.ctx = context;

        // Initialize offscreen buffer
        this.offscreenCanvas = document.createElement('canvas');
        const offscreenCtx = this.offscreenCanvas.getContext('2d', { alpha: false });
        if (!offscreenCtx) throw new Error("Could not get offscreen context");
        this.offscreenCtx = offscreenCtx;

        this.setupCanvas();
    }

    private setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        // Scale for High DPI screens (Retina)
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.offscreenCanvas.width = rect.width * dpr;
        this.offscreenCanvas.height = rect.height * dpr;
        
        this.ctx.resetTransform();
        this.offscreenCtx.resetTransform();
        
        this.ctx.scale(dpr, dpr);
        this.offscreenCtx.scale(dpr, dpr);

        [this.ctx, this.offscreenCtx].forEach(c => {
            c.lineCap = 'round';
            c.lineJoin = 'round';
            c.lineWidth = 2;
            c.strokeStyle = '#000000';
        });
        
        this.updateStaticBuffer();
    }

    // Caches all non-moving parts (old strokes, background) to the offscreen buffer
    private updateStaticBuffer() {
        const w = this.offscreenCanvas.width;
        const h = this.offscreenCanvas.height;
        
        this.offscreenCtx.save();
        this.offscreenCtx.setTransform(1, 0, 0, 1, 0, 0);
        this.offscreenCtx.fillStyle = '#ffffff';
        this.offscreenCtx.fillRect(0, 0, w, h);
        this.offscreenCtx.restore();

        if (this.backgroundImage) {
            this.offscreenCtx.drawImage(this.backgroundImage, 0, 0);
        }

        this.traces.forEach((trace, index) => {
            if (trace.length < 2) return;
            this.offscreenCtx.beginPath();
            this.offscreenCtx.strokeStyle = this.selectedTraceIndices.has(index) ? '#aa3bff' : '#000000';
            this.offscreenCtx.lineWidth = this.selectedTraceIndices.has(index) ? 3 : 2;
            this.offscreenCtx.moveTo(trace[0][0], trace[0][1]);
            for (let i = 1; i < trace.length; i++) this.offscreenCtx.lineTo(trace[i][0], trace[i][1]);
            this.offscreenCtx.stroke();
        });
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

    public setOnObjectsChange(callback: (objects: LatexObject[]) => void) {
        this.onObjectsChange = callback;
    }

    private isEventAllowed(e: PointerEvent): boolean {
        if (!this.penOnlyMode) return true;
        if (['draw', 'select', 'erase'].includes(this.mode)) {
            return e.pointerType === 'pen';
        }
        return true;
    }

    public handlePointerDown(e: PointerEvent) {
        if (!this.isEventAllowed(e)) return;
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

        const events = (e as PointerEvent & { getCoalescedEvents?: () => PointerEvent[] }).getCoalescedEvents?.() || [e];
        let needsRedraw = false;
        
        for (const ev of events) {
            const coords = this.getCoordinates(ev);
            
            if (this.resizingObjectId) {
                this.updateObjectSizeNoRedraw(coords);
                needsRedraw = true;
            } else if (this.draggingObjectId) {
                this.updateObjectPositionNoRedraw(coords);
                needsRedraw = true;
            } else if (this.mode === 'draw') {
                this.draw(coords);
            } else if (this.mode === 'select') {
                this.updateSelectionNoRedraw(coords);
                needsRedraw = true;
            } else if (this.mode === 'erase') {
                if (this.eraseAtNoRedraw(coords)) needsRedraw = true;
            }
        }

        if (needsRedraw) {
            this.redraw();
            if (this.draggingObjectId || this.resizingObjectId) {
                this.onObjectsChange?.([...this.latexObjects]);
            }
        }
    }

    public handlePointerUp() {
        const hadInteraction = this.draggingObjectId !== null || this.resizingObjectId !== null;

        this.draggingObjectId = null;
        this.resizingObjectId = null;

        if (this.mode === 'draw') this.endStroke();
        else if (this.mode === 'select') this.endSelection();
        else this.isActive = false;

        if (hadInteraction) {
            this.onObjectsChange?.([...this.latexObjects]);
        }
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
        this.redoStack = [];
        this.currentTrace = [];
        this.updateStaticBuffer();
        this.redraw();
    }

    private startSelection({ x, y }: { x: number, y: number }) {
        this.isActive = true;
        this.selectedTraceIndices.clear();
        this.onSelectionChange?.(0);
        this.selectionStart = { x, y };
        this.selectionEnd = { x, y };
    }

    private updateSelectionNoRedraw({ x, y }: { x: number, y: number }) {
        if (!this.isActive || this.mode !== 'select') return;
        this.selectionEnd = { x, y };
        this.updateSelectedTraces();
    }

    private endSelection() {
        if (!this.isActive || !this.selectionStart || !this.selectionEnd) return;
        this.isActive = false;
        const selected = this.getSelectedTraces();
        if (selected.length > 0) this.onSelectionComplete?.(selected);
        this.selectionStart = null;
        this.selectionEnd = null;
        this.redraw();
    }

    private startErasing(coords: { x: number, y: number }) {
        this.isActive = true;
        this.eraseAtNoRedraw(coords);
        this.redraw();
    }

    private eraseAtNoRedraw({ x, y }: { x: number, y: number }): boolean {
        if (!this.isActive || this.mode !== 'erase') return false;
        const eraserSize = 15;
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
            this.updateStaticBuffer();
            return true;
        }
        return false;
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
            this.updateStaticBuffer();
            this.redraw();
        }
    }

    public redo() {
        if (this.redoStack.length > 0) {
            const popped = this.redoStack.pop();
            if (popped) this.traces.push(popped);
            this.updateStaticBuffer();
            this.redraw();
        }
    }

    public canRedo() { return this.redoStack.length > 0; }

    public redraw() {
        const dpr = window.devicePixelRatio || 1;
        this.ctx.drawImage(this.offscreenCanvas, 0, 0, this.canvas.width / dpr, this.canvas.height / dpr);

        if (this.mode === 'select') {
            // Draw highlight for selected traces on top of the static buffer
            if (this.selectedTraceIndices.size > 0) {
                this.ctx.save();
                this.ctx.strokeStyle = '#aa3bff';
                this.ctx.lineWidth = 3;
                this.selectedTraceIndices.forEach(index => {
                    const trace = this.traces[index];
                    if (trace.length < 2) return;
                    this.ctx.beginPath();
                    this.ctx.moveTo(trace[0][0], trace[0][1]);
                    for (let i = 1; i < trace.length; i++) this.ctx.lineTo(trace[i][0], trace[i][1]);
                    this.ctx.stroke();
                });
                this.ctx.restore();
            }

            if (this.selectionStart && this.selectionEnd) {
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
        }

        if (this.mode === 'pointer') {
            this.latexObjects.forEach(obj => {
                this.ctx.strokeStyle = '#aa3bff';
                this.ctx.setLineDash([2, 2]);
                this.ctx.strokeRect(obj.x, obj.y, obj.width, obj.height);
                this.ctx.setLineDash([]);
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

    private updateObjectPositionNoRedraw({ x, y }: { x: number, y: number }) {
        const obj = this.latexObjects.find(o => o.id === this.draggingObjectId);
        if (obj) {
            obj.x = x - this.dragOffset.x;
            obj.y = y - this.dragOffset.y;
        }
    }

    private updateObjectSizeNoRedraw({ x, y }: { x: number, y: number }) {
        const obj = this.latexObjects.find(o => o.id === this.resizingObjectId);
        if (obj) {
            obj.width = Math.max(20, x - obj.x);
            obj.height = Math.max(20, y - obj.y);
        }
    }

    public replaceSelectedWithLatex(latex: string) {
        if (this.selectedTraceIndices.size === 0) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.traces.forEach((trace, index) => {
            if (this.selectedTraceIndices.has(index)) {
                trace.forEach(([tx, ty]) => {
                    minX = Math.min(minX, tx); minY = Math.min(minY, ty);
                    maxX = Math.max(maxX, tx); maxY = Math.max(maxY, ty);
                });
            }
        });
        const padding = 10;
        const newObj: LatexObject = {
            id: Date.now().toString(),
            latex,
            x: minX - padding, y: minY - padding,
            width: (maxX - minX) + (padding * 2),
            height: (maxY - minY) + (padding * 2)
        };
        this.traces = this.traces.filter((_, i) => !this.selectedTraceIndices.has(i));
        this.latexObjects.push(newObj);
        this.redoStack = []; 
        this.clearSelection();
        this.updateStaticBuffer();
        this.redraw();
        this.onObjectsChange?.([...this.latexObjects]);
        return newObj;
    }

    public replaceAllWithLatex(latex: string) {
        if (this.traces.length === 0) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.traces.forEach(trace => {
            trace.forEach(([tx, ty]) => {
                minX = Math.min(minX, tx); minY = Math.min(minY, ty);
                maxX = Math.max(maxX, tx); maxY = Math.max(maxY, ty);
            });
        });
        const padding = 10;
        const newObj: LatexObject = {
            id: Date.now().toString(),
            latex,
            x: minX - padding, y: minY - padding,
            width: (maxX - minX) + (padding * 2),
            height: (maxY - minY) + (padding * 2)
        };
        this.traces = [];
        this.latexObjects.push(newObj);
        this.redoStack = []; 
        this.clearSelection();
        this.updateStaticBuffer();
        this.redraw();
        this.onObjectsChange?.([...this.latexObjects]);
        return newObj;
    }

    public updateLatexObject(id: string, latex: string) {
        const obj = this.latexObjects.find(o => o.id === id);
        if (obj) {
            obj.latex = latex;
            this.onObjectsChange?.([...this.latexObjects]);
            this.redraw();
        }
    }

    public deleteLatexObject(id: string) {
        const initialCount = this.latexObjects.length;
        this.latexObjects = this.latexObjects.filter(o => o.id !== id);
        if (this.latexObjects.length !== initialCount) {
            this.onObjectsChange?.([...this.latexObjects]);
            this.redraw();
            return true;
        }
        return false;
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
        this.redoStack = [];
        this.isActive = false;
        this.clearSelection();
        this.updateStaticBuffer();
        this.redraw();
        this.onObjectsChange?.([]);
    }
    public setTraces(traces: number[][][]) {
        this.traces = traces;
        this.updateStaticBuffer();
        this.redraw();
    }
    public setLatexObjects(objects: LatexObject[]) {
        this.latexObjects = objects;
        this.updateStaticBuffer();
        this.redraw();
        this.onObjectsChange?.([...this.latexObjects]);
    }
    public setBackgroundImage(image: HTMLImageElement | null) {
        this.backgroundImage = image;
        this.updateStaticBuffer();
        this.redraw();
    }
    public resize() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        if (this.canvas.width !== rect.width * dpr || this.canvas.height !== rect.height * dpr) {
            this.setupCanvas();
            this.redraw();
        }
    }

    public dispose() {
        // Clear references and remove the offscreen canvas from memory
        this.offscreenCanvas.width = 0;
        this.offscreenCanvas.height = 0;
        
        // We use casting to null to help GC even if the TS types say they are non-nullable
        // as this instance is now being discarded.
        (this.offscreenCanvas as unknown) = null;
        (this.offscreenCtx as unknown) = null;
        (this.canvas as unknown) = null;
        (this.ctx as unknown) = null;
    }
}
