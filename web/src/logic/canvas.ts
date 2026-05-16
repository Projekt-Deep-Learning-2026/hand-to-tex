export type CanvasMode = 'draw' | 'select';

export class CanvasDrawing {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private isActive = false;
    private traces: number[][][] = [];
    private currentTrace: number[][] = [];
    private startTime = 0;
    
    private mode: CanvasMode = 'draw';
    private selectionStart: { x: number, y: number } | null = null;
    private selectionEnd: { x: number, y: number } | null = null;
    private onSelectionComplete?: (traces: number[][][]) => void;

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

    public setOnSelectionComplete(callback: (traces: number[][][]) => void) {
        this.onSelectionComplete = callback;
    }

    public handleMouseDown(e: React.MouseEvent | MouseEvent) {
        const coords = this.getCoordinates(e);
        this.mode === 'draw' ? this.startStroke(coords) : this.startSelection(coords);
    }

    public handleMouseMove(e: React.MouseEvent | MouseEvent) {
        const coords = this.getCoordinates(e);
        this.mode === 'draw' ? this.draw(coords) : this.updateSelection(coords);
    }

    public handleMouseUp() {
        this.mode === 'draw' ? this.endStroke() : this.endSelection();
    }

    public handleTouchStart(e: React.TouchEvent | TouchEvent) {
        e.preventDefault();
        const coords = this.getCoordinates(e);
        this.mode === 'draw' ? this.startStroke(coords) : this.startSelection(coords);
    }

    public handleTouchMove(e: React.TouchEvent | TouchEvent) {
        e.preventDefault();
        const coords = this.getCoordinates(e);
        this.mode === 'draw' ? this.draw(coords) : this.updateSelection(coords);
    }

    public handleTouchEnd() {
        this.mode === 'draw' ? this.endStroke() : this.endSelection();
    }

    private getCoordinates(event: any): { x: number, y: number } {
        const rect = this.canvas.getBoundingClientRect();
        let x: number, y: number;

        if (event.touches?.[0]) {
            x = event.touches[0].clientX - rect.left;
            y = event.touches[0].clientY - rect.top;
        } else if (event.changedTouches?.[0]) {
            x = event.changedTouches[0].clientX - rect.left;
            y = event.changedTouches[0].clientY - rect.top;
        } else {
            x = event.clientX - rect.left;
            y = event.clientY - rect.top;
        }

        return { x, y };
    }

    private startStroke({ x, y }: { x: number, y: number }) {
        this.isActive = true;
        this.currentTrace = [[x, y, 0]];
        this.startTime = Date.now();
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
    }

    private draw({ x, y }: { x: number, y: number }) {
        if (!this.isActive || this.mode !== 'draw') return;
        this.currentTrace.push([x, y, Date.now() - this.startTime]);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    private endStroke() {
        if (!this.isActive || this.currentTrace.length < 2) {
            this.isActive = false;
            return;
        }
        this.isActive = false;
        this.ctx.closePath();
        this.traces.push([...this.currentTrace]);
        this.currentTrace = [];
    }

    private startSelection({ x, y }: { x: number, y: number }) {
        this.isActive = true;
        this.selectionStart = { x, y };
        this.selectionEnd = { x, y };
    }

    private updateSelection({ x, y }: { x: number, y: number }) {
        if (!this.isActive || this.mode !== 'select') return;
        this.selectionEnd = { x, y };
        this.redraw();
    }

    private endSelection() {
        if (!this.isActive || !this.selectionStart || !this.selectionEnd) return;
        this.isActive = false;

        const selectedTraces = this.getSelectedTraces();
        if (selectedTraces.length > 0) {
            this.onSelectionComplete?.(selectedTraces);
        }
    }

    private getSelectedTraces(): number[][][] {
        if (!this.selectionStart || !this.selectionEnd) return [];

        const x1 = Math.min(this.selectionStart.x, this.selectionEnd.x);
        const y1 = Math.min(this.selectionStart.y, this.selectionEnd.y);
        const x2 = Math.max(this.selectionStart.x, this.selectionEnd.x);
        const y2 = Math.max(this.selectionStart.y, this.selectionEnd.y);

        return this.traces.filter(trace => 
            trace.some(([px, py]) => px >= x1 && px <= x2 && py >= y1 && py <= y2)
        );
    }

    private redraw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.beginPath();
        this.ctx.strokeStyle = '#000000';
        this.traces.forEach(trace => {
            if (trace.length < 2) return;
            this.ctx.moveTo(trace[0][0], trace[0][1]);
            for (let i = 1; i < trace.length; i++) {
                this.ctx.lineTo(trace[i][0], trace[i][1]);
            }
        });
        this.ctx.stroke();

        if (this.mode === 'select' && this.selectionStart && this.selectionEnd) {
            const x1 = Math.min(this.selectionStart.x, this.selectionEnd.x);
            const y1 = Math.min(this.selectionStart.y, this.selectionEnd.y);
            const w = Math.abs(this.selectionStart.x - this.selectionEnd.x);
            const h = Math.abs(this.selectionStart.y - this.selectionEnd.y);

            this.ctx.setLineDash([5, 5]);
            this.ctx.strokeStyle = '#007BFF';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(x1, y1, w, h);
            this.ctx.fillStyle = 'rgba(0, 123, 255, 0.1)';
            this.ctx.fillRect(x1, y1, w, h);
            this.ctx.setLineDash([]);
            this.ctx.lineWidth = 2;
        }
    }

    public clearSelection() {
        this.selectionStart = null;
        this.selectionEnd = null;
        this.redraw();
    }

    public getTraces() {
        return this.traces;
    }

    public hasStrokes() {
        return this.traces.length > 0;
    }

    public clear() {
        this.traces = [];
        this.currentTrace = [];
        this.isActive = false;
        this.clearSelection();
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
