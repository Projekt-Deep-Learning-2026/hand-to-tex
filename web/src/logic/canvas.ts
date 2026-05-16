/**
 * Canvas drawing utility for capturing hand-drawn mathematical expressions
 * Captures strokes as sequences of (x, y, timestamp) points
 */

export class CanvasDrawing {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private isDrawing: boolean = false;
    private traces: number[][][] = []; // Array of strokes
    private currentTrace: number[][] = []; // Current stroke being drawn
    private startTime: number = 0;

    constructor(canvasElement: HTMLCanvasElement) {
        this.canvas = canvasElement;
        const context = this.canvas.getContext('2d');
        if (!context) throw new Error("Could not get 2D context");
        this.ctx = context;

        this.setupCanvas();
    }

    private setupCanvas() {
        // Adjust for DPI if needed, but keeping it simple for now
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;

        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#000000';
    }

    public handleMouseDown(e: React.MouseEvent | MouseEvent) {
        this.startStroke(this.getCoordinates(e));
    }

    public handleMouseMove(e: React.MouseEvent | MouseEvent) {
        this.draw(this.getCoordinates(e));
    }

    public handleMouseUp() {
        this.endStroke();
    }

    public handleTouchStart(e: React.TouchEvent | TouchEvent) {
        e.preventDefault();
        this.startStroke(this.getCoordinates(e));
    }

    public handleTouchMove(e: React.TouchEvent | TouchEvent) {
        e.preventDefault();
        this.draw(this.getCoordinates(e));
    }

    public handleTouchEnd() {
        this.endStroke();
    }

    private getCoordinates(event: React.MouseEvent | MouseEvent | React.TouchEvent | TouchEvent): { x: number, y: number } {
        const rect = this.canvas.getBoundingClientRect();
        let x: number, y: number;

        if ('touches' in event) {
            x = event.touches[0].clientX - rect.left;
            y = event.touches[0].clientY - rect.top;
        } else {
            x = (event as MouseEvent).clientX - rect.left;
            y = (event as MouseEvent).clientY - rect.top;
        }

        return { x, y };
    }

    private startStroke({ x, y }: { x: number, y: number }) {
        this.isDrawing = true;
        this.currentTrace = [];
        this.startTime = Date.now();

        const t = 0;
        this.currentTrace.push([x, y, t]);
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
    }

    private draw({ x, y }: { x: number, y: number }) {
        if (!this.isDrawing) return;

        const t = Date.now() - this.startTime;
        this.currentTrace.push([x, y, t]);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    private endStroke() {
        if (!this.isDrawing || this.currentTrace.length === 0) return;

        this.isDrawing = false;
        this.ctx.closePath();

        if (this.currentTrace.length > 1) {
            this.traces.push([...this.currentTrace]);
        }

        this.currentTrace = [];
    }

    public getTraces(): number[][][] {
        return this.traces;
    }

    public hasStrokes(): boolean {
        return this.traces.length > 0;
    }

    public clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.traces = [];
        this.currentTrace = [];
        this.isDrawing = false;
    }

    public resize() {
        const rect = this.canvas.getBoundingClientRect();
        if (this.canvas.width !== rect.width || this.canvas.height !== rect.height) {
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
            this.setupCanvas();
            // Optional: redraw traces if you want to support resizing without clearing
            // For now, let's just clear
            this.clear();
        }
    }
}
