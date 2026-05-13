/**
 * Canvas drawing utility for capturing hand-drawn mathematical expressions
 * Captures strokes as sequences of (x, y, timestamp) points
 */

class CanvasDrawing {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.traces = []; // Array of strokes
        this.currentTrace = []; // Current stroke being drawn
        this.startTime = 0;

        this.setupCanvas();
        this.attachEventListeners();
    }

    setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;

        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#000000';
    }

    attachEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.startStroke(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.endStroke());
        this.canvas.addEventListener('mouseleave', () => this.endStroke());

        this.canvas.addEventListener('touchstart', (e) => this.startStroke(e));
        this.canvas.addEventListener('touchmove', (e) => this.draw(e));
        this.canvas.addEventListener('touchend', () => this.endStroke());
    }

    getCoordinates(event) {
        const rect = this.canvas.getBoundingClientRect();
        let x, y;

        if (event.touches) {
            x = event.touches[0].clientX - rect.left;
            y = event.touches[0].clientY - rect.top;
        } else {
            x = event.clientX - rect.left;
            y = event.clientY - rect.top;
        }

        return { x, y };
    }

    startStroke(event) {
        event.preventDefault();
        this.isDrawing = true;
        this.currentTrace = [];
        this.startTime = Date.now();

        const { x, y } = this.getCoordinates(event);
        const t = 0;

        this.currentTrace.push([x, y, t]);
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
    }

    draw(event) {
        if (!this.isDrawing) return;

        event.preventDefault();
        const { x, y } = this.getCoordinates(event);
        const t = Date.now() - this.startTime;

        this.currentTrace.push([x, y, t]);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    endStroke() {
        if (!this.isDrawing || this.currentTrace.length === 0) return;

        this.isDrawing = false;
        this.ctx.closePath();

        if (this.currentTrace.length > 1) {
            this.traces.push([...this.currentTrace]);
        }

        this.currentTrace = [];
    }

    /**
     * Get the captured traces in the format expected by the model
     * @returns {Array} Array of traces, each trace is an array of [x, y, t] points
     */
    getTraces() {
        return this.traces;
    }

    /**
     * Check if any strokes have been drawn
     * @returns {boolean}
     */
    hasStrokes() {
        return this.traces.length > 0;
    }

    /**
     * Clear the canvas and reset traces
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.traces = [];
        this.currentTrace = [];
        this.isDrawing = false;
    }
}
