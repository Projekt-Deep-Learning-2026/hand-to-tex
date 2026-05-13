const VOCAB_URL = './assets/vocab.json';
const ENCODER_URL = 'https://huggingface.co/m4jkiuwr/htt-mini/resolve/main/encoder.onnx?download=true';
const DECODER_URL = 'https://huggingface.co/m4jkiuwr/htt-mini/resolve/main/decoder_step.onnx?download=true';

let encoderSession = null;
let decoderSession = null;
let id2token = [];
let PAD_IDX = 0, SOS_IDX = 1, EOS_IDX = 2;
const EPS = 1e-6;

const loadBtn = document.getElementById('load-btn');
const initStatus = document.getElementById('init-status');
const inkmlInput = document.getElementById('inkml-input');
const outputDiv = document.getElementById('output');
const previewContainer = document.getElementById('preview-container');
const drawingCanvas = document.getElementById('drawing-canvas');
const recognizeBtn = document.getElementById('recognize-btn');
const clearBtn = document.getElementById('clear-btn');

let canvasDrawing = null;

/**
 * Render LaTeX string using KaTeX
 * @param {string} latex - LaTeX string to render
 */
function renderLatexPreview(latex) {
    previewContainer.innerHTML = '';
    try {
        katex.render(latex, previewContainer, { displayMode: true, throwOnError: false });
    } catch (error) {
        console.warn('KaTeX rendering error:', error);
        previewContainer.innerText = 'Preview unavailable';
    }
}

async function loadVocab() {
    initStatus.innerText = "Loading vocabulary...";
    const response = await fetch(VOCAB_URL);
    const vocabData = await response.json();

    id2token = [];
    for (const category in vocabData) {
        id2token.push(...vocabData[category]);
    }

    PAD_IDX = id2token.indexOf("<PAD>");
    SOS_IDX = id2token.indexOf("<SOS>");
    EOS_IDX = id2token.indexOf("<EOS>");

    console.log(`Vocab loaded with ${id2token.length} tokens. SOS: ${SOS_IDX}, EOS: ${EOS_IDX}, PAD: ${PAD_IDX}`);
}

async function loadModels() {
    initStatus.innerText = "Loading the encoder...";
    encoderSession = await ort.InferenceSession.create(ENCODER_URL, { executionProviders: ['wasm'] });

    initStatus.innerText = "Loading the decoder...";
    decoderSession = await ort.InferenceSession.create(DECODER_URL, { executionProviders: ['wasm'] });
}

loadBtn.addEventListener('click', async () => {
    try {
        loadBtn.disabled = true;
        await loadVocab();
        await loadModels();
        initStatus.innerText = "Models and vocabulary loaded successfully";
        initStatus.style.color = "green";
        inkmlInput.disabled = false;
        recognizeBtn.disabled = false;
        clearBtn.disabled = false;
        canvasDrawing = new CanvasDrawing(drawingCanvas);
    } catch (error) {
        console.error("Initialization error:", error);
        initStatus.innerText = "Initialization error. Check console.";
        initStatus.style.color = "red";
    }
});

function parseInkml(xmlString) {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlString, "text/xml");
    let traceNodes = xmlDoc.getElementsByTagNameNS("http://www.w3.org/2003/InkML", "trace");
    if (traceNodes.length === 0) traceNodes = xmlDoc.getElementsByTagName("trace");

    const traces = [];
    for (let node of traceNodes) {
        const text = node.textContent.trim();
        if (!text) continue;
        const pointsStr = text.split(',');
        const trace = [];
        for (let ptStr of pointsStr) {
            const coords = ptStr.trim().split(/\s+/).map(Number);
            if (coords.length >= 3 && !isNaN(coords[0])) {
                trace.push([coords[0], coords[1], coords[2]]);
            }
        }
        if (trace.length > 0) traces.push(trace);
    }
    return traces;
}

function extractFeatures(traces) {
    if (traces.length === 0) return { flatData: new Float32Array(0), numPoints: 0, numFeatures: 12 };

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minT = Infinity, maxT = -Infinity;

    for (let trace of traces) {
        for (let pt of trace) {
            let [x, y, t] = pt;
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (t < minT) minT = t; if (t > maxT) maxT = t;
        }
    }

    const xyRange = Math.max(maxX - minX, maxY - minY) + EPS;
    const tRange = maxT - minT + EPS;
    const exprYSpan = (maxY - minY) + EPS;

    let featuresList = [];

    for (let trace of traces) {
        let traceMinY = Infinity, traceMaxY = -Infinity;
        for (let pt of trace) {
            if (pt[1] < traceMinY) traceMinY = pt[1];
            if (pt[1] > traceMaxY) traceMaxY = pt[1];
        }
        let traceYCenter = 0.5 * (traceMinY + traceMaxY);
        let traceYSpan = traceMaxY - traceMinY;

        let speeds = [], uxs = [], uys = [], dists = [];

        for (let i = 0; i < trace.length; i++) {
            let pt = trace[i];
            let xNorm = (pt[0] - minX) / xyRange;
            let yNorm = (pt[1] - minY) / xyRange;
            let tNorm = (pt[2] - minT) / tRange;
            let yCenterRel = (traceYCenter - minY) / exprYSpan;
            let ySpanRel = traceYSpan / exprYSpan;

            let dx = 0, dy = 0, dt = 0, speed = 0, dist = 0, ux = 0, uy = 0;

            if (i > 0) {
                let prevPt = trace[i - 1];
                let prevXNorm = (prevPt[0] - minX) / xyRange;
                let prevYNorm = (prevPt[1] - minY) / xyRange;
                let prevTNorm = (prevPt[2] - minT) / tRange;

                dx = xNorm - prevXNorm;
                dy = yNorm - prevYNorm;
                dt = tNorm - prevTNorm;
                dist = Math.hypot(dx, dy);
                speed = dt > EPS ? dist / dt : 0;
                ux = dist > EPS ? dx / dist : 0;
                uy = dist > EPS ? dy / dist : 0;
            }

            speeds.push(speed); uxs.push(ux); uys.push(uy); dists.push(dist);

            let curve = 0, accTan = 0;
            if (i > 0) {
                let prevSpeed = speeds[i - 1];
                accTan = dt > EPS ? (speed - prevSpeed) / dt : 0;
                if (i > 1) {
                    let prevUx = uxs[i - 1], prevUy = uys[i - 1];
                    let dTheta = Math.atan2(prevUx * uy - prevUy * ux, prevUx * ux + prevUy * uy);
                    curve = dist > EPS ? dTheta / dist : 0;
                }
            }

            let isStrokeStart = i === 0 ? 1.0 : 0.0;
            featuresList.push([xNorm, yNorm, tNorm, dx, dy, dt, speed, curve, accTan, isStrokeStart, yCenterRel, ySpanRel]);
        }
    }

    const colsToNorm = [3, 4, 5, 6, 7, 8];
    for (let col of colsToNorm) {
        let sum = 0;
        for (let row of featuresList) sum += row[col];
        let mean = sum / featuresList.length;

        let varianceSum = 0;
        for (let row of featuresList) varianceSum += Math.pow(row[col] - mean, 2);
        let std = Math.sqrt(varianceSum / featuresList.length) + EPS;

        for (let row of featuresList) {
            row[col] = Math.max(-5.0, Math.min(5.0, (row[col] - mean) / std));
        }
    }

    const numPoints = featuresList.length;
    const flatFeatures = new Float32Array(numPoints * 12);
    let ptr = 0;
    for (let row of featuresList) {
        for (let val of row) flatFeatures[ptr++] = val;
    }

    return { flatData: flatFeatures, numPoints: numPoints, numFeatures: 12 };
}

async function runInference(flatData, numPoints, numFeatures) {
    const batchSize = 1;
    const srcDims = [batchSize, numPoints, numFeatures];

    const srcFeaturesTensor = new ort.Tensor('float32', flatData, srcDims);
    const srcLengthsTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(numPoints)]), [batchSize]);

    const encFeeds = { "src": srcFeaturesTensor, "src_lengths": srcLengthsTensor };
    const encResults = await encoderSession.run(encFeeds);

    let memK = encResults.mem_k;
    let memV = encResults.mem_v;
    const memMask = encResults.mem_mask;

    const numLayers = memK.dims[0];
    const numHeads = memK.dims[2];
    const headDim = memK.dims[4];

    let selfK = new ort.Tensor('float32', new Float32Array(0), [numLayers, batchSize, numHeads, 0, headDim]);
    let selfV = new ort.Tensor('float32', new Float32Array(0), [numLayers, batchSize, numHeads, 0, headDim]);

    let step = new ort.Tensor('int64', new BigInt64Array([0n]), [1]);
    let tgtLast = new ort.Tensor('int64', new BigInt64Array([BigInt(SOS_IDX)]), [batchSize, 1]);

    const maxLen = 150;
    const generatedTokenIds = [];

    for (let i = 1; i < maxLen; i++) {
        const decFeeds = {
            "tgt_last": tgtLast,
            "mem_k": memK,
            "mem_v": memV,
            "memory_key_padding_mask": memMask,
            "step": step,
            "self_k": selfK,
            "self_v": selfV
        };

        const decResults = await decoderSession.run(decFeeds);
        const logitsData = decResults.logits.data;

        selfK = decResults.self_k_out;
        selfV = decResults.self_v_out;

        let bestToken = 0;
        let maxLogit = logitsData[0];
        for (let j = 1; j < logitsData.length; j++) {
            if (logitsData[j] > maxLogit) {
                maxLogit = logitsData[j];
                bestToken = j;
            }
        }

        if (bestToken === EOS_IDX) break;
        generatedTokenIds.push(bestToken);

        tgtLast = new ort.Tensor('int64', new BigInt64Array([BigInt(bestToken)]), [batchSize, 1]);
        step = new ort.Tensor('int64', new BigInt64Array([BigInt(i)]), [1]);
    }

    return generatedTokenIds;
}

clearBtn.addEventListener('click', () => {
    if (canvasDrawing) {
        canvasDrawing.clear();
        outputDiv.innerText = "No results yet.";
        previewContainer.innerText = "Rendered preview will appear here";
    }
});

recognizeBtn.addEventListener('click', async () => {
    if (!canvasDrawing || !canvasDrawing.hasStrokes()) {
        outputDiv.innerText = "Please draw something first.";
        return;
    }

    outputDiv.innerText = "Processing drawing...";
    recognizeBtn.disabled = true;

    try {
        const traces = canvasDrawing.getTraces();
        const { flatData, numPoints, numFeatures } = extractFeatures(traces);

        if (numPoints === 0) {
            throw new Error("No valid drawing data captured.");
        }

        const tokenIds = await runInference(flatData, numPoints, numFeatures);
        const latexTokens = tokenIds
            .filter(id => id !== PAD_IDX && id !== SOS_IDX)
            .map(id => id2token[id]);

        const latexOutput = latexTokens.join(" ");
        outputDiv.innerText = latexOutput;
        renderLatexPreview(latexOutput);
    } catch (error) {
        console.error("Inference error:", error);
        outputDiv.innerText = "Recognition error: " + error.message;
    } finally {
        recognizeBtn.disabled = false;
    }
});

inkmlInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    outputDiv.innerText = "Processing...";

    const reader = new FileReader();
    reader.onload = async (e) => {
        try {
            const content = e.target.result;

            const traces = parseInkml(content);
            if (traces.length === 0) throw new Error("No strokes found in file.");

            const { flatData, numPoints, numFeatures } = extractFeatures(traces);

            const tokenIds = await runInference(flatData, numPoints, numFeatures);

            const latexTokens = tokenIds
                .filter(id => id !== PAD_IDX && id !== SOS_IDX)
                .map(id => id2token[id]);

            const latexOutput = latexTokens.join(" ");
            outputDiv.innerText = latexOutput;
            renderLatexPreview(latexOutput);

        } catch (error) {
            console.error("Inference error:", error);
            outputDiv.innerText = "Recognition error: " + error.message;
        }
    };
    reader.readAsText(file);
});