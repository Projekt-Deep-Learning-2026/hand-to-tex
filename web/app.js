ort.env.wasm.numThreads = 1;

let encoderSession = null;
let decoderSession = null;
let idToCharVocab = {};
let charToIdVocab = {};

let SOS_TOKEN = null;
let EOS_TOKEN = null;
let PAD_TOKEN = null;
let UNK_TOKEN = null;

const MAX_LENGTH = 150;
const EPS = 1e-6;

function cloneFloat32Tensor(ortTensor) {
    return new ort.Tensor('float32', new Float32Array(ortTensor.data), ortTensor.dims.slice());
}

async function loadVocab() {
    try {
        const response = await fetch('./data/vocab.json');
        const vocabData = await response.json();

        const expectedKeys = [
            "special", "syntactic", "digits", "letters_lower", "letters_upper",
            "blackboard", "punctuation", "greek", "constructs", "diacritics",
            "matrix", "delimiters", "comparisons", "equality", "sets",
            "operators", "arrows", "dots", "other"
        ];

        let currentIndex = 0;
        for (const category of expectedKeys) {
            if (vocabData[category]) {
                const tokens = vocabData[category];
                tokens.forEach(token => {
                    idToCharVocab[currentIndex] = token;
                    charToIdVocab[token] = currentIndex;
                    currentIndex++;
                });
            }
        }

        SOS_TOKEN = charToIdVocab["<SOS>"];
        EOS_TOKEN = charToIdVocab["<EOS>"];
        PAD_TOKEN = charToIdVocab["<PAD>"];
        UNK_TOKEN = charToIdVocab["<UNK>"];
        console.log(`Słownik załadowany poprawnie! Rozmiar: ${currentIndex}`);

    } catch (error) {
        console.error('Błąd podczas ładowania vocab.json:', error);
    }
}

async function loadModels() {
    try {
        await loadVocab();
        encoderSession = await ort.InferenceSession.create('./data/encoder.onnx');
        decoderSession = await ort.InferenceSession.create('./data/decoder.onnx');

        const runBtn = document.getElementById('runBtn');
        runBtn.disabled = false;
        runBtn.innerText = 'Rozpoznaj Równanie';
        console.log('Modele ONNX gotowe do pracy!');
    } catch (error) {
        console.error('Błąd inicjalizacji:', error);
    }
}

async function processJSON() {
    const fileInput = document.getElementById('jsonInput');
    if (fileInput.files.length === 0) return alert('Wybierz plik JSON ze śladami!');

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async function (event) {
        try {
            document.getElementById('result').innerText = 'Przetwarzanie (obliczanie cech i inferencja)...';
            const parsedData = JSON.parse(event.target.result);

            // --- ENKODER ---
            const { src, src_lengths } = prepareInputData(parsedData);
            const encoderFeeds = { "src": src, "src_lengths": src_lengths };
            const encoderResults = await encoderSession.run(encoderFeeds);

            const mem_k = encoderResults["mem_k"];
            const mem_v = encoderResults["mem_v"];
            const mem_mask = encoderResults["mem_mask"];

            // --- PRZYGOTOWANIE DEKODERA ---
            let generatedTokens = [];
            let tgt_last = new ort.Tensor('int64', new BigInt64Array([BigInt(SOS_TOKEN)]), [1, 1]);
            let stepVal = 0n;
            let step = new ort.Tensor('int64', new BigInt64Array([stepVal]), [1]);

            const num_layers = 4;
            const batch_size = 1;
            const num_heads = 8;
            const head_dim = 32;
            const cache_size = num_layers * batch_size * num_heads * 1 * head_dim; // Długość = 1, nie 0

            let self_k = new ort.Tensor('float32', new Float32Array(cache_size).fill(0), [num_layers, batch_size, num_heads, 1, head_dim]);
            let self_v = new ort.Tensor('float32', new Float32Array(cache_size).fill(0), [num_layers, batch_size, num_heads, 1, head_dim]);

            for (let i = 0; i < MAX_LENGTH; i++) {
                const decoderFeeds = {
                    "tgt_last": tgt_last, "mem_k": mem_k, "mem_v": mem_v,
                    "mem_mask": mem_mask, "step": step, "self_k": self_k, "self_v": self_v
                };

                const decoderResults = await decoderSession.run(decoderFeeds);

                const logits = decoderResults["logits"].data;
                let maxProb = -Infinity;
                let nextTokenId = -1;
                for (let j = 0; j < logits.length; j++) {
                    if (logits[j] > maxProb) { maxProb = logits[j]; nextTokenId = j; }
                }

                if (nextTokenId === EOS_TOKEN) break;
                generatedTokens.push(nextTokenId);

                self_k = cloneFloat32Tensor(decoderResults["self_k_out"]);
                self_v = cloneFloat32Tensor(decoderResults["self_v_out"]);

                tgt_last = new ort.Tensor('int64', new BigInt64Array([BigInt(nextTokenId)]), [1, 1]);
                stepVal += 1n;
                step = new ort.Tensor('int64', new BigInt64Array([stepVal]), [1]);
            }

            // --- DEKODOWANIE WYNIKU ---
            const finalLatex = generatedTokens
                .filter(id => id !== PAD_TOKEN && id !== SOS_TOKEN && id !== UNK_TOKEN)
                .map(id => idToCharVocab[id] || "")
                .join("")
                .replace(/ /g, "");

            document.getElementById('result').innerText = `Wynik: ${finalLatex}`;

        } catch (error) {
            console.error('Błąd inferencji:', error);
            document.getElementById('result').innerText = `Wystąpił błąd: ${error.message}`;
        }
    };
    reader.readAsText(file);
}

// Preprocessing (odtworzony ze _HMEDatasetBase.extract_features)
function prepareInputData(jsonData) {
    const rawTraces = Array.isArray(jsonData) && Array.isArray(jsonData[0]) ? jsonData : jsonData.traces;
    if (!rawTraces || rawTraces.length === 0) throw new Error("Brak danych w pliku JSON.");

    const traces = rawTraces.filter(t => t.length > 0);
    let minX = Infinity, minY = Infinity, minT = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxT = -Infinity;

    for (const trace of traces) {
        for (const pt of trace) {
            if (pt[0] < minX) minX = pt[0];
            if (pt[0] > maxX) maxX = pt[0];
            if (pt[1] < minY) minY = pt[1];
            if (pt[1] > maxY) maxY = pt[1];
            if (pt[2] < minT) minT = pt[2];
            if (pt[2] > maxT) maxT = pt[2];
        }
    }

    const xyRange = Math.max(maxX - minX, maxY - minY) + EPS;
    const tRange = (maxT - minT) + EPS;

    let normTraces = [];
    let exprYMin = Infinity, exprYMax = -Infinity;

    for (const trace of traces) {
        let normTrace = [];
        for (const pt of trace) {
            let nx = (pt[0] - minX) / xyRange;
            let ny = (pt[1] - minY) / xyRange;
            let nt = (pt[2] - minT) / tRange;
            normTrace.push([nx, ny, nt]);
            if (ny < exprYMin) exprYMin = ny;
            if (ny > exprYMax) exprYMax = ny;
        }
        normTraces.push(normTrace);
    }

    const exprYSpan = (exprYMax - exprYMin) + EPS;
    let allFeatures = [];
    const fround = Math.fround;

    for (const trace of normTraces) {
        const traceLen = trace.length;
        let traceYMin = Infinity, traceYMax = -Infinity;
        for (const pt of trace) {
            if (pt[1] < traceYMin) traceYMin = pt[1];
            if (pt[1] > traceYMax) traceYMax = pt[1];
        }
        const traceYCenter = 0.5 * (traceYMin + traceYMax);
        const traceYSpan = traceYMax - traceYMin;
        const yCenterRel = (traceYCenter - exprYMin) / exprYSpan;
        const ySpanRel = traceYSpan / exprYSpan;

        let d_xyt = new Array(traceLen).fill(0).map(() => [0, 0, 0]);
        let speed = new Array(traceLen).fill(0);
        let curve = new Array(traceLen).fill(0);
        let accTan = new Array(traceLen).fill(0);

        for (let i = 1; i < traceLen; i++) {
            const dx = fround(trace[i][0] - trace[i - 1][0]);
            const dy = fround(trace[i][1] - trace[i - 1][1]);
            const dt = fround(trace[i][2] - trace[i - 1][2]);
            d_xyt[i] = [dx, dy, dt];
            const dist = fround(Math.hypot(dx, dy));
            speed[i] = dt > EPS ? fround(dist / dt) : 0;
        }

        for (let i = 1; i < traceLen; i++) {
            if (i > 1) {
                let dx0 = d_xyt[i - 1][0], dy0 = d_xyt[i - 1][1];
                let dist0 = fround(Math.hypot(dx0, dy0));
                let ux0 = dist0 > EPS ? fround(dx0 / dist0) : 0;
                let uy0 = dist0 > EPS ? fround(dy0 / dist0) : 0;

                let dx1 = d_xyt[i][0], dy1 = d_xyt[i][1];
                let dist1 = fround(Math.hypot(dx1, dy1));
                let ux1 = dist1 > EPS ? fround(dx1 / dist1) : 0;
                let uy1 = dist1 > EPS ? fround(dy1 / dist1) : 0;

                let cross = fround(ux0 * uy1 - uy0 * ux1);
                let dot = fround(ux0 * ux1 + uy0 * uy1);
                let dtheta = fround(Math.atan2(cross, dot));
                curve[i] = dist1 > EPS ? fround(dtheta / dist1) : 0;
            }
            const dspeed = fround(speed[i] - speed[i - 1]);
            accTan[i] = d_xyt[i][2] > EPS ? fround(dspeed / d_xyt[i][2]) : 0;
        }

        for (let i = 0; i < traceLen; i++) {
            let isStrokeStart = (i === 0) ? 1.0 : 0.0;
            allFeatures.push([
                trace[i][0], trace[i][1], trace[i][2], d_xyt[i][0], d_xyt[i][1], d_xyt[i][2],
                speed[i], curve[i], accTan[i], isStrokeStart, yCenterRel, ySpanRel
            ]);
        }
    }

    const totalPoints = allFeatures.length;
    const colsToNorm = [3, 4, 5, 6, 7, 8];
    for (let col of colsToNorm) {
        let sum = 0;
        for (let i = 0; i < totalPoints; i++) sum += allFeatures[i][col];
        let mean = sum / totalPoints;
        let varSum = 0;
        for (let i = 0; i < totalPoints; i++) varSum += Math.pow(allFeatures[i][col] - mean, 2);

        let std = Math.sqrt(varSum / totalPoints);
        if (isNaN(std) || !Number.isFinite(std)) std = 0.0;
        std += EPS;

        for (let i = 0; i < totalPoints; i++) {
            let val = (allFeatures[i][col] - mean) / std;
            allFeatures[i][col] = Math.max(-5.0, Math.min(5.0, val));
        }
    }

    const flattened = new Float32Array(totalPoints * 12);
    for (let i = 0; i < totalPoints; i++) {
        for (let j = 0; j < 12; j++) {
            flattened[i * 12 + j] = allFeatures[i][j];
        }
    }

    const srcTensor = new ort.Tensor('float32', flattened, [1, totalPoints, 12]);
    const lengthsTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(totalPoints)]), [1]);

    return { src: srcTensor, src_lengths: lengthsTensor };
}

window.addEventListener('DOMContentLoaded', () => {
    loadModels();
    document.getElementById('runBtn').addEventListener('click', processJSON);
});