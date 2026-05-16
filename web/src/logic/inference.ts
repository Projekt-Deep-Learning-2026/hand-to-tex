import * as ort from 'onnxruntime-web';

export interface InferenceFeatures {
    flatData: Float32Array;
    numPoints: number;
    numFeatures: number;
}

export const VOCAB_URL = 'assets/vocab.json';
export const ENCODER_URL = 'https://huggingface.co/m4jkiuwr/htt-mini/resolve/main/encoder.onnx?download=true';
export const DECODER_URL = 'https://huggingface.co/m4jkiuwr/htt-mini/resolve/main/decoder_step.onnx?download=true';

const EPS = 1e-6;

export async function loadVocab(): Promise<{ id2token: string[], PAD_IDX: number, SOS_IDX: number, EOS_IDX: number }> {
    const response = await fetch(VOCAB_URL);
    const vocabData = await response.json();

    const id2token: string[] = [];
    for (const category in vocabData) {
        id2token.push(...vocabData[category]);
    }

    const PAD_IDX = id2token.indexOf("<PAD>");
    const SOS_IDX = id2token.indexOf("<SOS>");
    const EOS_IDX = id2token.indexOf("<EOS>");

    return { id2token, PAD_IDX, SOS_IDX, EOS_IDX };
}

export function extractFeatures(traces: number[][][]): InferenceFeatures {
    if (traces.length === 0) return { flatData: new Float32Array(0), numPoints: 0, numFeatures: 12 };

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minT = Infinity, maxT = -Infinity;

    for (const trace of traces) {
        for (const pt of trace) {
            const [x, y, t] = pt;
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (t < minT) minT = t; if (t > maxT) maxT = t;
        }
    }

    const xyRange = Math.max(maxX - minX, maxY - minY) + EPS;
    const tRange = maxT - minT + EPS;
    const exprYSpan = (maxY - minY) + EPS;

    const featuresList: number[][] = [];

    for (const trace of traces) {
        let traceMinY = Infinity, traceMaxY = -Infinity;
        for (const pt of trace) {
            if (pt[1] < traceMinY) traceMinY = pt[1];
            if (pt[1] > traceMaxY) traceMaxY = pt[1];
        }
        const traceYCenter = 0.5 * (traceMinY + traceMaxY);
        const traceYSpan = traceMaxY - traceMinY;

        const speeds: number[] = [], uxs: number[] = [], uys: number[] = [], dists: number[] = [];

        for (let i = 0; i < trace.length; i++) {
            const pt = trace[i];
            const xNorm = (pt[0] - minX) / xyRange;
            const yNorm = (pt[1] - minY) / xyRange;
            const tNorm = (pt[2] - minT) / tRange;
            const yCenterRel = (traceYCenter - minY) / exprYSpan;
            const ySpanRel = traceYSpan / exprYSpan;

            let dx = 0, dy = 0, dt = 0, speed = 0, dist = 0, ux = 0, uy = 0;

            if (i > 0) {
                const prevPt = trace[i - 1];
                const prevXNorm = (prevPt[0] - minX) / xyRange;
                const prevYNorm = (prevPt[1] - minY) / xyRange;
                const prevTNorm = (prevPt[2] - minT) / tRange;

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
                const prevSpeed = speeds[i - 1];
                accTan = dt > EPS ? (speed - prevSpeed) / dt : 0;
                if (i > 1) {
                    const prevUx = uxs[i - 1], prevUy = uys[i - 1];
                    const dTheta = Math.atan2(prevUx * uy - prevUy * ux, prevUx * ux + prevUy * uy);
                    curve = dist > EPS ? dTheta / dist : 0;
                }
            }

            const isStrokeStart = i === 0 ? 1.0 : 0.0;
            featuresList.push([xNorm, yNorm, tNorm, dx, dy, dt, speed, curve, accTan, isStrokeStart, yCenterRel, ySpanRel]);
        }
    }

    const colsToNorm = [3, 4, 5, 6, 7, 8];
    for (const col of colsToNorm) {
        let sum = 0;
        for (const row of featuresList) sum += row[col];
        const mean = sum / featuresList.length;

        let varianceSum = 0;
        for (const row of featuresList) varianceSum += Math.pow(row[col] - mean, 2);
        const std = Math.sqrt(varianceSum / featuresList.length) + EPS;

        for (const row of featuresList) {
            row[col] = Math.max(-5.0, Math.min(5.0, (row[col] - mean) / std));
        }
    }

    const numPoints = featuresList.length;
    const flatFeatures = new Float32Array(numPoints * 12);
    let ptr = 0;
    for (const row of featuresList) {
        for (const val of row) flatFeatures[ptr++] = val;
    }

    return { flatData: flatFeatures, numPoints, numFeatures: 12 };
}

export async function runInference(
    encoderSession: ort.InferenceSession,
    decoderSession: ort.InferenceSession,
    flatData: Float32Array,
    numPoints: number,
    numFeatures: number,
    SOS_IDX: number,
    EOS_IDX: number
): Promise<number[]> {
    const batchSize = 1;
    const srcDims = [batchSize, numPoints, numFeatures];

    const srcFeaturesTensor = new ort.Tensor('float32', flatData, srcDims);
    const srcLengthsTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(numPoints)]), [batchSize]);

    const encFeeds = { "src": srcFeaturesTensor, "src_lengths": srcLengthsTensor };
    const encResults = await encoderSession.run(encFeeds);

    const memK = encResults.mem_k;
    const memV = encResults.mem_v;
    const memMask = encResults.mem_mask;

    const numLayers = memK.dims[0];
    const numHeads = memK.dims[2];
    const headDim = memK.dims[4];

    let selfK = new ort.Tensor('float32', new Float32Array(0), [numLayers, batchSize, numHeads, 0, headDim]);
    let selfV = new ort.Tensor('float32', new Float32Array(0), [numLayers, batchSize, numHeads, 0, headDim]);

    let step = new ort.Tensor('int64', new BigInt64Array([0n]), [1]);
    let tgtLast = new ort.Tensor('int64', new BigInt64Array([BigInt(SOS_IDX)]), [batchSize, 1]);

    const maxLen = 150;
    const generatedTokenIds: number[] = [];

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
        const logitsData = decResults.logits.data as Float32Array;

        selfK = decResults.self_k_out as ort.TypedTensor<'float32'>;
        selfV = decResults.self_v_out as ort.TypedTensor<'float32'>;

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

export function parseInkml(xmlString: string): number[][][] {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlString, "text/xml");
    let traceNodes = xmlDoc.getElementsByTagNameNS("http://www.w3.org/2003/InkML", "trace");
    if (traceNodes.length === 0) traceNodes = xmlDoc.getElementsByTagName("trace");

    const traces: number[][][] = [];
    for (const node of Array.from(traceNodes)) {
        const text = node.textContent?.trim();
        if (!text) continue;
        const pointsStr = text.split(',');
        const trace: number[][] = [];
        for (const ptStr of pointsStr) {
            const coords = ptStr.trim().split(/\s+/).map(Number);
            if (coords.length >= 3 && !isNaN(coords[0])) {
                trace.push([coords[0], coords[1], coords[2]]);
            }
        }
        if (trace.length > 0) traces.push(trace);
    }
    return traces;
}
