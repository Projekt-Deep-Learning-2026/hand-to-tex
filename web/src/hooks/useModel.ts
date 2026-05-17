import { useState } from 'react';
import * as ort from 'onnxruntime-web';
import { loadVocab, ENCODER_URL, DECODER_URL, runInference } from '../logic/inference';

export interface ModelState {
    encoderSession: ort.InferenceSession | null;
    decoderSession: ort.InferenceSession | null;
    vocab: { id2token: string[], PAD_IDX: number, SOS_IDX: number, EOS_IDX: number } | null;
    status: 'idle' | 'loading' | 'success' | 'error';
    error: string | null;
    progress: string;
}

export function useModel() {
    const [state, setState] = useState<ModelState>({
        encoderSession: null,
        decoderSession: null,
        vocab: null,
        status: 'idle',
        error: null,
        progress: 'Waiting to load...'
    });

    const load = async () => {
        setState(prev => ({ ...prev, status: 'loading', progress: 'Loading vocabulary...' }));
        try {
            const vocab = await loadVocab();

            setState(prev => ({ ...prev, progress: 'Loading encoder...' }));
            const encoderSession = await ort.InferenceSession.create(ENCODER_URL, { executionProviders: ['wasm'] });

            setState(prev => ({ ...prev, progress: 'Loading decoder...' }));
            const decoderSession = await ort.InferenceSession.create(DECODER_URL, { executionProviders: ['wasm'] });

            setState({
                encoderSession,
                decoderSession,
                vocab,
                status: 'success',
                error: null,
                progress: 'Models loaded successfully'
            });
        } catch (err) {
            console.error("Model loading error:", err);
            setState(prev => ({
                ...prev,
                status: 'error',
                error: (err as Error).message,
                progress: 'Error loading models'
            }));
        }
    };

    const recognize = async (
        flatData: Float32Array,
        numPoints: number,
        numFeatures: number
    ): Promise<number[]> => {
        if (!state.encoderSession || !state.decoderSession || !state.vocab) {
            throw new Error("Models not ready");
        }

        return runInference(
            state.encoderSession,
            state.decoderSession,
            flatData,
            numPoints,
            numFeatures,
            state.vocab.SOS_IDX,
            state.vocab.EOS_IDX
        );
    };

    return { ...state, load, recognize };
}
