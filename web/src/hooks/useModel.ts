import { useState, useRef, useEffect, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import { loadVocab, ENCODER_URL, DECODER_URL, runInference } from '../logic/inference';

export interface ModelState {
    vocab: { id2token: string[], PAD_IDX: number, SOS_IDX: number, EOS_IDX: number } | null;
    status: 'idle' | 'loading' | 'success' | 'error';
    error: string | null;
    progress: string;
}

export function useModel() {
    const [state, setState] = useState<ModelState>({
        vocab: null,
        status: 'idle',
        error: null,
        progress: 'Waiting to load...'
    });

    const sessionsRef = useRef<{
        encoder: ort.InferenceSession;
        decoder: ort.InferenceSession;
    } | null>(null);

    const releaseSessions = useCallback(async () => {
        if (sessionsRef.current) {
            try {
                const { encoder, decoder } = sessionsRef.current;
                await (encoder as any).handler?.dispose?.();
                await (decoder as any).handler?.dispose?.();
            } catch (e) {
                console.warn("Error during session disposal:", e);
            }
            sessionsRef.current = null;
        }
    }, []);

    useEffect(() => {
        return () => {
            if (sessionsRef.current) {
                const { encoder, decoder } = sessionsRef.current;
                (encoder as any).handler?.dispose?.();
                (decoder as any).handler?.dispose?.();
            }
        };
    }, []);

    const load = async () => {
        if (state.status === 'loading') return;
        
        setState(prev => ({ ...prev, status: 'loading', progress: 'Loading vocabulary...' }));
        
        try {
            await releaseSessions();
            const vocab = await loadVocab();

            const sessionOptions: ort.InferenceSession.SessionOptions = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            };

            setState(prev => ({ ...prev, progress: 'Loading encoder...' }));
            const encoder = await ort.InferenceSession.create(ENCODER_URL, sessionOptions);

            setState(prev => ({ ...prev, progress: 'Loading decoder...' }));
            const decoder = await ort.InferenceSession.create(DECODER_URL, sessionOptions);

            sessionsRef.current = { encoder, decoder };

            setState({
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
        if (!sessionsRef.current || !state.vocab) {
            throw new Error("Models not ready");
        }

        return runInference(
            sessionsRef.current.encoder,
            sessionsRef.current.decoder,
            flatData,
            numPoints,
            numFeatures,
            state.vocab.SOS_IDX,
            state.vocab.EOS_IDX
        );
    };

    return { 
        ...state, 
        load, 
        recognize,
        isReady: state.status === 'success' && !!sessionsRef.current
    };
}
