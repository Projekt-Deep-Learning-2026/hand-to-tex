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

    const encoderRef = useRef<ort.InferenceSession | null>(null);
    const decoderRef = useRef<ort.InferenceSession | null>(null);

    const releaseSessions = useCallback(async () => {
        if (encoderRef.current) {
            try { await (encoderRef.current as any).handler?.dispose(); } catch {} // Deep cleanup if available
            encoderRef.current = null;
        }
        if (decoderRef.current) {
            try { await (decoderRef.current as any).handler?.dispose(); } catch {}
            decoderRef.current = null;
        }
    }, []);

    useEffect(() => {
        return () => {
            // Cleanup on unmount
            if (encoderRef.current) (encoderRef.current as any).handler?.dispose();
            if (decoderRef.current) (decoderRef.current as any).handler?.dispose();
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
            encoderRef.current = await ort.InferenceSession.create(ENCODER_URL, sessionOptions);

            setState(prev => ({ ...prev, progress: 'Loading decoder...' }));
            decoderRef.current = await ort.InferenceSession.create(DECODER_URL, sessionOptions);

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
        if (!encoderRef.current || !decoderRef.current || !state.vocab) {
            throw new Error("Models not ready");
        }

        return runInference(
            encoderRef.current,
            decoderRef.current,
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
        isReady: state.status === 'success' && !!encoderRef.current && !!decoderRef.current
    };
}
