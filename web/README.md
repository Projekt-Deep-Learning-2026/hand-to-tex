# Web App Blueprint

This folder is reserved for the browser-based handwriting inference app.

## Intended layout
- `public/` - static assets served directly by the browser.
- `public/model/` - exported ONNX model files and runtime artifacts.
- `public/assets/` - icons, images, and other static UI assets.
- `src/` - frontend source code.
- `src/components/` - reusable UI components.
- `src/hooks/` - React hooks for canvas and inference state.
- `src/lib/` - browser utilities for feature extraction, vocab, and inference.
- `src/styles/` - global styling and theme files.
- `src/workers/` - background worker entrypoints for ONNX/WASM inference.

The Python package remains the source of truth for model export and preprocessing contracts.