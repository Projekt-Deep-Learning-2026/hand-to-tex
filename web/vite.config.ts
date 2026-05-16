import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('onnxruntime-web')) return 'onnx-runtime';
            if (id.includes('jspdf') || id.includes('html2canvas')) return 'pdf-libs';
            if (id.includes('katex')) return 'katex-vendor';
            return 'vendor';
          }
        },
      },
    },
    chunkSizeWarningLimit: 1000,
  },
})
