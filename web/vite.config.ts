import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/hand-to-tex/',
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
