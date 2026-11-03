import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()], base: './',
  optimizeDeps: {entries: ['index.html'], include: ['react', 'react-dom/client', 'lucide-react']},
  server: {proxy: {'/api': {target: 'http://127.0.0.1:8000', changeOrigin: true}}, watch: {ignored: ['**/extension-dist/**', '**/artifacts/**', '**/backend/**']}},
  worker: {format: 'es'}, build: {target: 'es2022'},
});
