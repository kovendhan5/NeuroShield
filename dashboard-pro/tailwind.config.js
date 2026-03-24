/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neon: {
          green: '#00ff88',
          cyan: '#00ccff',
          pink: '#ff006e',
          gold: '#ffd60a',
          purple: '#b537f2',
        },
        dark: {
          bg: '#0a0e27',
          card: '#1a1f3a',
          border: '#2d3561',
        }
      },
      backgroundImage: {
        'gradient-neon': 'linear-gradient(135deg, #00ff88 0%, #00ccff 100%)',
        'gradient-dark': 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)',
      },
      backdropFilter: {
        'blur': 'blur(20px)',
      },
      boxShadow: {
        'glow-green': '0 0 20px rgba(0, 255, 136, 0.3)',
        'glow-cyan': '0 0 20px rgba(0, 204, 255, 0.3)',
        'glow-pink': '0 0 20px rgba(255, 0, 110, 0.3)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in': 'slide-in 0.6s ease-out',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
        'slide-in': {
          'from': { opacity: '0', transform: 'translateY(20px)' },
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [],
}
