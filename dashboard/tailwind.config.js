/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#2563eb', // Enterprise blue
        success: '#10b981', // Clean green
        warning: '#f59e0b', // Yielding amber
        danger: '#ef4444',  // Alert red
        background: '#f8fafc', // Very subtle slate for corporate look
        surface: '#ffffff',
        border: '#e2e8f0',
        textMain: '#0f172a',
        textMuted: '#64748b'
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
        'elevated': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)'
      }
    },
  },
  plugins: [],
}
