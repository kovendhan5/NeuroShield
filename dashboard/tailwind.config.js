/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          50: "#f8f9fa",
          100: "#f1f3f5",
          200: "#e9ecef",
          300: "#dee2e6",
          400: "#ced4da",
          500: "#adb5bd",
          600: "#868e96",
          700: "#495057",
          800: "#343a40",
          900: "#1a1d23",
          950: "#0f1117",
        },
        blue: {
          primary: "#0969da",
          hover: "#0860ca",
          active: "#033d8b",
        },
      },
      keyframes: {
        "slide-in": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-ring": {
          "0%": { boxShadow: "0 0 0 0 rgba(9, 105, 218, 0.7)" },
          "70%": { boxShadow: "0 0 0 10px rgba(9, 105, 218, 0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(9, 105, 218, 0)" },
        },
      },
      animation: {
        "slide-in": "slide-in 0.3s ease-out",
        "pulse-ring": "pulse-ring 2s infinite",
      },
    },
  },
  plugins: [],
}
