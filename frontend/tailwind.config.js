
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#f5f7fa',
        surface: '#ffffff',
        panel: '#f8fafc',
        panelMuted: '#eef2f7',
        line: '#d9e1ea',
        lineStrong: '#c6d0dc',
        ink: '#111827',
        muted: '#5f6f82',
        subtle: '#8a97a8',
        primary: '#1f2937',
        secondary: '#475569',
        accent: '#94a3b8',
        success: '#0f766e',
        danger: '#b42318',
      },
      fontFamily: {
        sans: ['"Segoe UI"', '"Helvetica Neue"', 'Arial', 'sans-serif'],
        mono: ['"SFMono-Regular"', '"Consolas"', '"Liberation Mono"', 'monospace'],
      },
      boxShadow: {
        soft: '0 12px 32px rgba(15, 23, 42, 0.06)',
        card: '0 10px 30px rgba(15, 23, 42, 0.05)',
        lift: '0 18px 48px rgba(15, 23, 42, 0.09)',
      },
    },
  },
  plugins: [],
}
