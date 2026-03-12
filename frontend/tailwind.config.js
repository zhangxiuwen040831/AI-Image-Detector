
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#050505',
        primary: '#DA205A',
        secondary: '#00D1FF',
        accent: '#7C3AED',
      },
      fontFamily: {
        sans: ['sans-serif'],
        mono: ['monospace'],
      },
    },
  },
  plugins: [],
}
