/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'oklch(0.9816 0.0017 247.8577)',
        foreground: 'oklch(0.3351 0.0331 260.0317)',
        card: 'oklch(1.0000 0 0)',
        'card-foreground': 'oklch(0.3351 0.0331 260.0317)',
        primary: 'oklch(0.5628 0.2061 281.7104)',
        'primary-foreground': 'oklch(1.0000 0 0)',
        secondary: 'oklch(0.9188 0.1312 142.4943)',
        'secondary-foreground': 'oklch(0.2131 0.0324 259.2012)',
        muted: 'oklch(0.9683 0.0069 247.8959)',
        'muted-foreground': 'oklch(0.5946 0.0331 257.8717)',
        accent: 'oklch(0.8795 0.0189 247.8959)',
        'accent-foreground': 'oklch(0.3351 0.0331 260.0317)',
        destructive: 'oklch(0.6137 0.2039 25.5645)',
        'destructive-foreground': 'oklch(1.0000 0 0)',
        border: 'oklch(0.9288 0.0057 247.8869)',
        input: 'oklch(0.9288 0.0057 247.8869)',
        ring: 'oklch(0.5628 0.2061 281.7104)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
