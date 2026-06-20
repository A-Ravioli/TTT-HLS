# Pinkdeer — Agent UI (mockup)

A front-end mockup for a Pinkdeer chat agent. It fuses three reference looks:

- **Layout** of the "chat jimmy" app (status bar, Chat/Settings panel, composer, footer).
- **Aesthetic** of the "Hermes Agent" UI — clean **white**, big serif hero wordmark, left
  sessions sidebar.
- **Brand**: hibiscus-purple `#B43757` accent, on a **white galaxy** particle background.

Built with Vite + React + TypeScript + Tailwind v4, reusing the **GIODESDR** shadcn/ui
primitives and Mercury design language, recolored to a light "Mercury Light" theme.

## Run

```bash
cd web
bun install      # or: npm install
bun dev          # or: npm run dev   ->  http://localhost:5173
```

Production build: `bun run build` (type-checks with `tsc -b`, then `vite build`).

## Design notes

- **Fonts** — the hero wordmark uses **Kabisat Demo** (free *Italic Tall* demo variant by Mofr24,
  self-hosted at `public/fonts/`, personal-use). Its glyphs carry a `-20°` italic angle, so the
  title is counter-skewed `+20°` (`.hero-wordmark` in `src/index.css`) to read upright, and the
  name is set in mixed case (`Pinkdeer`) so the lowercase letters break up the heavy verticals.
  All other text is **IBM Plex Sans**; numerals/counters use **JetBrains Mono**.
- **Rectangular everything** — a global `*{border-radius:0 !important}` rule in `src/index.css`
  guarantees no rounded corners (Hermes style), plus `--radius*: 0` design tokens.
- **Galaxy** — `src/components/GalaxyBackground.tsx` draws a 2-arm spiral of ~560 hibiscus-purple
  particles with differential rotation and a soft core glow over white (or a glowing additive
  version in dark mode). Honors `prefers-reduced-motion`.
- **Theme** — light by default (per the white brief); the Settings → Dark mode toggle flips to a
  coherent dark hibiscus variant.

This is a self-contained UI mockup; the composer echoes a canned assistant reply and is not wired
to the Python TTT/HLS backend.
