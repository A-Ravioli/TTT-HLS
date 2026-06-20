import { useEffect, useRef } from "react"

interface GalaxyBackgroundProps {
  /** When true, render for a dark surface (brighter core, additive glow). */
  dark?: boolean
  className?: string
}

// Hibiscus-purple particle palette (rose -> magenta -> violet).
const PALETTE: Array<[number, number, number]> = [
  [180, 55, 87], // #b43757 hibiscus
  [210, 68, 112], // #d24470 bright
  [142, 44, 80], // #8e2c50 deep
  [124, 46, 99], // #7c2e63 purple
  [94, 42, 110], // #5e2a6e violet
]

interface Particle {
  /** radius as a fraction of the galaxy extent (0..1) */
  r: number
  angle: number
  size: number
  color: [number, number, number]
  baseAlpha: number
  twPhase: number
  twSpeed: number
}

export function GalaxyBackground({ dark = false, className = "" }: GalaxyBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const darkRef = useRef(dark)

  // Keep the latest `dark` available to the animation loop without re-running it.
  useEffect(() => {
    darkRef.current = dark
  }, [dark])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const prefersReduced =
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false

    let w = 0
    let h = 0
    let dpr = 1
    let cx = 0
    let cy = 0
    let extent = 0

    // Build a 2-arm logarithmic spiral galaxy.
    const COUNT = 560
    const ARMS = 2
    const SPINS = 2.4
    const particles: Particle[] = []
    for (let i = 0; i < COUNT; i++) {
      const t = i / COUNT
      const arm = i % ARMS
      // sqrt distribution -> denser core
      const r = Math.sqrt(t)
      const armOffset = (arm / ARMS) * Math.PI * 2
      const swirl = t * SPINS * Math.PI * 2
      const jitter = (Math.sin(i * 12.9898) * 43758.5453) % 1 // deterministic pseudo-random
      const angle = swirl + armOffset + (jitter - 0.5) * 0.9
      const radialJitter = ((Math.sin(i * 78.233) * 12543.123) % 1) * 0.06
      particles.push({
        r: Math.min(1, r + radialJitter),
        angle,
        size: 0.6 + (Math.abs(jitter) % 1) * 2.2,
        color: PALETTE[i % PALETTE.length],
        baseAlpha: 0.28 + (Math.abs(Math.sin(i * 3.7)) % 1) * 0.5,
        twPhase: (i % 17) * 0.5,
        twSpeed: 0.4 + ((i % 11) / 11) * 0.9,
      })
    }

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = window.innerWidth
      h = window.innerHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      cx = w * 0.5
      cy = h * 0.46
      extent = Math.max(w, h) * 0.62
    }

    let rotation = 0
    let raf = 0
    let last = 0

    const draw = (time: number) => {
      const dt = last ? Math.min((time - last) / 1000, 0.05) : 0.016
      last = time
      if (!prefersReduced) rotation += dt * 0.05
      const isDark = darkRef.current

      ctx.clearRect(0, 0, w, h)

      // Soft galaxy-core glow at the center.
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, extent * 0.95)
      if (isDark) {
        core.addColorStop(0, "rgba(210, 68, 112, 0.22)")
        core.addColorStop(0.35, "rgba(124, 46, 99, 0.10)")
        core.addColorStop(1, "rgba(20, 15, 18, 0)")
      } else {
        core.addColorStop(0, "rgba(180, 55, 87, 0.12)")
        core.addColorStop(0.4, "rgba(124, 46, 99, 0.05)")
        core.addColorStop(1, "rgba(255, 255, 255, 0)")
      }
      ctx.fillStyle = core
      ctx.fillRect(0, 0, w, h)

      ctx.globalCompositeOperation = isDark ? "lighter" : "source-over"

      const t2 = time / 1000
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i]
        // Differential rotation: inner particles orbit faster.
        const speed = 0.6 + (1 - p.r) * 1.1
        const a = p.angle + rotation * speed
        const rad = p.r * extent
        const x = cx + Math.cos(a) * rad
        const y = cy + Math.sin(a) * rad * 0.62 // flatten -> disc seen at an angle
        if (x < -20 || x > w + 20 || y < -20 || y > h + 20) continue

        const tw = prefersReduced ? 1 : 0.55 + 0.45 * Math.sin(p.twPhase + t2 * p.twSpeed)
        const alpha = p.baseAlpha * tw * (isDark ? 0.9 : 0.8)
        const [cr, cg, cb] = p.color
        ctx.fillStyle = `rgba(${cr}, ${cg}, ${cb}, ${alpha})`
        // Round particles — the galaxy stars are decorative, exempt from the
        // rectangular-UI rule (which applies to interface elements only).
        ctx.beginPath()
        ctx.arc(x, y, p.size, 0, Math.PI * 2)
        ctx.fill()
      }

      ctx.globalCompositeOperation = "source-over"

      if (!prefersReduced) raf = requestAnimationFrame(draw)
    }

    resize()
    window.addEventListener("resize", resize)
    raf = requestAnimationFrame(draw)
    if (prefersReduced) draw(0)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener("resize", resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={`pointer-events-none fixed inset-0 -z-10 h-full w-full ${className}`}
    />
  )
}
