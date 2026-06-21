import { useEffect, useMemo, useRef, useState } from "react"
import {
  ArrowUp,
  Bug,
  CheckCircle2,
  MessageSquare,
  Mic,
  Moon,
  PanelLeft,
  Pin,
  Plus,
  Search,
  Settings as SettingsIcon,
  Sun,
  Trash2,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { GalaxyBackground } from "@/components/GalaxyBackground"

const AGENT_NAME = "burnt"
const NEW_TITLE = "New session"
const STORE_KEY = "pinkdeer.sessions.v1"
const EMPTY_MESSAGES: ChatMessage[] = []

interface ChatMessage {
  role: "user" | "assistant"
  text: string
  ms?: number // generation ("compile") time in milliseconds, set when the stream completes
}

interface Session {
  id: string
  title: string
  messages: ChatMessage[]
  pinned: boolean
  createdAt: number
  updatedAt: number
}

function uid(): string {
  return globalThis.crypto?.randomUUID?.() ?? `s_${Date.now()}_${Math.random().toString(36).slice(2)}`
}

function makeSession(partial?: Partial<Session>): Session {
  const now = Date.now()
  return { id: uid(), title: NEW_TITLE, messages: [], pinned: false, createdAt: now, updatedAt: now, ...partial }
}

function deriveTitle(text: string): string {
  const t = text.replace(/\s+/g, " ").trim()
  if (!t) return NEW_TITLE
  return t.length > 42 ? `${t.slice(0, 42)}…` : t
}

function relTime(ts: number): string {
  const s = Math.max(0, Math.floor((Date.now() - ts) / 1000))
  if (s < 45) return "now"
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h`
  const d = Math.floor(h / 24)
  if (d < 7) return `${d}d`
  const w = Math.floor(d / 7)
  if (w < 5) return `${w}w`
  return `${Math.floor(d / 30)}mo`
}

// Seed a few example conversations on first run so the sessions list isn't empty.
function seedSessions(): Session[] {
  const now = Date.now()
  const mk = (title: string, ageH: number, messages: ChatMessage[]): Session => ({
    id: uid(),
    title,
    pinned: false,
    createdAt: now - ageH * 3600_000,
    updatedAt: now - ageH * 3600_000,
    messages,
  })
  return [
    mk("Help with HLS pragma tuning", 2, [
      { role: "user", text: "How do I pipeline this inner loop in Vitis HLS?" },
      {
        role: "assistant",
        text: "Put `#pragma HLS pipeline II=1` at the top of the inner loop body and keep the bounds compile-time constant. If you hit II violations, partition the arrays you read each iteration.",
      },
    ]),
    mk("Running search on the pod", 5, [
      { role: "user", text: "What command launches the BurnTTT search on the pod?" },
      {
        role: "assistant",
        text: "Activate the `burnttt` env, then run `python scripts/02_run_burnttt_search.py`. It reads results/runs.csv and resumes from the last checkpoint.",
      },
    ]),
    mk("Confirming reward shaping", 26, [
      { role: "user", text: "Is the latency term in the reward normalized?" },
      {
        role: "assistant",
        text: "Yes — see `ttt/reward.py`: latency is divided by the board budget before being combined with the accuracy term, so both sit on a 0–1 scale.",
      },
    ]),
  ]
}

function loadStore(): { sessions: Session[]; activeId: string } {
  try {
    const raw = localStorage.getItem(STORE_KEY)
    if (raw) {
      const p = JSON.parse(raw)
      if (p && Array.isArray(p.sessions) && p.sessions.length) {
        const activeId = p.sessions.some((s: Session) => s.id === p.activeId)
          ? p.activeId
          : p.sessions[0].id
        return { sessions: p.sessions as Session[], activeId }
      }
    }
  } catch {
    /* ignore corrupt storage */
  }
  const draft = makeSession()
  return { sessions: [draft, ...seedSessions()], activeId: draft.id }
}

// Render assistant text with fenced ```code``` blocks as styled code panels.
// Splitting on ``` makes odd-index segments code; an unclosed fence (mid-stream)
// still renders as a code block as it types.
function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () =>
    navigator.clipboard?.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    })
  return (
    <div className="my-2 border border-[var(--hairline)] bg-[var(--accent)]">
      <div className="flex items-center justify-between border-b border-[var(--hairline)] px-3 py-1">
        <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--hibiscus)]">
          {lang || "code"}
        </span>
        <button
          onClick={copy}
          className="font-mono text-[10px] text-ink-3 transition-colors hover:text-[var(--hibiscus)]"
        >
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre className="overflow-x-auto px-3 py-2 text-[12.5px] leading-relaxed">
        <code className="font-mono text-ink">{code}</code>
      </pre>
    </div>
  )
}

function renderBody(text: string) {
  const parts = text.split("```")
  return parts.map((seg, i) => {
    if (i % 2 === 1) {
      const nl = seg.indexOf("\n")
      const lang = nl >= 0 ? seg.slice(0, nl).trim() : ""
      const code = (nl >= 0 ? seg.slice(nl + 1) : seg).replace(/\n$/, "")
      return <CodeBlock key={i} lang={lang} code={code} />
    }
    return seg ? (
      <span key={i} className="whitespace-pre-wrap">
        {seg}
      </span>
    ) : null
  })
}

export default function App() {
  const init = useMemo(loadStore, [])
  const [sessions, setSessions] = useState<Session[]>(init.sessions)
  const [activeId, setActiveId] = useState<string>(init.activeId)
  const [query, setQuery] = useState("")

  const [tab, setTab] = useState<"chat" | "settings">("chat")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [dark, setDark] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a code generator. Output ONLY code inside a single fenced markdown block " +
      "(```language). No explanations or prose unless the user explicitly asks. Keep it minimal and correct."
  )
  const [topK, setTopK] = useState(20)
  const [input, setInput] = useState("")
  const [pending, setPending] = useState(false)
  const modelName = "NanoCoder-v3" // displayed label (fixed, regardless of backend)

  const active = sessions.find((s) => s.id === activeId)
  const messages = active ? active.messages : EMPTY_MESSAGES

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
  }, [dark])

  // Persist sessions + which one is open across reloads.
  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ sessions, activeId }))
    } catch {
      /* storage full / unavailable — non-fatal */
    }
  }, [sessions, activeId])

  // Always keep a valid active session (e.g. after deleting the open one).
  useEffect(() => {
    if (sessions.length && !sessions.some((s) => s.id === activeId)) {
      setActiveId(sessions[0].id)
    }
  }, [sessions, activeId])


  const tokens = useMemo(
    () => (input.trim() ? Math.max(1, Math.ceil(input.trim().length / 4)) : 0),
    [input]
  )

  // Keep the conversation pinned to the most recent message as it grows.
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [messages, pending])

  // ── Session operations ─────────────────────────────────────────────
  const newSession = () => {
    setTab("chat")
    setInput("")
    // Reuse the current empty draft instead of stacking empty sessions.
    if (active && active.messages.length === 0) return
    const s = makeSession()
    setSessions((prev) => [s, ...prev])
    setActiveId(s.id)
  }

  const selectSession = (id: string) => {
    setActiveId(id)
    setInput("")
  }

  const togglePin = (id: string) =>
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, pinned: !s.pinned } : s)))

  const deleteSession = (id: string) =>
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id)
      return next.length ? next : [makeSession()]
    })

  // Update the messages of the session that was active when called.
  const patchActive = (
    updater: (msgs: ChatMessage[]) => ChatMessage[],
    titleFromText?: string
  ) =>
    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeId
          ? {
              ...s,
              messages: updater(s.messages),
              updatedAt: Date.now(),
              title:
                titleFromText && (s.title === NEW_TITLE || s.messages.length === 0)
                  ? titleFromText
                  : s.title,
            }
          : s
      )
    )

  // ── Backend: talk to the coding-model server (scripts/21_serve_chat.py),
  // proxied via /api in vite.config.ts. Streams the reply token-by-token. ──
  const send = async () => {
    const text = input.trim()
    if (!text || pending) return
    const userMsg: ChatMessage = { role: "user", text }
    const history: ChatMessage[] = [...messages, userMsg]
    patchActive((m) => [...m, userMsg], deriveTitle(text))
    setInput("")
    setPending(true)

    const payload = {
      messages: history.map((m) => ({ role: m.role, content: m.text })),
      system: systemPrompt.trim(),
      top_k: topK,
      max_new_tokens: 512,
    }

    try {
      const res = await fetch("/api/generate_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!res.ok || !res.body) {
        const err = await res.text().catch(() => res.statusText)
        patchActive((m) => [...m, { role: "assistant", text: `⚠️ ${err}` }])
        return
      }
      // Append an empty assistant message, then stream tokens into it live.
      const t0 = performance.now()
      patchActive((m) => [...m, { role: "assistant", text: "" }])
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let acc = ""
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        acc += decoder.decode(value, { stream: true })
        patchActive((m) => {
          const copy = [...m]
          copy[copy.length - 1] = { role: "assistant", text: acc }
          return copy
        })
      }
      const ms = Math.round(performance.now() - t0)
      patchActive((m) => {
        const copy = [...m]
        copy[copy.length - 1] = { role: "assistant", text: acc.trim() || "…", ms }
        return copy
      })
    } catch {
      patchActive((m) => [
        ...m,
        {
          role: "assistant",
          text: "⚠️ Can't reach the model backend. Start it with: python scripts/27_serve_qwen_mlx.py",
        },
      ])
    } finally {
      setPending(false)
    }
  }

  // ── Derived session lists (search + pin split) ─────────────────────
  const q = query.trim().toLowerCase()
  const listed = sessions
    .filter((s) => s.messages.length > 0)
    .filter(
      (s) =>
        !q ||
        s.title.toLowerCase().includes(q) ||
        s.messages.some((m) => m.text.toLowerCase().includes(q))
    )
    .sort((a, b) => b.updatedAt - a.updatedAt)
  const pinned = listed.filter((s) => s.pinned)
  const recent = listed.filter((s) => !s.pinned)

  const renderSession = (s: Session) => (
    <div
      key={s.id}
      onClick={(e) => (e.shiftKey ? togglePin(s.id) : selectSession(s.id))}
      title="Click to open · Shift-click to pin"
      className={`group/row flex cursor-pointer items-center justify-between gap-2 px-2 py-2 text-sm transition-colors ${
        s.id === activeId
          ? "bg-[var(--accent)] text-ink"
          : "text-ink-2 hover:bg-[var(--accent)] hover:text-ink"
      }`}
    >
      <span className="flex min-w-0 items-center gap-2">
        <span
          className={`size-1.5 shrink-0 ${s.pinned ? "bg-[var(--hibiscus)]" : "bg-[var(--hairline)]"}`}
        />
        <span className="truncate">{s.title}</span>
      </span>
      <span className="flex shrink-0 items-center gap-1">
        <span className="font-mono text-[11px] text-ink-3 group-hover/row:hidden">
          {relTime(s.updatedAt)}
        </span>
        <span className="hidden items-center gap-0.5 group-hover/row:flex">
          <button
            onClick={(e) => {
              e.stopPropagation()
              togglePin(s.id)
            }}
            aria-label={s.pinned ? "Unpin session" : "Pin session"}
            className={`flex size-6 items-center justify-center transition-colors hover:text-[var(--hibiscus)] ${
              s.pinned ? "text-[var(--hibiscus)]" : "text-ink-3"
            }`}
          >
            <Pin className={`size-3.5 ${s.pinned ? "fill-current" : ""}`} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation()
              deleteSession(s.id)
            }}
            aria-label="Delete session"
            className="flex size-6 items-center justify-center text-ink-3 transition-colors hover:text-[var(--hibiscus)]"
          >
            <Trash2 className="size-3.5" />
          </button>
        </span>
      </span>
    </div>
  )

  return (
    <div className="relative flex h-screen w-screen overflow-hidden text-foreground">
      <GalaxyBackground dark={dark} />

      {/* ───────────────────────── Sidebar ───────────────────────── */}
      {sidebarOpen && (
        <aside className="surface z-10 flex h-full w-[300px] shrink-0 flex-col border-r border-[var(--sidebar-border)]">
          <Tabs value={tab} onValueChange={(v) => setTab(v as "chat" | "settings")}>
            <TabsList>
              <TabsTrigger value="chat">
                <MessageSquare className="size-4" />
              </TabsTrigger>
              <TabsTrigger value="settings">
                <SettingsIcon className="size-4" />
              </TabsTrigger>
            </TabsList>

            {/* ── Chat tab: sessions ── */}
            <TabsContent value="chat" className="flex min-h-0 flex-col">
              <div className="flex flex-col gap-1 p-3">
                <button
                  onClick={newSession}
                  className="group flex items-center justify-between border border-[var(--hairline)] px-3 py-2.5 text-sm font-medium transition-colors hover:border-[var(--hibiscus)] hover:text-[var(--hibiscus)]"
                >
                  <span className="flex items-center gap-2">
                    <Plus className="size-4" />
                    New session
                  </span>
                  <span className="flex gap-1 text-[10px] text-ink-3">
                    <kbd className="border border-[var(--hairline)] px-1 py-0.5">⌘</kbd>
                    <kbd className="border border-[var(--hairline)] px-1 py-0.5">N</kbd>
                  </span>
                </button>
              </div>

              <div className="px-3 pb-2">
                <div className="relative">
                  <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-ink-3" />
                  <Input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search sessions…"
                    className="h-9 pl-8"
                  />
                </div>
              </div>

              <div className="px-3 pt-3">
                <p className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.14em] text-[var(--hibiscus)]">
                  PINNED {pinned.length > 0 && <span className="text-ink-3">{pinned.length}</span>}
                </p>
                {pinned.length > 0 ? (
                  <div className="flex flex-col">{pinned.map(renderSession)}</div>
                ) : (
                  <p className="text-xs leading-relaxed text-ink-3">
                    Shift-click a session to pin it here.
                  </p>
                )}
              </div>

              <div className="mt-3 flex min-h-0 flex-1 flex-col overflow-y-auto px-3">
                <p className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.14em] text-[var(--hibiscus)]">
                  SESSIONS <span className="text-ink-3">{recent.length}</span>
                </p>
                {recent.length > 0 ? (
                  <div className="flex flex-col">{recent.map(renderSession)}</div>
                ) : (
                  <p className="px-2 text-xs leading-relaxed text-ink-3">
                    {q ? "No sessions match your search." : "No sessions yet — start chatting."}
                  </p>
                )}
              </div>
            </TabsContent>

            {/* ── Settings tab ── */}
            <TabsContent value="settings" className="flex flex-col gap-6 p-4">
              <div className="flex flex-col gap-2">
                <Label className="text-ink-2">System prompt</Label>
                <Textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="min-h-32 text-ink"
                />
              </div>

              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <Label className="text-ink-2">Top K</Label>
                  <span className="border border-[var(--hairline)] px-2 py-0.5 font-mono text-sm text-ink">
                    {topK}
                  </span>
                </div>
                <Slider
                  value={[topK]}
                  min={1}
                  max={40}
                  step={1}
                  onValueChange={(v) => setTopK(v[0])}
                />
              </div>

              <button
                onClick={() => setDark((d) => !d)}
                className="flex items-center justify-between border border-[var(--hairline)] px-3 py-2.5 text-sm text-ink-2 transition-colors hover:border-[var(--hibiscus)]"
              >
                <span className="flex items-center gap-2">
                  {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
                  {dark ? "Light mode" : "Dark mode"}
                </span>
                <Switch checked={dark} onCheckedChange={setDark} />
              </button>
            </TabsContent>
          </Tabs>
        </aside>
      )}

      {/* ───────────────────────── Main column ───────────────────────── */}
      <main className="relative z-0 flex h-full min-w-0 flex-1 flex-col">
        {/* Top status bar */}
        <header className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSidebarOpen((o) => !o)}
              className="flex size-8 items-center justify-center border border-[var(--hairline)] text-ink-2 transition-colors hover:border-[var(--hibiscus)] hover:text-[var(--hibiscus)]"
              aria-label="Toggle sidebar"
            >
              <PanelLeft className="size-4" />
            </button>
            <div className="surface-strong flex items-center gap-1.5 border border-[var(--hairline)] px-3 py-1.5 text-sm font-medium">
              {pending ? (
                <>
                  <span className="size-2 animate-pulse bg-[var(--hibiscus)]" />
                  Generating…
                </>
              ) : (
                <>
                  <CheckCircle2 className="size-4 text-emerald-600" />
                  {modelName}
                </>
              )}
            </div>
          </div>

          <button className="surface-strong flex items-center gap-2 border border-[var(--hairline)] px-3 py-1.5 text-sm text-ink-2 transition-colors hover:border-[var(--hibiscus)] hover:text-[var(--hibiscus)]">
            Report Bug
            <Bug className="size-4 text-[var(--hibiscus)]" />
          </button>
        </header>

        {/* Center / conversation */}
        {messages.length === 0 ? (
          <section className="flex flex-1 flex-col items-center justify-center px-6 text-center">
            <h1 className="hero-wordmark text-[clamp(2.35rem,5.1vw,4.25rem)]">
              {AGENT_NAME}
            </h1>
            <p className="mt-6 text-2xl font-light text-ink-2">
              How can I help you today?
            </p>
            <p className="mt-3 max-w-md text-sm leading-relaxed text-ink-3">
              Type a task, question, or snippet. I remember the session, cite my
              sources, and stop to ask when I'm unsure.
            </p>
          </section>
        ) : (
          <section className="flex-1 overflow-y-auto px-6 py-8">
            <div className="mx-auto flex max-w-3xl flex-col gap-6">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={
                    m.role === "user" ? "flex justify-end" : "flex justify-start"
                  }
                >
                  <div
                    className={
                      m.role === "user"
                        ? "max-w-[80%] border border-[var(--hibiscus-line)] bg-[var(--accent)] px-4 py-3 text-sm text-ink"
                        : "surface-strong max-w-[80%] border border-[var(--hairline)] px-4 py-3 text-sm text-ink"
                    }
                  >
                    <p className="mb-1 font-mono text-[11px] tracking-wide text-[var(--hibiscus)]">
                      {m.role === "user" ? "YOU" : AGENT_NAME.toUpperCase()}
                    </p>
                    {m.role === "assistant" ? renderBody(m.text) : m.text}
                    {m.role === "assistant" && m.ms != null && (
                      <p className="mt-1.5 font-mono text-[10px] text-ink-3">
                        ⚡ compiled in {(m.ms / 1000).toFixed(2)}s · ~
                        {Math.max(1, Math.round(m.text.length / 4 / (m.ms / 1000)))} tok/s
                      </p>
                    )}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          </section>
        )}

        {/* Composer */}
        <div className="px-6 pb-5">
          <div className="mx-auto w-full max-w-3xl">
            <div className="surface-strong focus-hibiscus flex items-center gap-3 border border-[var(--hairline)] px-4 py-3 transition-colors">
              <input
                value={input}
                disabled={pending}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    send()
                  }
                }}
                placeholder={pending ? "Generating…" : "Ask me to write or debug code…"}
                className="min-w-0 flex-1 bg-transparent text-base outline-none placeholder:text-ink-3 disabled:opacity-60"
              />
              <span className="shrink-0 font-mono text-xs text-ink-3">
                {tokens} tokens
              </span>
              <button
                className="flex size-7 items-center justify-center text-ink-3 transition-colors hover:text-[var(--hibiscus)]"
                aria-label="Voice input"
              >
                <Mic className="size-4" />
              </button>
              <Button
                onClick={send}
                disabled={pending}
                size="icon"
                className="size-9"
                aria-label="Send"
              >
                <ArrowUp className="size-4" />
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
