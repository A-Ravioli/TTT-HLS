import { useEffect, useMemo, useState } from "react"
import {
  ArrowUp,
  Bug,
  CheckCircle2,
  MessageSquare,
  Mic,
  Moon,
  PanelLeft,
  Plus,
  Search,
  Settings as SettingsIcon,
  Sun,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { GalaxyBackground } from "@/components/GalaxyBackground"

const AGENT_NAME = "Pinkdeer"

const SESSIONS = [
  { title: "Help with HLS pragma tuning", meta: "2h" },
  { title: "Running search on the pod", meta: "5h" },
  { title: "Confirming reward shaping", meta: "1d" },
]

interface ChatMessage {
  role: "user" | "assistant"
  text: string
}

export default function App() {
  const [tab, setTab] = useState<"chat" | "settings">("chat")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [dark, setDark] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.")
  const [topK, setTopK] = useState(8)
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
  }, [dark])

  const tokens = useMemo(
    () => (input.trim() ? Math.max(1, Math.ceil(input.trim().length / 4)) : 0),
    [input]
  )

  const send = () => {
    const text = input.trim()
    if (!text) return
    setMessages((m) => [
      ...m,
      { role: "user", text },
      {
        role: "assistant",
        text: `Got it — let me work through "${text}". I'll cite what I find and stop to ask if I'm unsure.`,
      },
    ])
    setInput("")
  }

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

            {/* ── Chat tab: sessions & navigation ── */}
            <TabsContent value="chat" className="flex flex-col">
              <div className="flex flex-col gap-1 p-3">
                <button className="group flex items-center justify-between border border-[var(--hairline)] px-3 py-2.5 text-sm font-medium transition-colors hover:border-[var(--hibiscus)] hover:text-[var(--hibiscus)]">
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
                  <Input placeholder="Search sessions…" className="h-9 pl-8" />
                </div>
              </div>

              <div className="px-3 pt-3">
                <p className="mb-1 text-[11px] font-semibold tracking-[0.14em] text-[var(--hibiscus)]">
                  PINNED
                </p>
                <p className="text-xs leading-relaxed text-ink-3">
                  Shift-click a chat to pin · drag to reorder
                </p>
              </div>

              <div className="flex-1 overflow-y-auto px-3 pt-4">
                <p className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.14em] text-[var(--hibiscus)]">
                  SESSIONS <span className="text-ink-3">{SESSIONS.length}</span>
                </p>
                <div className="flex flex-col">
                  {SESSIONS.map((s) => (
                    <button
                      key={s.title}
                      className="flex items-center justify-between gap-2 px-2 py-2 text-left text-sm text-ink-2 transition-colors hover:bg-[var(--accent)] hover:text-ink"
                    >
                      <span className="flex items-center gap-2 truncate">
                        <span className="size-1.5 shrink-0 bg-[var(--hibiscus)]" />
                        <span className="truncate">{s.title}</span>
                      </span>
                      <span className="shrink-0 font-mono text-[11px] text-ink-3">
                        {s.meta}
                      </span>
                    </button>
                  ))}
                </div>
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
              <CheckCircle2 className="size-4 text-emerald-600" />
              Ready
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
                    {m.text}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Composer */}
        <div className="px-6 pb-5">
          <div className="mx-auto w-full max-w-3xl">
            <div className="surface-strong focus-hibiscus flex items-center gap-3 border border-[var(--hairline)] px-4 py-3 transition-colors">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    send()
                  }
                }}
                placeholder="Ask me anything…"
                className="min-w-0 flex-1 bg-transparent text-base outline-none placeholder:text-ink-3"
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
                size="icon"
                className="size-9"
                aria-label="Send"
              >
                <ArrowUp className="size-4" />
              </Button>
            </div>

            <p className="mt-3 text-center text-xs text-ink-3">
              By messaging {AGENT_NAME}, you agree to our{" "}
              <a
                href="#"
                className="text-ink-2 underline underline-offset-2 hover:text-[var(--hibiscus)]"
              >
                Terms and Conditions
              </a>{" "}
              and acknowledge you have read our{" "}
              <a
                href="#"
                className="text-ink-2 underline underline-offset-2 hover:text-[var(--hibiscus)]"
              >
                Privacy Policy
              </a>
              .
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}
