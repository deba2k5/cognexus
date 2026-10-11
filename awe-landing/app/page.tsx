"use client";

import "./chat.css";
import "./chat-polish.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  Plus, Menu, X, ArrowUp, Globe, ShieldCheck, Network, Sparkles, GitCompareArrows, FileText,
  Table2, Trash2, Square, Search, BookOpen, ArrowDown, Check, Loader2, Download, Link2,
  MessageSquare,
} from "lucide-react";
import AssistantMessage from "./components/chat/AssistantMessage";
import { API_URL } from "./lib/api";
import { downloadDocument, sendChat, type ChatResponse, type Strategy, type Intent } from "./lib/chat";

interface Msg {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: string;
  res?: ChatResponse;
}

interface Chat {
  id: string;
  title: string;
  siteUrl: string | null;
  siteDomain: string | null;
  messages: Msg[];
  updatedAt: number;
}

const STORAGE_KEY = "cognexus.chats.v1";
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

const STRATEGIES: { id: Strategy; label: string; hint: string }[] = [
  { id: "beam", label: "Beam", hint: "Beam search: keep the top-k thoughts at every depth (balanced)" },
  { id: "bfs", label: "BFS", hint: "Breadth-first: expand every viable thought level by level (broadest)" },
  { id: "dfs", label: "DFS", hint: "Depth-first: follow the best branch, backtrack on failure (fastest to a goal)" },
];

const QUICK_ACTIONS: { label: string; message: string; intent: Intent; icon: React.ReactNode }[] = [
  { label: "Overview", message: "Give me a deep-research overview of this website", intent: "overview", icon: <Search size={13} /> },
  { label: "Structure", message: "Show the internal structure and sitemap", intent: "structure", icon: <Network size={13} /> },
  { label: "Security", message: "Run a security audit with fixes", intent: "security", icon: <ShieldCheck size={13} /> },
  { label: "Similar sites", message: "Suggest similar websites", intent: "similar", icon: <Sparkles size={13} /> },
  { label: "Compare", message: "Compare this website with its closest alternatives", intent: "compare", icon: <GitCompareArrows size={13} /> },
  { label: "Extract data", message: "Extract the key structured data from this website as a table", intent: "extract", icon: <Table2 size={13} /> },
  { label: "Report", message: "Generate a full research report", intent: "report", icon: <FileText size={13} /> },
];

const CAPABILITIES: { title: string; desc: string; prompt: string; icon: React.ReactNode; tint: string }[] = [
  { title: "Deep overview", desc: "Crawl key pages and summarise what the site offers", prompt: "stripe.com", icon: <Search size={17} />, tint: "#8f7cff" },
  { title: "Security audit", desc: "Find weaknesses and get copy-paste fixes", prompt: "Run a security audit of github.com", icon: <ShieldCheck size={17} />, tint: "#5ee6a8" },
  { title: "Internal structure", desc: "Sitemap, navigation, tech stack and SEO health", prompt: "Show the sitemap and tech stack of vercel.com", icon: <Network size={17} />, tint: "#7cc4ff" },
  { title: "Similar websites", desc: "Discover live alternatives and competitors", prompt: "Suggest websites similar to notion.so", icon: <Sparkles size={17} />, tint: "#f2a6f7" },
  { title: "Compare sites", desc: "Side-by-side metrics with a ranked verdict", prompt: "Compare github.com with gitlab.com", icon: <GitCompareArrows size={17} />, tint: "#ffc65c" },
  { title: "Extract data", desc: "Turn pages into tables, CSV and JSON", prompt: "Extract all quotes with authors and tags from quotes.toscrape.com", icon: <Table2 size={17} />, tint: "#ff8fab" },
];

const STAGES: { at: number; label: string }[] = [
  { at: 0, label: "Routing request & guardrail check" },
  { at: 2500, label: "Crawling key pages" },
  { at: 7000, label: "Analysing structure, tech & security" },
  { at: 14000, label: "Generating candidate thoughts" },
  { at: 19000, label: "Evaluating & pruning branches" },
  { at: 24000, label: "Expanding promising branches" },
  { at: 32000, label: "Ranking options by confidence" },
];

function SiteIcon({ domain, size = 14 }: { domain: string | null; size?: number }) {
  const [ok, setOk] = useState(true);
  if (!domain || !ok) return <Globe size={size} className="shrink-0" style={{ color: "var(--c-muted)" }} />;
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={`https://${domain}/favicon.ico`} alt="" width={size} height={size} className="shrink-0 rounded-sm" onError={() => setOk(false)} />
  );
}

function Mark({ size }: { size: number }) {
  return (
    <div className="avatar logo-mark" style={{ width: size, height: size }}>
      <Image src="/brand/cognexus-mark.png" alt="CogNexus" width={64} height={64} className="w-full h-full object-cover" />
    </div>
  );
}

function Thinking({ strategy, cached }: { strategy: Strategy; cached: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const t = setInterval(() => setElapsed(Date.now() - start), 250);
    return () => clearInterval(t);
  }, []);
  // A site that is already researched skips the crawl stages
  const stages = cached ? STAGES.filter((s) => s.at === 0 || s.at >= 14000).map((s) => ({ ...s, at: s.at && s.at - 12000 })) : STAGES;
  const current = stages.reduce((idx, s, i) => (elapsed >= s.at ? i : idx), 0);
  return (
    <div className="flex gap-3 fade-in">
      <Mark size={32} />
      <div className="glass card !py-3 !px-4 w-full max-w-md">
        <div className="flex items-center gap-2 mb-2.5">
          <span className="pulse-dot" />
          <span className="shimmer-text text-[13.5px] font-semibold">Researching</span>
          <span className="ml-auto text-[11.5px] tabular-nums" style={{ color: "var(--c-faint)" }}>
            {strategy.toUpperCase()} · {(elapsed / 1000).toFixed(0)}s
          </span>
        </div>
        <ol className="stepper">
          {stages.map((s, i) => (
            <li key={s.label} className={i < current ? "done" : i === current ? "active" : ""}>
              <span className="step-dot">
                {i < current ? <Check size={10} strokeWidth={3} /> : i === current ? <Loader2 size={11} className="animate-spin" /> : null}
              </span>
              {s.label}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function groupChats(chats: Chat[]) {
  const startOfDay = new Date().setHours(0, 0, 0, 0);
  const day = 86_400_000;
  const groups: { label: string; items: Chat[] }[] = [
    { label: "Today", items: [] }, { label: "Yesterday", items: [] },
    { label: "Previous 7 days", items: [] }, { label: "Older", items: [] },
  ];
  [...chats].sort((a, b) => b.updatedAt - a.updatedAt).forEach((c) => {
    const g = c.updatedAt >= startOfDay ? 0 : c.updatedAt >= startOfDay - day ? 1 : c.updatedAt >= startOfDay - 7 * day ? 2 : 3;
    groups[g].items.push(c);
  });
  return groups.filter((g) => g.items.length);
}

function chatToMarkdown(chat: Chat) {
  const lines = [`# ${chat.title}`, "", chat.siteUrl ? `Website: ${chat.siteUrl}` : "", ""];
  chat.messages.forEach((m) => {
    lines.push(m.role === "user" ? `## You\n\n${m.content}\n` : `## CogNexus\n\n${m.res?.message ?? m.content}\n`);
  });
  return lines.join("\n");
}

export default function ChatApp() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [strategy, setStrategy] = useState<Strategy>("beam");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [atBottom, setAtBottom] = useState(true);
  const [online, setOnline] = useState<boolean | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const loaded = useRef(false);

  const active = chats.find((c) => c.id === activeId) || null;

  // Restore conversations (per-browser convenience only)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw) as { chats: Chat[]; strategy?: Strategy };
        setChats(saved.chats || []);
        if (saved.strategy) setStrategy(saved.strategy);
      }
    } catch { /* storage unavailable */ }
    loaded.current = true;
  }, []);

  useEffect(() => {
    if (!loaded.current) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ chats: chats.slice(0, 30), strategy }));
    } catch { /* quota or private mode */ }
  }, [chats, strategy]);

  // Research engine status
  useEffect(() => {
    const ping = () =>
      fetch(`${API_URL}/health`).then((r) => setOnline(r.ok)).catch(() => setOnline(false));
    ping();
    const t = setInterval(ping, 60_000);
    return () => clearInterval(t);
  }, []);

  const scrollToBottom = (smooth = true) =>
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: smooth ? "smooth" : "auto" });

  useEffect(() => {
    scrollToBottom();
  }, [active?.messages.length, loading]);

  useEffect(() => {
    scrollToBottom(false);
    setAtBottom(true);
  }, [activeId]);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [input]);

  const updateChat = (id: string, fn: (c: Chat) => Chat) =>
    setChats((cs) => cs.map((c) => (c.id === id ? fn(c) : c)));

  const send = useCallback(async (text: string, intent?: string) => {
    const message = text.trim();
    if (!message || loading) return;
    setInput("");

    let chat = active;
    if (!chat) {
      chat = { id: uid(), title: message.slice(0, 48), siteUrl: null, siteDomain: null, messages: [], updatedAt: Date.now() };
      const fresh = chat;
      setChats((cs) => [fresh, ...cs]);
      setActiveId(fresh.id);
    }
    const chatId = chat.id;
    const userMsg: Msg = { id: uid(), role: "user", content: message, intent };
    const history = chat.messages.slice(-6).map((m) => ({
      role: m.role,
      content: (m.res?.message ?? m.content).slice(0, 600),
    }));
    updateChat(chatId, (c) => ({ ...c, messages: [...c.messages, userMsg], updatedAt: Date.now() }));

    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    try {
      const res = await sendChat(
        { message, site_url: chat.siteUrl, history, strategy, intent: intent as Intent | undefined, depth: 2 },
        controller.signal,
      );
      updateChat(chatId, (c) => ({
        ...c,
        siteUrl: res.site?.url ?? c.siteUrl,
        siteDomain: res.site?.domain ?? c.siteDomain,
        title: c.messages.length <= 1 && res.site ? res.site.title || res.site.domain : c.title,
        messages: [...c.messages, { id: uid(), role: "assistant", content: res.message, res }],
        updatedAt: Date.now(),
      }));
    } catch (e) {
      const aborted = (e as Error).name === "AbortError";
      const res: ChatResponse = {
        intent: "error", site: null, options: [], artifacts: {}, documents: [], reasoning: null,
        suggestions: aborted ? [] : ["Try again"],
        message: aborted
          ? "_Stopped._"
          : `I couldn't reach the research server.\n\n\`${(e as Error).message}\`\n\nMake sure the backend is running and try again.`,
      };
      updateChat(chatId, (c) => ({ ...c, messages: [...c.messages, { id: uid(), role: "assistant", content: res.message, res }] }));
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [active, loading, strategy]);

  const regenerate = useCallback(() => {
    if (!active || loading) return;
    const msgs = active.messages;
    const lastUserIdx = msgs.map((m) => m.role).lastIndexOf("user");
    if (lastUserIdx < 0) return;
    const lastUser = msgs[lastUserIdx];
    updateChat(active.id, (c) => ({ ...c, messages: c.messages.slice(0, lastUserIdx) }));
    send(lastUser.content, lastUser.intent);
  }, [active, loading, send]);

  const ask = useCallback((m: string, intent?: string) => {
    if (m === "Try again") return regenerate();
    send(m, intent);
  }, [regenerate, send]);

  const newChat = useCallback(() => {
    setActiveId(null);
    setInput("");
    setSidebarOpen(false);
    setTimeout(() => taRef.current?.focus(), 0);
  }, []);

  // Keyboard shortcuts: Ctrl/Cmd+K new research, "/" focuses the composer
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.closest("input, textarea");
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        newChat();
      } else if (e.key === "/" && !typing) {
        e.preventDefault();
        taRef.current?.focus();
      } else if (e.key === "Escape" && loading) {
        abortRef.current?.abort();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [newChat, loading]);

  const deleteChat = (id: string) => {
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  const clearSite = () => {
    if (active) updateChat(active.id, (c) => ({ ...c, siteUrl: null, siteDomain: null }));
  };

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    return groupChats(q ? chats.filter((c) => `${c.title} ${c.siteDomain ?? ""}`.toLowerCase().includes(q)) : chats);
  }, [chats, query]);

  const empty = !active || active.messages.length === 0;
  const siteDomain = active?.siteDomain ?? null;
  const lastAssistant = active ? [...active.messages].reverse().find((m) => m.role === "assistant")?.id : undefined;

  const composer = (
    <div className="w-full">
      {siteDomain && !empty && (
        <div className="flex gap-1.5 overflow-x-auto pb-2.5 -mx-1 px-1 no-scrollbar">
          {QUICK_ACTIONS.map((q) => (
            <button key={q.label} className="chip !py-1.5 !text-[12.5px] whitespace-nowrap" disabled={loading}
                    onClick={() => send(q.message, q.intent)}>
              {q.icon} {q.label}
            </button>
          ))}
        </div>
      )}
      <div className="glass-strong composer flex items-end gap-2">
        <span className="composer-icon" aria-hidden>{siteDomain ? <SiteIcon domain={siteDomain} size={16} /> : <Link2 size={16} />}</span>
        <textarea
          ref={taRef}
          rows={1}
          value={input}
          autoFocus
          aria-label="Message"
          placeholder={siteDomain ? `Ask anything about ${siteDomain}, or paste another link…` : "Paste a website link to start researching…"}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
        />
        {loading ? (
          <button className="send-btn" aria-label="Stop" title="Stop (Esc)" onClick={() => abortRef.current?.abort()}>
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button className="send-btn" aria-label="Send" title="Send (Enter)" disabled={!input.trim()} onClick={() => send(input)}>
            <ArrowUp size={18} strokeWidth={2.5} />
          </button>
        )}
      </div>
      <div className="flex items-center justify-center gap-3 mt-2 text-[11.5px]" style={{ color: "var(--c-faint)" }}>
        <span>Answers are grounded only in the website you provide.</span>
        <span className="hidden md:inline-flex items-center gap-1"><kbd>Enter</kbd> send</span>
        <span className="hidden md:inline-flex items-center gap-1"><kbd>Shift</kbd>+<kbd>Enter</kbd> new line</span>
      </div>
    </div>
  );

  return (
    <div className="chat-root">
      {/* Sidebar */}
      <aside className={`chat-sidebar glass-strong ${sidebarOpen ? "open" : ""}`}>
        <div className="flex items-center justify-between px-4 pt-4 pb-3">
          <Link href="/" onClick={newChat} className="flex items-center gap-2.5">
            <Mark size={36} />
            <div>
              <div className="font-bold text-[15.5px] leading-tight tracking-tight">CogNexus</div>
              <div className="text-[10px] uppercase tracking-[0.16em]" style={{ color: "var(--c-faint)" }}>Website deep research</div>
            </div>
          </Link>
          <button className="min-[901px]:hidden p-1" style={{ color: "var(--c-muted)" }} onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>
        <div className="px-3 flex flex-col gap-2">
          <button className="btn-glow w-full justify-between" onClick={newChat}>
            <span className="flex items-center gap-2"><Plus size={16} /> New research</span>
            <span className="kbd-light">Ctrl K</span>
          </button>
          <label className="search-box">
            <Search size={14} />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search conversations" aria-label="Search conversations" />
          </label>
        </div>
        <nav className="flex-1 overflow-y-auto px-2 mt-3 pb-2">
          {chats.length === 0 && (
            <div className="mx-2 mt-2 rounded-xl p-3 text-[12.5px] leading-relaxed" style={{ color: "var(--c-faint)", border: "1px dashed var(--c-border-strong)" }}>
              Your research threads will appear here. Start by pasting any public website link.
            </div>
          )}
          {chats.length > 0 && groups.length === 0 && (
            <div className="px-3 text-[12.5px]" style={{ color: "var(--c-faint)" }}>No matches.</div>
          )}
          {groups.map((g) => (
            <div key={g.label} className="mb-3">
              <div className="px-2.5 mb-1 text-[10.5px] font-semibold uppercase tracking-[0.14em]" style={{ color: "var(--c-faint)" }}>{g.label}</div>
              {g.items.map((c) => (
                <div key={c.id} className="group relative">
                  <button className={`sidebar-item pr-8 ${c.id === activeId ? "active" : ""}`}
                          onClick={() => { setActiveId(c.id); setSidebarOpen(false); }}>
                    {c.siteDomain ? <SiteIcon domain={c.siteDomain} /> : <MessageSquare size={14} className="shrink-0" />}
                    <span className="min-w-0 flex-1">
                      <span className="block truncate">{c.title}</span>
                      {c.siteDomain && <span className="block truncate text-[11px]" style={{ color: "var(--c-faint)" }}>{c.siteDomain}</span>}
                    </span>
                  </button>
                  <button className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded-md hover:bg-white/10 hover:text-white"
                          style={{ color: "var(--c-faint)" }} aria-label="Delete conversation" onClick={() => deleteChat(c.id)}>
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          ))}
        </nav>
        <div className="p-3 border-t flex flex-col gap-1" style={{ borderColor: "var(--c-border)" }}>
          <div className="flex items-center gap-2 px-2.5 py-1.5 text-[12px]" style={{ color: "var(--c-muted)" }}>
            <span className={`status-dot ${online === null ? "" : online ? "on" : "off"}`} />
            {online === null ? "Connecting to research engine…" : online ? "Research engine online" : "Research engine offline"}
          </div>
          <Link href="/about" className="sidebar-item"><BookOpen size={14} /> About the AWE framework</Link>
        </div>
      </aside>
      {sidebarOpen && <div className="fixed inset-0 z-30 bg-black/50 min-[901px]:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0">
        <header className="chat-header flex items-center gap-2 px-3 sm:px-5 py-3">
          <button className="min-[901px]:hidden p-2 rounded-lg glass" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <Menu size={18} />
          </button>
          {siteDomain ? (
            <div className="glass flex items-center gap-2 rounded-full pl-3 pr-1.5 py-1 min-w-0">
              <SiteIcon domain={siteDomain} />
              <a href={active?.siteUrl || "#"} target="_blank" rel="noopener noreferrer" className="text-[13px] font-medium truncate hover:underline">{siteDomain}</a>
              <button className="p-1 rounded-full hover:bg-white/10" style={{ color: "var(--c-muted)" }} onClick={clearSite} aria-label="Research a different site" title="Research a different site">
                <X size={13} />
              </button>
            </div>
          ) : (
            <span className="text-[13px] font-medium truncate min-w-0" style={{ color: "var(--c-muted)" }}>{active ? active.title : "New research"}</span>
          )}
          <div className="ml-auto flex items-center gap-2">
            <span className="hidden lg:inline text-[10.5px] font-semibold uppercase tracking-[0.14em]" style={{ color: "var(--c-faint)" }}>Tree-of-Thought</span>
            <div className="segmented" role="radiogroup" aria-label="Tree-of-Thought search strategy">
              {STRATEGIES.map((s) => (
                <button key={s.id} role="radio" aria-checked={strategy === s.id} title={s.hint}
                        className={strategy === s.id ? "on" : ""} onClick={() => setStrategy(s.id)}>{s.label}</button>
              ))}
            </div>
            {active && active.messages.length > 0 && (
              <button className="icon-btn glass !rounded-lg hidden sm:grid" title="Export conversation (.md)" aria-label="Export conversation"
                      onClick={() => downloadDocument({ name: `${active.siteDomain || "cognexus"}-conversation.md`, mime: "text/markdown", content: chatToMarkdown(active) })}>
                <Download size={15} />
              </button>
            )}
          </div>
        </header>

        {empty ? (
          <div className="flex-1 overflow-y-auto px-4">
            <div className="w-full max-w-3xl mx-auto flex flex-col items-center text-center pt-[6vh] pb-10">
              <Image src="/brand/cognexus-logo.png" alt="CogNexus" width={132} height={132} priority className="logo-hero mb-6" />
              <h1 className="hero-title text-[32px] sm:text-[44px] font-bold tracking-tight leading-[1.1] mb-3">
                Research any website,<br className="hidden sm:block" /> <span className="hero-accent">deeply.</span>
              </h1>
              <p className="text-[15px] mb-8 max-w-xl" style={{ color: "var(--c-muted)" }}>
                Paste a link. Agents crawl it, map its structure, audit its security and answer with
                Tree-of-Thought options ranked by confidence.
              </p>
              <div className="w-full max-w-2xl">{composer}</div>
              <div className="cap-grid w-full mt-9">
                {CAPABILITIES.map((c) => (
                  <button key={c.title} className="cap-card" onClick={() => send(c.prompt)} style={{ "--tint": c.tint } as React.CSSProperties}>
                    <span className="cap-icon">{c.icon}</span>
                    <span className="block text-[14px] font-semibold mt-3">{c.title}</span>
                    <span className="block text-[12.5px] mt-1 leading-snug" style={{ color: "var(--c-muted)" }}>{c.desc}</span>
                    <span className="cap-prompt">“{c.prompt}”</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            <div
              ref={scrollRef}
              className="chat-scroll flex-1 overflow-y-auto px-3 sm:px-6"
              onScroll={(e) => {
                const el = e.currentTarget;
                setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 160);
              }}
            >
              <div className="max-w-3xl mx-auto py-6 flex flex-col gap-8">
                {active!.messages.map((m) =>
                  m.role === "user" ? (
                    <div key={m.id} className="msg-user fade-in">{m.content}</div>
                  ) : (
                    <AssistantMessage
                      key={m.id}
                      res={m.res!}
                      ask={ask}
                      isLast={m.id === lastAssistant && !loading}
                      onRegenerate={m.id === lastAssistant && !loading ? regenerate : undefined}
                    />
                  ),
                )}
                {loading && <Thinking strategy={strategy} cached={!!siteDomain} />}
              </div>
            </div>
            <div className="relative px-3 sm:px-6 pb-4 pt-2">
              {!atBottom && (
                <button className="scroll-fab" onClick={() => scrollToBottom()} aria-label="Scroll to latest">
                  <ArrowDown size={16} />
                </button>
              )}
              <div className="max-w-3xl mx-auto">{composer}</div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
