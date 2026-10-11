"use client";

import "./chat.css";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  Plus, MessageSquare, Menu, X, ArrowUp, Globe, ShieldCheck, Network, Sparkles,
  GitCompareArrows, FileText, Table2, Trash2, Square, Search, BookOpen,
} from "lucide-react";
import AssistantMessage from "./components/chat/AssistantMessage";
import { sendChat, type ChatResponse, type Strategy, type Intent } from "./lib/chat";

interface Msg {
  id: string;
  role: "user" | "assistant";
  content: string;
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
  { id: "beam", label: "Beam", hint: "Keep the top-k thoughts at every depth (balanced)" },
  { id: "bfs", label: "BFS", hint: "Expand every viable thought level by level (broadest)" },
  { id: "dfs", label: "DFS", hint: "Follow the best branch deep, backtrack on failure (fastest to a goal)" },
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

const EXAMPLES = [
  "https://quotes.toscrape.com",
  "stripe.com",
  "vercel.com",
  "Compare github.com with gitlab.com",
];

const STAGES: [number, string][] = [
  [0, "Routing request & checking guardrails"],
  [2500, "Observer is crawling key pages"],
  [7000, "Analysing structure, tech stack & security"],
  [14000, "Generating candidate thoughts"],
  [19000, "Evaluating & pruning branches"],
  [24000, "Expanding the most promising branches"],
  [34000, "Re-scoring refinements & ranking options"],
];

function Thinking({ strategy, cached }: { strategy: Strategy; cached: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const t = setInterval(() => setElapsed(Date.now() - start), 250);
    return () => clearInterval(t);
  }, []);
  // A site that is already researched skips the crawl stages
  const offset = cached ? 14000 : 0;
  const stage = [...STAGES].reverse().find(([t]) => elapsed + offset >= t)?.[1] ?? STAGES[0][1];
  return (
    <div className="flex gap-3 fade-in">
      <div className="avatar"><Sparkles size={16} color="#fff" /></div>
      <div className="flex items-center gap-3 pt-1.5">
        <span className="pulse-dot" />
        <span className="shimmer-text text-[14px] font-medium">{stage}…</span>
        <span className="text-[12px] tabular-nums" style={{ color: "var(--c-faint)" }}>
          {(elapsed / 1000).toFixed(0)}s · {strategy.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

export default function ChatApp() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [strategy, setStrategy] = useState<Strategy>("beam");
  const [sidebarOpen, setSidebarOpen] = useState(false);
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

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [active?.messages.length, loading]);

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
    const userMsg: Msg = { id: uid(), role: "user", content: message };
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

  const ask = useCallback((m: string, intent?: string) => {
    if (m === "Try again") {
      const lastUser = [...(active?.messages || [])].reverse().find((x) => x.role === "user");
      if (lastUser) send(lastUser.content);
      return;
    }
    send(m, intent);
  }, [active, send]);

  const newChat = () => {
    setActiveId(null);
    setInput("");
    setSidebarOpen(false);
    taRef.current?.focus();
  };

  const deleteChat = (id: string) => {
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  const clearSite = () => {
    if (active) updateChat(active.id, (c) => ({ ...c, siteUrl: null, siteDomain: null }));
  };

  const empty = !active || active.messages.length === 0;
  const siteDomain = active?.siteDomain;
  const lastAssistant = active ? [...active.messages].reverse().find((m) => m.role === "assistant")?.id : undefined;

  const composer = (
    <div className="w-full">
      {siteDomain && !empty && (
        <div className="flex gap-1.5 overflow-x-auto pb-2 -mx-1 px-1">
          {QUICK_ACTIONS.map((q) => (
            <button key={q.label} className="chip !py-1.5 !text-[12.5px] whitespace-nowrap" disabled={loading}
                    onClick={() => send(q.message, q.intent)}>
              {q.icon} {q.label}
            </button>
          ))}
        </div>
      )}
      <div className="glass-strong composer flex items-end gap-2">
        <textarea
          ref={taRef}
          rows={1}
          value={input}
          autoFocus
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
          <button className="btn-glow !p-2.5 !rounded-full" aria-label="Stop" onClick={() => abortRef.current?.abort()}>
            <Square size={16} fill="#fff" />
          </button>
        ) : (
          <button className="btn-glow !p-2.5 !rounded-full" aria-label="Send" disabled={!input.trim()} onClick={() => send(input)}>
            <ArrowUp size={18} />
          </button>
        )}
      </div>
      <p className="text-center text-[11.5px] mt-2" style={{ color: "var(--c-faint)" }}>
        Answers are grounded only in the website you provide. Each reply shows ranked Tree-of-Thought options with confidence scores.
      </p>
    </div>
  );

  return (
    <div className="chat-root">
      {/* Sidebar */}
      <aside className={`chat-sidebar glass-strong ${sidebarOpen ? "open" : ""}`}>
        <div className="flex items-center justify-between px-4 pt-4 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="avatar !w-8 !h-8"><Sparkles size={16} color="#fff" /></div>
            <div>
              <div className="font-bold text-[15px] leading-tight">CogNexus</div>
              <div className="text-[10.5px] uppercase tracking-widest" style={{ color: "var(--c-faint)" }}>Website deep research</div>
            </div>
          </div>
          <button className="min-[901px]:hidden p-1" style={{ color: "var(--c-muted)" }} onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>
        <div className="px-3">
          <button className="btn-glow w-full" onClick={newChat}><Plus size={16} /> New research</button>
        </div>
        <div className="flex-1 overflow-y-auto px-2 mt-4">
          <div className="px-2 mb-1.5 text-[11px] uppercase tracking-widest" style={{ color: "var(--c-faint)" }}>Recent</div>
          {chats.length === 0 && <div className="px-2 text-[13px]" style={{ color: "var(--c-faint)" }}>No conversations yet.</div>}
          {chats.map((c) => (
            <div key={c.id} className="group relative">
              <button className={`sidebar-item pr-8 ${c.id === activeId ? "active" : ""}`}
                      onClick={() => { setActiveId(c.id); setSidebarOpen(false); }}>
                {c.siteDomain ? <Globe size={14} className="shrink-0" /> : <MessageSquare size={14} className="shrink-0" />}
                <span className="truncate">{c.title}</span>
              </button>
              <button className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 p-1 hover:text-white"
                      style={{ color: "var(--c-faint)" }} aria-label="Delete conversation" onClick={() => deleteChat(c.id)}>
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
        <div className="p-3 border-t" style={{ borderColor: "var(--c-border)" }}>
          <Link href="/about" className="sidebar-item"><BookOpen size={14} /> About the AWE framework</Link>
        </div>
      </aside>
      {sidebarOpen && <div className="fixed inset-0 z-30 bg-black/50 min-[901px]:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center gap-2 px-3 sm:px-5 py-3">
          <button className="min-[901px]:hidden p-2 rounded-lg glass" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <Menu size={18} />
          </button>
          {siteDomain ? (
            <div className="glass flex items-center gap-2 rounded-full pl-3 pr-1.5 py-1 min-w-0">
              <Globe size={14} style={{ color: "var(--c-accent)" }} />
              <a href={active?.siteUrl || "#"} target="_blank" rel="noopener noreferrer" className="text-[13px] font-medium truncate hover:underline">{siteDomain}</a>
              <button className="p-1 rounded-full hover:bg-white/10" style={{ color: "var(--c-muted)" }} onClick={clearSite} aria-label="Research a different site" title="Research a different site">
                <X size={13} />
              </button>
            </div>
          ) : (
            <span className="text-[13px] font-medium" style={{ color: "var(--c-muted)" }}>New research</span>
          )}
          <div className="ml-auto flex items-center gap-2">
            <span className="hidden sm:inline text-[11px] uppercase tracking-widest" style={{ color: "var(--c-faint)" }}>Tree-of-Thought</span>
            <div className="segmented">
              {STRATEGIES.map((s) => (
                <button key={s.id} title={s.hint} className={strategy === s.id ? "on" : ""} onClick={() => setStrategy(s.id)}>{s.label}</button>
              ))}
            </div>
          </div>
        </header>

        {empty ? (
          <div className="flex-1 overflow-y-auto flex flex-col items-center justify-center px-4 pb-10">
            <div className="w-full max-w-2xl flex flex-col items-center text-center">
              <div className="avatar !w-14 !h-14 !rounded-2xl mb-5"><Sparkles size={26} color="#fff" /></div>
              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">Which website should we research?</h1>
              <p className="text-[15px] mb-8 max-w-xl" style={{ color: "var(--c-muted)" }}>
                Paste a link and CogNexus agents crawl it, map its internal structure, audit its security, find similar sites,
                and answer with ranked options you can choose from.
              </p>
              {composer}
              <div className="flex flex-wrap justify-center gap-2 mt-5">
                {EXAMPLES.map((e) => <button key={e} className="chip" onClick={() => send(e)}>{e}</button>)}
              </div>
            </div>
          </div>
        ) : (
          <>
            <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 sm:px-6">
              <div className="max-w-3xl mx-auto py-4 flex flex-col gap-7">
                {active!.messages.map((m) =>
                  m.role === "user" ? (
                    <div key={m.id} className="msg-user fade-in">{m.content}</div>
                  ) : (
                    <AssistantMessage key={m.id} res={m.res!} ask={ask} isLast={m.id === lastAssistant && !loading} />
                  ),
                )}
                {loading && <Thinking strategy={strategy} cached={!!siteDomain} />}
              </div>
            </div>
            <div className="px-3 sm:px-6 pb-4 pt-2">
              <div className="max-w-3xl mx-auto">{composer}</div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
