"use client";

import { useMemo, useState } from "react";
import Image from "next/image";
import {
  Brain, ChevronDown, ChevronRight, Copy, Check, Download, ShieldAlert, Workflow, Sparkles,
  RefreshCcw, FileDown, FileText,
} from "lucide-react";
import Markdown from "./Markdown";
import ArtifactsView from "./Artifacts";
import { downloadDocument, type ChatResponse, type ThoughtNode } from "../../lib/chat";

type Ask = (message: string, intent?: string) => void;

const INTENT_LABEL: Record<string, string> = {
  ask_url: "Getting started", overview: "Site overview", qa: "Research answer", security: "Security audit",
  structure: "Structure analysis", similar: "Similar websites", compare: "Comparison", extract: "Data extraction",
  report: "Research report", error: "Error",
};

const STRATEGY_LABEL: Record<string, string> = { beam: "Beam search", bfs: "Breadth-first", dfs: "Depth-first" };

function ThoughtTree({ nodes, selectedId }: { nodes: ThoughtNode[]; selectedId: string | null }) {
  const children = useMemo(() => {
    const m = new Map<string | null, ThoughtNode[]>();
    nodes.forEach((n) => m.set(n.parent_id, [...(m.get(n.parent_id) || []), n]));
    m.forEach((list) => list.sort((a, b) => b.score - a.score));
    return m;
  }, [nodes]);

  const render = (n: ThoughtNode) => {
    const kids = children.get(n.id) || [];
    const isSel = n.id === selectedId;
    const statusBadge =
      isSel ? <span className="badge badge-pass">chosen</span>
        : n.status === "pruned" ? <span className="badge badge-critical">pruned</span>
          : n.status === "goal" ? <span className="badge badge-pass">goal</span>
            : n.status === "expanded" ? <span className="badge badge-accent">expanded</span>
              : <span className="badge badge-muted">leaf</span>;
    return (
      <div key={n.id}>
        <div className={`tree-node ${isSel ? "selected" : n.status}`} title={n.critique}>
          {n.order > 0 && <span className="tabular-nums text-[10px]" style={{ color: "var(--c-faint)" }}>#{n.order}</span>}
          <span className="flex-1 truncate">{n.approach}</span>
          <span className="tabular-nums font-semibold" style={{ color: "#d6cfff" }}>{Math.round(n.score * 100)}%</span>
          {statusBadge}
        </div>
        {kids.length > 0 && <div className="tree-children">{kids.map(render)}</div>}
      </div>
    );
  };

  const roots = children.get(null) || [];
  return <div className="flex flex-col gap-1.5">{roots.map(render)}</div>;
}

function ReasoningPanel({ r, selectedId }: { r: NonNullable<ChatResponse["reasoning"]>; selectedId: string | null }) {
  const [open, setOpen] = useState(false);
  const secs = (r.duration_ms / 1000).toFixed(r.duration_ms < 10000 ? 1 : 0);
  return (
    <div className="mb-3">
      <button onClick={() => setOpen(!open)} className="flex items-center gap-2 text-[13px] hover:text-white transition" style={{ color: "var(--c-muted)" }}>
        <Brain size={14} />
        <span>Researched for {secs}s</span>
        <span style={{ color: "var(--c-faint)" }}>· {STRATEGY_LABEL[r.strategy] || r.strategy} · {r.tree.length} thoughts · {r.llm_calls} model calls</span>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {open && (
        <div className="glass card mt-2 fade-in grid md:grid-cols-2 gap-4">
          <div>
            <div className="section-title"><Workflow size={13} /> Agent pipeline</div>
            <ol className="relative flex flex-col gap-2.5 pl-4" style={{ borderLeft: "1px solid var(--c-border-strong)" }}>
              {r.steps.map((s, i) => (
                <li key={i} className="text-[12.5px] relative">
                  <span className="absolute -left-[21px] top-1.5 w-2 h-2 rounded-full" style={{ background: "var(--c-accent)" }} />
                  <div className="flex gap-2 items-baseline">
                    <span className="font-semibold text-white">{s.agent}</span>
                    <span className="tabular-nums text-[10.5px]" style={{ color: "var(--c-faint)" }}>{(s.t_ms / 1000).toFixed(1)}s</span>
                  </div>
                  <div style={{ color: "var(--c-muted)" }}>{s.action}</div>
                  {s.detail && <div className="truncate text-[11.5px]" style={{ color: "var(--c-faint)" }} title={s.detail}>{s.detail}</div>}
                </li>
              ))}
            </ol>
          </div>
          <div>
            <div className="section-title"><Sparkles size={13} /> Tree of thoughts</div>
            {r.tree.length ? <ThoughtTree nodes={r.tree} selectedId={selectedId} /> : (
              <div className="text-[13px]" style={{ color: "var(--c-muted)" }}>No model reasoning was needed for this reply.</div>
            )}
            <p className="text-[11.5px] mt-3 leading-relaxed" style={{ color: "var(--c-faint)" }}>
              Score = 30% groundedness + 25% relevance + 20% completeness (model evaluator)
              + 15% evidence overlap with crawled pages + 10% self-confidence.
              {r.model && <> Model: {r.model}.</>}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AssistantMessage({ res, ask, isLast, onRegenerate }: {
  res: ChatResponse;
  ask: Ask;
  isLast: boolean;
  onRegenerate?: () => void;
}) {
  const [sel, setSel] = useState(0);
  const [copied, setCopied] = useState(false);
  const [why, setWhy] = useState(false);
  const opts = res.options || [];
  const current = opts[sel];
  const answer = current ? current.answer : res.message;

  return (
    <div className="flex gap-3 fade-in" data-role="assistant">
      <div className="avatar logo-mark mt-0.5"><Image src="/brand/cognexus-mark.png" alt="CogNexus" width={64} height={64} className="w-full h-full object-cover" /></div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          <span className="font-semibold text-[14px]">CogNexus</span>
          <span className="badge badge-muted">{INTENT_LABEL[res.intent] || res.intent}</span>
          {res.site && <span className="badge badge-accent">{res.site.domain}</span>}
        </div>

        {res.reasoning && res.reasoning.tree.length > 0 && <ReasoningPanel r={res.reasoning} selectedId={current?.node_id ?? null} />}

        {res.guardrail && !res.guardrail.allowed && (
          <div className="flex items-center gap-2 text-[12.5px] mb-2" style={{ color: "var(--c-warn)" }}>
            <ShieldAlert size={14} /> Guardrail: {res.guardrail.reason === "prompt_injection" ? "instruction override blocked" : "outside this website's scope"}
          </div>
        )}

        {opts.length > 1 && (
          <div className="mb-3">
            <div className="text-[12.5px] mb-2" style={{ color: "var(--c-muted)" }}>
              Tree-of-Thought produced {opts.length} alternatives. Choose an option. <b className="text-white">Option 1</b> scored highest.
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {opts.map((o, i) => (
                <button key={i} className={`option-tab ${i === sel ? "on" : ""}`} onClick={() => setSel(i)}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[12.5px] font-bold">Option {o.rank}{i === 0 ? " · Best" : ""}</span>
                    <span className="text-[12px] tabular-nums font-semibold" style={{ color: "#d6cfff" }}>{o.confidence}%</span>
                  </div>
                  <div className="conf-bar"><span style={{ width: `${o.confidence}%` }} /></div>
                  <span className="text-[11.5px] truncate max-w-[180px]" style={{ color: "var(--c-muted)" }}>{o.approach}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className={opts.length > 1 ? "glass card" : ""}>
          <Markdown>{answer || "_No answer._"}</Markdown>
          {current && (current.rationale || current.critique || current.evidence.length > 0) && (
            <div className="mt-3 pt-3" style={{ borderTop: "1px solid var(--c-border)" }}>
              <button className="flex items-center gap-1.5 text-[12.5px] hover:text-white" style={{ color: "var(--c-muted)" }} onClick={() => setWhy(!why)}>
                {why ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                Why option {current.rank}? <span style={{ color: "var(--c-faint)" }}>({current.approach}, confidence {current.confidence}%)</span>
              </button>
              {why && (
                <div className="mt-2 text-[12.5px] space-y-2" style={{ color: "var(--c-muted)" }}>
                  {current.rationale && <div><b className="text-white">Approach:</b> {current.rationale}</div>}
                  {current.critique && <div><b className="text-white">Evaluator critique:</b> {current.critique}</div>}
                  {current.evidence.length > 0 && (
                    <div>
                      <b className="text-white">Evidence from the site:</b>
                      <ul className="mt-1 space-y-1">
                        {current.evidence.map((e, i) => <li key={i} className="pl-2" style={{ borderLeft: "2px solid var(--c-accent)" }}>“{e}”</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="msg-actions flex items-center gap-0.5 mt-2">
          <button className="icon-btn" title={copied ? "Copied" : "Copy answer"} aria-label="Copy answer" onClick={() => {
            navigator.clipboard?.writeText(answer);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}>
            {copied ? <Check size={15} /> : <Copy size={15} />}
          </button>
          <button className="icon-btn" title="Download answer (.md)" aria-label="Download answer" onClick={() => downloadDocument({
            name: `${res.site?.domain || "cognexus"}-answer.md`, mime: "text/markdown", content: answer,
          })}>
            <FileDown size={15} />
          </button>
          {onRegenerate && (
            <button className="icon-btn" title="Regenerate" aria-label="Regenerate" onClick={onRegenerate}>
              <RefreshCcw size={15} />
            </button>
          )}
          {current && opts.length > 1 && (
            <span className="ml-2 text-[11.5px]" style={{ color: "var(--c-faint)" }}>
              Showing option {current.rank} of {opts.length}
            </span>
          )}
        </div>

        {res.documents.length > 0 && (
          <div className="grid sm:grid-cols-2 gap-2 mt-3">
            {res.documents.map((d) => (
              <button key={d.name} className="file-card" onClick={() => downloadDocument(d)}>
                <span className="file-icon"><FileText size={16} /></span>
                <span className="min-w-0 flex-1 text-left">
                  <span className="block text-[13px] font-medium truncate">{d.name}</span>
                  <span className="block text-[11px]" style={{ color: "var(--c-faint)" }}>
                    {d.name.split(".").pop()?.toUpperCase()} · {(new Blob([d.content]).size / 1024).toFixed(1)} KB
                  </span>
                </span>
                <Download size={15} style={{ color: "var(--c-muted)" }} />
              </button>
            ))}
          </div>
        )}

        {Object.keys(res.artifacts || {}).length > 0 && (
          <div className="mt-4"><ArtifactsView artifacts={res.artifacts} ask={ask} /></div>
        )}

        {isLast && res.suggestions?.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-4">
            {res.suggestions.map((s) => <button key={s} className="chip" onClick={() => ask(s)}>{s}</button>)}
          </div>
        )}
      </div>
    </div>
  );
}
