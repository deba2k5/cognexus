"use client";

import { useState } from "react";
import {
  ShieldCheck, ShieldAlert, Network, Sparkles, GitCompareArrows, Table2, FileText,
  ChevronDown, ChevronRight, Copy, Check, ExternalLink, Globe, Layers, Search,
} from "lucide-react";
import type { Artifacts, Finding, SimilarSite, SiteDocument } from "../../lib/chat";

type Ask = (message: string, intent?: string) => void;

function Grade({ grade }: { grade?: string }) {
  const g = grade || "?";
  return <div className={`grade grade-${g}`}>{g}</div>;
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl px-3 py-2.5" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid var(--c-border)" }}>
      <div className="text-[11px] uppercase tracking-wider" style={{ color: "var(--c-faint)" }}>{label}</div>
      <div className="text-[17px] font-bold mt-0.5 tabular-nums">{value}</div>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="btn-ghost !py-1 !px-2 !text-[11px]"
      onClick={() => {
        navigator.clipboard?.writeText(text);
        setDone(true);
        setTimeout(() => setDone(false), 1500);
      }}
    >
      {done ? <Check size={12} /> : <Copy size={12} />} {done ? "Copied" : "Copy"}
    </button>
  );
}

function Favicon({ domain, size = 18 }: { domain: string; size?: number }) {
  const [ok, setOk] = useState(true);
  if (!ok) return <Globe size={size} style={{ color: "var(--c-muted)" }} />;
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`https://${domain}/favicon.ico`}
      alt=""
      width={size}
      height={size}
      className="rounded"
      onError={() => setOk(false)}
    />
  );
}

/* ---------------- Overview ---------------- */

function pagePath(url: string) {
  try {
    const u = new URL(url);
    return u.pathname === "/" ? u.hostname : u.pathname;
  } catch {
    return url;
  }
}

function OverviewPanel({ a, ask }: { a: NonNullable<Artifacts["overview"]>; ask: Ask }) {
  const socials = Object.entries(a.contacts.socials || {});
  return (
    <div className="glass card fade-in">
      <div className="section-title"><Search size={14} /> Research snapshot</div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <button onClick={() => ask("Run a security audit with fixes", "security")} className="text-left">
          <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 h-full hover:brightness-125 transition" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid var(--c-border)" }}>
            <Grade grade={a.security.grade} />
            <div><div className="text-[11px] uppercase tracking-wider" style={{ color: "var(--c-faint)" }}>Security</div><div className="font-bold">{a.security.score}/100</div></div>
          </div>
        </button>
        <button onClick={() => ask("Show the internal structure and sitemap", "structure")} className="text-left">
          <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 h-full hover:brightness-125 transition" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid var(--c-border)" }}>
            <Grade grade={a.structure.grade} />
            <div><div className="text-[11px] uppercase tracking-wider" style={{ color: "var(--c-faint)" }}>Structure</div><div className="font-bold">{a.structure.score}/100</div></div>
          </div>
        </button>
        <Stat label="Response" value={`${a.response_ms} ms`} />
        <Stat label="Words analysed" value={a.word_count.toLocaleString()} />
      </div>

      {a.tech.length > 0 && (
        <div className="mt-4">
          <div className="text-xs mb-2" style={{ color: "var(--c-faint)" }}>Technology detected</div>
          <div className="flex flex-wrap gap-1.5">{a.tech.map((t) => <span key={t} className="badge badge-accent">{t}</span>)}</div>
        </div>
      )}

      <div className="grid sm:grid-cols-2 gap-4 mt-4">
        <div>
          <div className="text-xs mb-2" style={{ color: "var(--c-faint)" }}>Pages crawled ({a.pages.length})</div>
          <ul className="space-y-1 text-[13px]">
            {a.pages.map((p) => (
              <li key={p.url} className="truncate">
                <a href={p.url} target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: "#c9c0ff" }}>{pagePath(p.url)}</a>
                <span className="ml-2" style={{ color: "var(--c-faint)" }}>{p.title}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <div className="text-xs mb-2" style={{ color: "var(--c-faint)" }}>Contacts & presence</div>
          <div className="text-[13px] space-y-1" style={{ color: "var(--c-muted)" }}>
            {a.contacts.emails.slice(0, 3).map((e) => <div key={e} className="truncate">✉ {e}</div>)}
            {a.contacts.phones.slice(0, 2).map((p) => <div key={p}>☎ {p}</div>)}
            {socials.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-1">
                {socials.map(([k, v]) => (
                  <a key={k} href={v} target="_blank" rel="noopener noreferrer" className="badge badge-muted hover:text-white">{k}</a>
                ))}
              </div>
            )}
            {!a.contacts.emails.length && !a.contacts.phones.length && !socials.length && <div>No public contact details found.</div>}
          </div>
        </div>
      </div>
      {a.documents.length > 0 && <SiteDocsList docs={a.documents} />}
    </div>
  );
}

function SiteDocsList({ docs }: { docs: SiteDocument[] }) {
  return (
    <div className="mt-4">
      <div className="text-xs mb-2" style={{ color: "var(--c-faint)" }}>Documents published on the site ({docs.length})</div>
      <div className="flex flex-col gap-1.5">
        {docs.map((d) => (
          <a key={d.url} href={d.url} target="_blank" rel="noopener noreferrer"
             className="flex items-center gap-2 text-[13px] rounded-lg px-2.5 py-1.5 hover:bg-white/5" style={{ border: "1px solid var(--c-border)" }}>
            <FileText size={14} style={{ color: "var(--c-accent)" }} />
            <span className="truncate flex-1">{d.title}</span>
            <span className="badge badge-muted">{d.type}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

/* ---------------- Security ---------------- */

function FindingRow({ f }: { f: Finding }) {
  const [open, setOpen] = useState(f.severity === "critical");
  return (
    <div className="rounded-xl" style={{ border: "1px solid var(--c-border)", background: "rgba(255,255,255,0.025)" }}>
      <button className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        <span className={`badge badge-${f.severity}`}>{f.severity}</span>
        <span className="text-[13.5px] font-medium flex-1">{f.message}</span>
        <span className="text-[11px] hidden sm:inline" style={{ color: "var(--c-faint)" }}>{f.check}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pl-9 text-[13px] space-y-2">
          {f.detail && <div style={{ color: "var(--c-muted)" }}>{f.detail}</div>}
          {f.fix && (
            <div className="rounded-lg p-2.5" style={{ background: "rgba(94,230,168,0.06)", border: "1px solid rgba(94,230,168,0.22)" }}>
              <div className="flex items-start justify-between gap-2">
                <div><span className="font-semibold" style={{ color: "var(--c-good)" }}>Fix: </span>{f.fix.summary}</div>
                {f.fix.snippet && <CopyButton text={f.fix.snippet} />}
              </div>
              {f.fix.snippet && (
                <pre className="mt-2 text-[12px] overflow-x-auto rounded-md p-2" style={{ background: "rgba(4,4,18,0.7)", fontFamily: "var(--font-geist-mono)" }}>{f.fix.snippet}</pre>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SecurityPanel({ a }: { a: NonNullable<Artifacts["security"]> }) {
  const [showPassed, setShowPassed] = useState(false);
  const s = a.summary || {};
  return (
    <div className="glass card fade-in">
      <div className="section-title">{(a.score ?? 0) >= 70 ? <ShieldCheck size={14} /> : <ShieldAlert size={14} />} Security audit</div>
      <div className="flex items-center gap-4 mb-4">
        <Grade grade={a.grade} />
        <div className="flex-1">
          <div className="text-sm" style={{ color: "var(--c-muted)" }}>Score <b className="text-white text-lg">{a.score ?? "—"}</b>/100</div>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            <span className="badge badge-critical">{s.critical ?? 0} critical</span>
            <span className="badge badge-warning">{s.warning ?? 0} warnings</span>
            <span className="badge badge-info">{s.info ?? 0} info</span>
            <span className="badge badge-pass">{s.pass ?? 0} passed</span>
          </div>
        </div>
      </div>
      {a.error && <div className="text-sm mb-3" style={{ color: "var(--c-bad)" }}>{a.error}</div>}
      <div className="flex flex-col gap-2">
        {a.findings.map((f, i) => <FindingRow key={i} f={f} />)}
        {a.findings.length === 0 && <div className="text-sm" style={{ color: "var(--c-good)" }}>No issues detected.</div>}
      </div>
      {a.passed.length > 0 && (
        <div className="mt-3">
          <button className="btn-ghost" onClick={() => setShowPassed(!showPassed)}>
            {showPassed ? <ChevronDown size={14} /> : <ChevronRight size={14} />} {a.passed.length} passing checks
          </button>
          {showPassed && (
            <ul className="mt-2 space-y-1 text-[13px]" style={{ color: "var(--c-muted)" }}>
              {a.passed.map((p, i) => <li key={i}><span className="badge badge-pass mr-2">pass</span>{p.message}</li>)}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------------- Structure ---------------- */

function StructurePanel({ a }: { a: NonNullable<Artifacts["structure"]> }) {
  const issues = [...(a.headings.issues || []), ...(a.metadata.seo_issues || [])];
  return (
    <div className="glass card fade-in">
      <div className="section-title"><Network size={14} /> Internal structure</div>
      <div className="flex items-center gap-4 mb-4">
        <Grade grade={a.grade} />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 flex-1">
          <Stat label="DOM depth" value={a.dom.max_depth} />
          <Stat label="Elements" value={a.dom.total_elements.toLocaleString()} />
          <Stat label="Int / Ext links" value={`${a.links.internal_links} / ${a.links.external_links}`} />
          <Stat label="Sitemap paths" value={a.sitemap.total_paths} />
        </div>
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <div className="text-xs mb-2 flex items-center gap-1.5" style={{ color: "var(--c-faint)" }}><Layers size={12} /> Sitemap tree</div>
          <div className="rounded-xl p-3 max-h-72 overflow-auto text-[12.5px]" style={{ background: "rgba(4,4,18,0.55)", border: "1px solid var(--c-border)", fontFamily: "var(--font-geist-mono)" }}>
            {a.sitemap.tree.map((n, i) => (
              <div key={i} className="truncate" style={{ paddingLeft: n.depth * 14 }}>
                <span style={{ color: "var(--c-faint)" }}>{n.depth ? "└ " : ""}</span>
                <a href={n.url} target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: n.depth ? "#cdc6ff" : "#fff" }}>
                  /{n.path.replace(/^\/|\/$/g, "")}
                </a>
                {n.children_count > 0 && <span style={{ color: "var(--c-faint)" }}> ({n.children_count})</span>}
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs mb-2" style={{ color: "var(--c-faint)" }}>Heading outline</div>
          <div className="rounded-xl p-3 max-h-72 overflow-auto text-[13px] space-y-1" style={{ background: "rgba(4,4,18,0.55)", border: "1px solid var(--c-border)" }}>
            {a.headings.headings.slice(0, 30).map((h, i) => (
              <div key={i} className="truncate" style={{ paddingLeft: (h.level - 1) * 12 }}>
                <span className="badge badge-muted mr-1.5 !text-[10px]">{h.tag}</span>{h.text}
              </div>
            ))}
            {a.headings.headings.length === 0 && <div style={{ color: "var(--c-muted)" }}>No headings found.</div>}
          </div>
        </div>
      </div>
      <div className="grid sm:grid-cols-3 gap-2 mt-4">
        <Stat label="Scripts / Styles" value={`${a.resources.scripts.total} / ${a.resources.stylesheets.total}`} />
        <Stat label="Images (alt ok)" value={`${a.resources.images.total} (${a.resources.images.accessibility_score})`} />
        <Stat label="Complexity" value={a.dom.complexity} />
      </div>
      {issues.length > 0 && (
        <div className="mt-4">
          <div className="text-xs mb-2" style={{ color: "var(--c-faint)" }}>SEO & semantic issues</div>
          <ul className="space-y-1.5 text-[13px]">
            {issues.map((i, k) => (
              <li key={k} className="flex gap-2 items-start"><span className={`badge badge-${i.severity}`}>{i.severity}</span><span>{i.message}</span></li>
            ))}
          </ul>
        </div>
      )}
      {a.links.external_domains?.length > 0 && (
        <div className="mt-4">
          <div className="text-xs mb-2" style={{ color: "var(--c-faint)" }}>Top external domains</div>
          <div className="flex flex-wrap gap-1.5">
            {a.links.external_domains.map((d) => <span key={d.domain} className="badge badge-muted">{d.domain} · {d.count}</span>)}
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------------- Similar sites ---------------- */

function SimilarCard({ s, rank, ask }: { s: SimilarSite; rank: number; ask: Ask }) {
  return (
    <div className="rounded-xl p-3 flex flex-col gap-2" style={{ border: "1px solid var(--c-border)", background: "rgba(255,255,255,0.03)" }}>
      <div className="flex items-center gap-2">
        <Favicon domain={s.domain} />
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-[14px] truncate">{s.name}</div>
          <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-[12px] hover:underline flex items-center gap-1" style={{ color: "#b9adff" }}>
            {s.domain} <ExternalLink size={10} />
          </a>
        </div>
        <span className="badge badge-accent">#{rank}</span>
      </div>
      <div>
        <div className="flex justify-between text-[11px] mb-1" style={{ color: "var(--c-faint)" }}>
          <span>Similarity</span><span className="tabular-nums">{s.similarity}%</span>
        </div>
        <div className="conf-bar"><span style={{ width: `${s.similarity}%` }} /></div>
      </div>
      <p className="text-[12.5px]" style={{ color: "var(--c-muted)" }}>{s.why}</p>
      {s.difference && <p className="text-[12px]" style={{ color: "var(--c-faint)" }}><b>Difference:</b> {s.difference}</p>}
      <div className="flex items-center gap-1.5 mt-auto pt-1 flex-wrap">
        <span className={`badge ${s.reachable ? "badge-pass" : "badge-critical"}`}>{s.reachable ? (s.status && s.status >= 400 ? "Live · bot-protected" : "Live") : "Unreachable"}</span>
        <button className="btn-ghost !py-1 !px-2 !text-[12px] ml-auto" onClick={() => ask(`compare with ${s.url}`, "compare")}>
          <GitCompareArrows size={12} /> Compare
        </button>
        <button className="btn-ghost !py-1 !px-2 !text-[12px]" onClick={() => ask(s.url, "overview")}>
          <Search size={12} /> Research
        </button>
      </div>
    </div>
  );
}

function SimilarPanel({ a, ask }: { a: NonNullable<Artifacts["similar"]>; ask: Ask }) {
  return (
    <div className="glass card fade-in">
      <div className="section-title"><Sparkles size={14} /> Similar websites {a.category && <span className="badge badge-muted normal-case tracking-normal">{a.category}</span>}</div>
      <div className="grid sm:grid-cols-2 gap-2.5">
        {a.sites.map((s, i) => <SimilarCard key={s.url} s={s} rank={i + 1} ask={ask} />)}
      </div>
      {a.sites.length === 0 && <div className="text-sm" style={{ color: "var(--c-muted)" }}>No similar websites could be identified.</div>}
    </div>
  );
}

/* ---------------- Comparison ---------------- */

function ComparisonPanel({ a }: { a: NonNullable<Artifacts["comparison"]> }) {
  return (
    <div className="glass card fade-in">
      <div className="section-title"><GitCompareArrows size={14} /> Side-by-side comparison</div>
      <div className="overflow-x-auto rounded-xl" style={{ border: "1px solid var(--c-border)" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              {a.sites.map((s) => (
                <th key={s.url}>
                  <a href={s.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1.5 hover:underline">
                    <Favicon domain={s.domain} size={14} /> {s.domain}
                  </a>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {a.rows.map((r) => (
              <tr key={r.metric}>
                <td style={{ color: "var(--c-muted)" }} className="whitespace-nowrap">{r.metric}</td>
                {r.values.map((v, i) => <td key={i}>{v}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ---------------- Extraction ---------------- */

function ExtractionPanel({ a }: { a: NonNullable<Artifacts["extraction"]> }) {
  const cols = Array.from(new Set(a.rows.flatMap((r) => Object.keys(r)))).slice(0, 8);
  const fmt = (v: unknown) => (typeof v === "object" && v !== null ? JSON.stringify(v) : String(v ?? ""));
  return (
    <div className="glass card fade-in">
      <div className="section-title"><Table2 size={14} /> Extracted data · {a.count} records</div>
      {a.rows.length ? (
        <div className="overflow-auto max-h-96 rounded-xl" style={{ border: "1px solid var(--c-border)" }}>
          <table className="data-table">
            <thead><tr><th>#</th>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>
              {a.rows.slice(0, 100).map((r, i) => (
                <tr key={i}>
                  <td style={{ color: "var(--c-faint)" }}>{i + 1}</td>
                  {cols.map((c) => <td key={c} className="max-w-[320px]">{fmt(r[c])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-sm" style={{ color: "var(--c-muted)" }}>No structured records were found for this request.</div>
      )}
    </div>
  );
}

export default function ArtifactsView({ artifacts, ask }: { artifacts: Artifacts; ask: Ask }) {
  return (
    <div className="flex flex-col gap-3">
      {artifacts.overview && <OverviewPanel a={artifacts.overview} ask={ask} />}
      {artifacts.security && <SecurityPanel a={artifacts.security} />}
      {artifacts.structure && <StructurePanel a={artifacts.structure} />}
      {artifacts.comparison && <ComparisonPanel a={artifacts.comparison} />}
      {artifacts.similar && <SimilarPanel a={artifacts.similar} ask={ask} />}
      {artifacts.extraction && <ExtractionPanel a={artifacts.extraction} />}
      {artifacts.site_documents && artifacts.site_documents.length > 0 && (
        <div className="glass card fade-in"><SiteDocsList docs={artifacts.site_documents} /></div>
      )}
    </div>
  );
}
