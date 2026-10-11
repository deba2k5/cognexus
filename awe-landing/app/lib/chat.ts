/**
 * Deep-research chat client
 * =========================
 * Types and API calls for the /chat endpoint of the AWE backend.
 */

import { API_URL } from './api';

export type Strategy = 'beam' | 'bfs' | 'dfs';

export type Intent =
    | 'ask_url' | 'overview' | 'qa' | 'security' | 'structure'
    | 'similar' | 'compare' | 'extract' | 'report' | 'error';

export interface SiteCard {
    url: string;
    domain: string;
    title: string;
    description: string;
    favicon: string;
}

export interface ChatOption {
    rank: number;
    label: string;
    approach: string;
    answer: string;
    confidence: number;
    rationale: string;
    critique: string;
    evidence: string[];
    data: Record<string, unknown>[];
    node_id: string | null;
    depth: number;
}

export interface ThoughtNode {
    id: string;
    parent_id: string | null;
    depth: number;
    approach: string;
    score: number;
    status: 'generated' | 'expanded' | 'pruned' | 'goal' | 'selected';
    order: number;
    critique: string;
    scores: Record<string, number>;
}

export interface AgentStep {
    agent: string;
    action: string;
    detail: string;
    t_ms: number;
}

export interface Reasoning {
    strategy: Strategy;
    depth?: number;
    steps: AgentStep[];
    tree: ThoughtNode[];
    llm_calls: number;
    llm_errors?: number;
    model?: string;
    duration_ms: number;
}

export interface Finding {
    severity: 'critical' | 'warning' | 'info' | 'pass';
    message: string;
    detail: string;
    check: string;
    fix?: { summary: string; snippet: string };
}

export interface SimilarSite {
    name: string;
    url: string;
    domain: string;
    why: string;
    difference: string;
    similarity: number;
    reachable?: boolean;
    status?: number | null;
    final_url?: string;
    page_title?: string;
}

export interface SiteDocument {
    url: string;
    title: string;
    type: string;
}

export interface Artifacts {
    overview?: {
        tech: string[];
        contacts: { emails: string[]; phones: string[]; socials: Record<string, string> };
        pages: { url: string; title: string; chars: number }[];
        documents: SiteDocument[];
        response_ms: number;
        security: { grade: string; score: number; summary: Record<string, number> };
        structure: { grade: string; score: number; internal_links: number; external_links: number };
        word_count: number;
    };
    security?: {
        score: number;
        grade: string;
        summary: Record<string, number>;
        findings: Finding[];
        passed: Finding[];
        error?: string;
    };
    structure?: {
        score: number;
        grade: string;
        dom: { max_depth: number; total_elements: number; unique_tags: number; complexity: string; top_tags: { tag: string; count: number }[] };
        headings: { headings: { level: number; text: string; tag: string }[]; heading_counts: Record<string, number>; issues: { severity: string; message: string }[] };
        links: { total_links: number; internal_links: number; external_links: number; anchor_links: number; external_domains: { domain: string; count: number }[] };
        metadata: { title: string | null; description: string | null; language: string | null; seo_issues: { severity: string; message: string }[] };
        sitemap: { total_paths: number; max_depth: number; tree: { path: string; depth: number; url: string; children_count: number }[] };
        resources: { scripts: { total: number }; stylesheets: { total: number }; images: { total: number; missing_alt: number; accessibility_score: string }; iframes: number };
        tech: string[];
        pages: { url: string; title: string }[];
    };
    similar?: { category: string; sites: SimilarSite[] };
    comparison?: { sites: SiteCard[]; rows: { metric: string; values: string[] }[] };
    extraction?: { rows: Record<string, unknown>[]; count: number };
    site_documents?: SiteDocument[];
}

export interface GeneratedDocument {
    name: string;
    mime: string;
    content: string;
}

export interface ChatResponse {
    id?: string;
    intent: Intent;
    strategy?: Strategy;
    site: SiteCard | null;
    message: string;
    options: ChatOption[];
    artifacts: Artifacts;
    documents: GeneratedDocument[];
    suggestions: string[];
    guardrail?: { allowed: boolean; reason: string };
    reasoning: Reasoning | null;
}

export interface ChatRequest {
    message: string;
    site_url?: string | null;
    history?: { role: 'user' | 'assistant'; content: string }[];
    strategy?: Strategy;
    intent?: Intent;
    depth?: number;
}

export async function sendChat(req: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
    const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
        signal,
    });
    if (!response.ok) {
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`Research request failed (${response.status}): ${text.slice(0, 200)}`);
    }
    return response.json();
}

export function downloadDocument(doc: GeneratedDocument) {
    const blob = new Blob([doc.content], { type: doc.mime || 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = doc.name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
