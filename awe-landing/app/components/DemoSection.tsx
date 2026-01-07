"use client";

import { useState, useCallback } from "react";
import { aweApi, type DemoResult } from "../lib/api";

interface ExtractedItem {
    [key: string]: unknown;
}

interface ToTInfo {
    best_strategy?: {
        strategy: string;
        reasoning: string;
        score: number;
        items_extracted: number;
    };
    all_strategies?: Array<{
        strategy: string;
        reasoning: string;
        score: number;
        status: string;
    }>;
}

interface LiveFetchState {
    status: "idle" | "loading" | "success" | "error";
    result?: DemoResult & {
        stats: {
            model?: string;
            tokens_used?: number;
            pages_visited: number;
            items_extracted: number;
            duration_ms: number;
            tot_enabled?: boolean;
            thoughts_generated?: number;
            thoughts_tried?: number;
        };
        tot_info?: ToTInfo;
        links?: Array<{ url: string; text: string }>;
    };
    error?: string;
}

export default function DemoSection() {
    const [url, setUrl] = useState("https://quotes.toscrape.com/");
    const [fetchState, setFetchState] = useState<LiveFetchState>({ status: "idle" });
    const [apiStatus, setApiStatus] = useState<"unknown" | "online" | "offline">("unknown");
    const [showStrategies, setShowStrategies] = useState(false);

    const checkApiStatus = useCallback(async () => {
        try {
            await aweApi.healthCheck();
            setApiStatus("online");
        } catch {
            setApiStatus("offline");
        }
    }, []);

    const runLiveFetch = async () => {
        setFetchState({ status: "loading" });
        setShowStrategies(false);

        try {
            const result = await aweApi.runDemo(url);

            if (result.status === "error") {
                setFetchState({
                    status: "error",
                    error: result.message || "Extraction failed",
                });
            } else {
                setFetchState({ status: "success", result: result as LiveFetchState["result"] });
            }
        } catch (error) {
            setFetchState({
                status: "error",
                error: error instanceof Error ? error.message : "Failed to fetch data",
            });
        }
    };

    const renderValue = (value: unknown): string => {
        if (Array.isArray(value)) {
            return value.join(", ");
        }
        if (typeof value === "object" && value !== null) {
            return JSON.stringify(value);
        }
        return String(value);
    };

    const isTotEnabled = fetchState.result?.stats?.tot_enabled;
    const totInfo = fetchState.result?.tot_info;

    return (
        <section id="demo" className="py-32 px-6">
            <div className="max-w-5xl mx-auto">
                <div className="text-center mb-12">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-purple-500/30 bg-purple-500/10 mb-6">
                        <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
                        <span className="text-sm text-purple-300">
                            Tree of Thought + SLM • Multi-Strategy Extraction
                        </span>
                    </div>
                    <h2 className="text-4xl md:text-5xl font-bold mb-6">
                        <span className="gradient-text">Live Fetch</span> with ToT Reasoning
                    </h2>
                    <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                        Uses Tree of Thought reasoning to generate multiple extraction strategies,
                        evaluate each approach, and pick the best one – enabling small models to perform like large ones.
                    </p>
                </div>

                <div className="glass-card p-8">
                    {/* API Status */}
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                            <span className="text-sm text-gray-400">API Status</span>
                            <span className="text-xs px-2 py-1 rounded bg-cyan-500/20 text-cyan-300">
                                ToT + SLM
                            </span>
                        </div>
                        <button
                            onClick={checkApiStatus}
                            className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm transition-all ${apiStatus === "online"
                                    ? "bg-green-500/20 text-green-400"
                                    : apiStatus === "offline"
                                        ? "bg-red-500/20 text-red-400"
                                        : "bg-gray-500/20 text-gray-400 hover:bg-gray-500/30"
                                }`}
                        >
                            <span
                                className={`w-2 h-2 rounded-full ${apiStatus === "online"
                                        ? "bg-green-400 animate-pulse"
                                        : apiStatus === "offline"
                                            ? "bg-red-400"
                                            : "bg-gray-400"
                                    }`}
                            />
                            {apiStatus === "unknown" ? "Check Status" : apiStatus.toUpperCase()}
                        </button>
                    </div>

                    {/* URL Input */}
                    <div className="mb-6">
                        <label htmlFor="fetch-url" className="block text-sm text-gray-400 mb-2">
                            Website URL
                        </label>
                        <input
                            id="fetch-url"
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://example.com"
                            className="w-full px-4 py-3 rounded-xl bg-black/50 border border-purple-500/30 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 transition-colors font-mono text-sm"
                        />
                    </div>

                    {/* Run Button */}
                    <button
                        onClick={runLiveFetch}
                        disabled={fetchState.status === "loading" || !url}
                        className={`w-full btn-primary flex items-center justify-center gap-2 text-lg ${fetchState.status === "loading" ? "opacity-70 cursor-wait" : ""
                            }`}
                    >
                        {fetchState.status === "loading" ? (
                            <>
                                <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                Generating Strategies & Extracting...
                            </>
                        ) : (
                            <>
                                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                                Extract with ToT
                            </>
                        )}
                    </button>

                    {/* Error */}
                    {fetchState.status === "error" && (
                        <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400">
                            <p className="font-semibold mb-1">Extraction Failed</p>
                            <p className="text-sm">{fetchState.error}</p>
                        </div>
                    )}

                    {/* Success */}
                    {fetchState.status === "success" && fetchState.result && (
                        <div className="mt-6 space-y-6">
                            {/* Stats Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                                <div className="text-center p-3 rounded-xl bg-purple-500/10 border border-purple-500/30">
                                    <div className="text-xl font-bold text-purple-400">
                                        {fetchState.result.stats.items_extracted}
                                    </div>
                                    <div className="text-xs text-gray-400">Items</div>
                                </div>
                                <div className="text-center p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/30">
                                    <div className="text-xl font-bold text-cyan-400">
                                        {(fetchState.result.stats.duration_ms / 1000).toFixed(1)}s
                                    </div>
                                    <div className="text-xs text-gray-400">Duration</div>
                                </div>
                                {isTotEnabled && (
                                    <>
                                        <div className="text-center p-3 rounded-xl bg-orange-500/10 border border-orange-500/30">
                                            <div className="text-xl font-bold text-orange-400">
                                                {fetchState.result.stats.thoughts_generated || 0}
                                            </div>
                                            <div className="text-xs text-gray-400">Strategies</div>
                                        </div>
                                        <div className="text-center p-3 rounded-xl bg-pink-500/10 border border-pink-500/30">
                                            <div className="text-xl font-bold text-pink-400">
                                                {fetchState.result.stats.thoughts_tried || 0}
                                            </div>
                                            <div className="text-xs text-gray-400">Tried</div>
                                        </div>
                                    </>
                                )}
                                <div className="text-center p-3 rounded-xl bg-green-500/10 border border-green-500/30">
                                    <div className="text-sm font-bold text-green-400">
                                        {isTotEnabled ? "ToT" : "LIVE"}
                                    </div>
                                    <div className="text-xs text-gray-400">Mode</div>
                                </div>
                            </div>

                            {/* Best Strategy (ToT) */}
                            {isTotEnabled && totInfo?.best_strategy && (
                                <div className="p-4 rounded-xl bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-500/30">
                                    <div className="flex items-center gap-2 mb-2">
                                        <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <h4 className="font-semibold text-white">Best Strategy Selected</h4>
                                        <span className="text-xs px-2 py-0.5 rounded bg-purple-500/30 text-purple-300">
                                            Score: {totInfo.best_strategy.score}
                                        </span>
                                    </div>
                                    <p className="text-sm text-gray-300">{totInfo.best_strategy.strategy}</p>
                                    {totInfo.best_strategy.reasoning && (
                                        <p className="text-xs text-gray-500 mt-1">
                                            Why: {totInfo.best_strategy.reasoning}
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* All Strategies Toggle */}
                            {isTotEnabled && totInfo?.all_strategies && totInfo.all_strategies.length > 1 && (
                                <button
                                    onClick={() => setShowStrategies(!showStrategies)}
                                    className="text-sm text-purple-400 hover:text-purple-300 flex items-center gap-1"
                                >
                                    {showStrategies ? "▼" : "▶"} View all {totInfo.all_strategies.length} strategies
                                </button>
                            )}

                            {showStrategies && totInfo?.all_strategies && (
                                <div className="space-y-2 max-h-48 overflow-y-auto">
                                    {totInfo.all_strategies.map((s, i) => (
                                        <div
                                            key={i}
                                            className={`p-3 rounded-lg text-sm ${s.status === "succeeded"
                                                    ? "bg-green-500/10 border border-green-500/30"
                                                    : s.status === "failed"
                                                        ? "bg-red-500/10 border border-red-500/30"
                                                        : "bg-gray-500/10 border border-gray-500/30"
                                                }`}
                                        >
                                            <div className="flex items-center justify-between">
                                                <span className="text-gray-300">{s.strategy}</span>
                                                <span className="text-xs text-gray-500">
                                                    {s.score} • {s.status}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Extracted Data */}
                            {fetchState.result.data.length > 0 && (
                                <div>
                                    <h4 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                                        <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        Extracted Data ({fetchState.result.data.length} items)
                                    </h4>
                                    <div className="grid gap-3 max-h-80 overflow-y-auto pr-2">
                                        {fetchState.result.data.map((item: ExtractedItem, index: number) => (
                                            <div
                                                key={index}
                                                className="p-3 rounded-xl bg-black/30 border border-purple-500/20 hover:border-purple-500/40 transition-colors"
                                            >
                                                <div className="grid gap-1">
                                                    {Object.entries(item).map(([key, value]) => (
                                                        <div key={key} className="flex gap-2 text-sm">
                                                            <span className="text-purple-400 font-medium min-w-20">{key}:</span>
                                                            <span className="text-gray-300">{renderValue(value)}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Raw JSON */}
                            <details className="group">
                                <summary className="cursor-pointer text-sm text-gray-400 hover:text-white">
                                    View Raw JSON
                                </summary>
                                <div className="mt-3 p-3 rounded-xl bg-black/50 border border-purple-500/20">
                                    <pre className="text-xs overflow-x-auto text-gray-300 max-h-48 overflow-y-auto">
                                        {JSON.stringify(fetchState.result.data, null, 2)}
                                    </pre>
                                </div>
                            </details>
                        </div>
                    )}
                </div>

                {/* Feature Highlights */}
                <div className="mt-12 grid md:grid-cols-3 gap-6 text-center">
                    <div className="p-6 rounded-xl border border-purple-500/10">
                        <div className="text-2xl mb-3">🧠</div>
                        <h4 className="font-semibold text-white mb-2">Tree of Thought</h4>
                        <p className="text-sm text-gray-400">Generates multiple strategies, evaluates each, picks the best.</p>
                    </div>
                    <div className="p-6 rounded-xl border border-purple-500/10">
                        <div className="text-2xl mb-3">🤏</div>
                        <h4 className="font-semibold text-white mb-2">SLM Optimized</h4>
                        <p className="text-sm text-gray-400">Works great with 8B models like llama-3.1-8b-instant.</p>
                    </div>
                    <div className="p-6 rounded-xl border border-purple-500/10">
                        <div className="text-2xl mb-3">⚡</div>
                        <h4 className="font-semibold text-white mb-2">Self-Correcting</h4>
                        <p className="text-sm text-gray-400">Tries multiple approaches if the first one fails.</p>
                    </div>
                </div>
            </div>
        </section>
    );
}
