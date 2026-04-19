"use client";

import { useState, useCallback } from "react";
import { aweApi, type DemoResult, type SecurityResult, type VisualizationResult, type StructureResult } from "../lib/api";

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

type AnalysisTab = "extract" | "security" | "charts" | "structure";

export default function DemoSection() {
    const [url, setUrl] = useState("https://quotes.toscrape.com/");
    const [fetchState, setFetchState] = useState<LiveFetchState>({ status: "idle" });
    const [apiStatus, setApiStatus] = useState<"unknown" | "online" | "offline">("unknown");
    const [showStrategies, setShowStrategies] = useState(false);
    const [activeTab, setActiveTab] = useState<AnalysisTab>("extract");

    const [securityState, setSecurityState] = useState<{ status: string; result?: SecurityResult; error?: string }>({ status: "idle" });
    const [chartState, setChartState] = useState<{ status: string; result?: VisualizationResult; error?: string; chartType: string }>({ status: "idle", chartType: "dashboard" });
    const [structureState, setStructureState] = useState<{ status: string; result?: StructureResult; error?: string }>({ status: "idle" });

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
                setFetchState({ status: "error", error: result.message || "Extraction failed" });
            } else {
                setFetchState({ status: "success", result: result as LiveFetchState["result"] });
            }
        } catch (error) {
            setFetchState({ status: "error", error: error instanceof Error ? error.message : "Failed to fetch data" });
        }
    };

    const runSecurityScan = async () => {
        setSecurityState({ status: "loading" });
        try {
            const result = await aweApi.runSecurityScan(url);
            setSecurityState({ status: "success", result });
        } catch (error) {
            setSecurityState({ status: "error", error: error instanceof Error ? error.message : "Security scan failed" });
        }
    };

    const runVisualization = async (chartType: string = "dashboard") => {
        if (!fetchState.result?.data?.length) {
            setChartState({ status: "error", error: "Run extraction first to generate charts", chartType });
            return;
        }
        setChartState({ status: "loading", chartType });
        try {
            const result = await aweApi.getVisualization(fetchState.result.data, chartType, undefined, undefined, url);
            setChartState({ status: "success", result, chartType });
        } catch (error) {
            setChartState({ status: "error", error: error instanceof Error ? error.message : "Visualization failed", chartType });
        }
    };

    const runStructureAnalysis = async () => {
        setStructureState({ status: "loading" });
        try {
            const result = await aweApi.analyzeStructure(url);
            setStructureState({ status: "success", result });
        } catch (error) {
            setStructureState({ status: "error", error: error instanceof Error ? error.message : "Structure analysis failed" });
        }
    };

    const renderValue = (value: unknown): string => {
        if (Array.isArray(value)) return value.join(", ");
        if (typeof value === "object" && value !== null) return JSON.stringify(value);
        return String(value);
    };

    const severityStyles = (severity: string) => {
        switch (severity) {
            case "critical": return "bg-[#fecaca] border-[#dc2626]";
            case "warning": return "bg-[#fed7aa] border-[#ea580c]";
            case "info": return "bg-[#bfdbfe] border-[#2563eb]";
            case "pass": return "bg-[#bbf7d0] border-[#15803d]";
            default: return "bg-white border-[#1a1a1a]";
        }
    };

    const severityIcon = (severity: string) => {
        switch (severity) {
            case "critical": return "🔴";
            case "warning": return "🟡";
            case "info": return "🔵";
            case "pass": return "🟢";
            default: return "⚪";
        }
    };

    const gradeStyles = (grade: string) => {
        switch (grade) {
            case "A": return "text-[#15803d] bg-[#bbf7d0]";
            case "B": return "text-[#0891b2] bg-[#a5f3fc]";
            case "C": return "text-[#a16207] bg-[#fef08a]";
            case "D": return "text-[#ea580c] bg-[#fed7aa]";
            default: return "text-[#dc2626] bg-[#fecaca]";
        }
    };

    const isTotEnabled = fetchState.result?.stats?.tot_enabled;
    const totInfo = fetchState.result?.tot_info;

    const tabs: { id: AnalysisTab; label: string; icon: string; color: string }[] = [
        { id: "extract", label: "Extract", icon: "⚡", color: "bg-[#c4b5fd]" },
        { id: "security", label: "Security", icon: "🛡️", color: "bg-[#a5f3fc]" },
        { id: "charts", label: "Charts", icon: "📊", color: "bg-[#bbf7d0]" },
        { id: "structure", label: "Structure", icon: "🌐", color: "bg-[#fed7aa]" },
    ];

    const isLoading =
        (activeTab === "extract" && fetchState.status === "loading") ||
        (activeTab === "security" && securityState.status === "loading") ||
        (activeTab === "charts" && chartState.status === "loading") ||
        (activeTab === "structure" && structureState.status === "loading");

    const btnLabels: Record<AnalysisTab, { icon: string; text: string; color: string }> = {
        extract: { icon: "⚡", text: "Extract with ToT", color: "bg-[#8b5cf6]" },
        security: { icon: "🛡️", text: "Run Security Scan", color: "bg-[#06b6d4]" },
        charts: { icon: "📊", text: "Generate Charts", color: "bg-[#22c55e]" },
        structure: { icon: "🌐", text: "Analyze Structure", color: "bg-[#f97316]" },
    };

    return (
        <section id="demo" className="py-32 px-6">
            <div className="max-w-5xl mx-auto">
                <div className="text-center mb-12">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#c4b5fd] border-[3px] border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-6">
                        <span className="w-3 h-3 bg-[#22c55e] border-2 border-[#1a1a1a]" />
                        <span className="text-sm text-[#1a1a1a] font-bold uppercase tracking-wide">
                            CogNexus • Tree of Thought + SLM
                        </span>
                    </div>
                    <h2 className="text-4xl md:text-5xl font-extrabold mb-6 tracking-tight text-[#1a1a1a]">
                        <span className="text-[#8b5cf6]">Live Fetch</span> with ToT Reasoning
                    </h2>
                    <p className="text-xl text-[#666] max-w-2xl mx-auto font-medium">
                        CogNexus uses Tree of Thought reasoning to generate multiple extraction strategies,
                        evaluate each approach, and pick the best one.
                    </p>
                </div>

                <div className="bg-white border-4 border-[#1a1a1a] shadow-[8px_8px_0px_#1a1a1a] p-8">
                    {/* API Status */}
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                            <span className="text-sm text-[#666] font-bold uppercase">API Status</span>
                            <span className="text-xs px-3 py-1.5 bg-[#c4b5fd] border-2 border-[#1a1a1a] font-extrabold uppercase">
                                ToT + SLM
                            </span>
                        </div>
                        <button
                            onClick={checkApiStatus}
                            className={`flex items-center gap-2 px-4 py-2 text-sm font-bold uppercase border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a] transition-all hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[5px_5px_0px_#1a1a1a] active:translate-x-[1px] active:translate-y-[1px] active:shadow-[1px_1px_0px_#1a1a1a] ${apiStatus === "online"
                                ? "bg-[#bbf7d0] text-[#15803d]"
                                : apiStatus === "offline"
                                    ? "bg-[#fecaca] text-[#dc2626]"
                                    : "bg-white text-[#666]"
                                }`}
                        >
                            <span className={`w-3 h-3 border-2 border-[#1a1a1a] ${apiStatus === "online" ? "bg-[#22c55e]" : apiStatus === "offline" ? "bg-[#ef4444]" : "bg-[#999]"}`} />
                            {apiStatus === "unknown" ? "Check Status" : apiStatus.toUpperCase()}
                        </button>
                    </div>

                    {/* URL Input */}
                    <div className="mb-6">
                        <label htmlFor="fetch-url" className="block text-sm text-[#1a1a1a] mb-2 font-bold uppercase tracking-wide">
                            Website URL
                        </label>
                        <input
                            id="fetch-url"
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://example.com"
                            className="w-full px-4 py-3 bg-white border-[3px] border-[#1a1a1a] text-[#1a1a1a] placeholder-[#999] focus:outline-none focus:shadow-[4px_4px_0px_#8b5cf6] transition-shadow font-mono text-sm font-bold"
                        />
                    </div>

                    {/* Tab Navigation */}
                    <div className="flex gap-0 mb-6 border-[3px] border-[#1a1a1a]">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-extrabold uppercase tracking-wide transition-all border-r-[3px] border-[#1a1a1a] last:border-r-0 ${activeTab === tab.id
                                    ? `${tab.color} text-[#1a1a1a]`
                                    : "bg-white text-[#666] hover:bg-[#f5f0e8]"
                                    }`}
                            >
                                <span>{tab.icon}</span>
                                <span className="hidden sm:inline">{tab.label}</span>
                            </button>
                        ))}
                    </div>

                    {/* Run Button */}
                    <button
                        onClick={() => {
                            if (activeTab === "extract") runLiveFetch();
                            else if (activeTab === "security") runSecurityScan();
                            else if (activeTab === "charts") runVisualization(chartState.chartType);
                            else if (activeTab === "structure") runStructureAnalysis();
                        }}
                        disabled={isLoading || !url}
                        className={`w-full ${btnLabels[activeTab].color} text-white font-extrabold text-lg uppercase tracking-wide py-4 border-[3px] border-[#1a1a1a] shadow-[6px_6px_0px_#1a1a1a] flex items-center justify-center gap-3 transition-all ${isLoading
                            ? "opacity-70 cursor-wait"
                            : "hover:translate-x-[-3px] hover:translate-y-[-3px] hover:shadow-[9px_9px_0px_#1a1a1a] active:translate-x-[2px] active:translate-y-[2px] active:shadow-[2px_2px_0px_#1a1a1a]"
                            }`}
                    >
                        {isLoading ? (
                            <>
                                <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                Processing...
                            </>
                        ) : (
                            <>
                                <span className="text-xl">{btnLabels[activeTab].icon}</span>
                                {btnLabels[activeTab].text}
                            </>
                        )}
                    </button>

                    {/* ==================== EXTRACT TAB ==================== */}
                    {activeTab === "extract" && (
                        <>
                            {fetchState.status === "error" && (
                                <div className="mt-6 p-4 bg-[#fecaca] border-[3px] border-[#dc2626] text-[#991b1b] font-bold">
                                    <p className="font-extrabold mb-1">⚠ Extraction Failed</p>
                                    <p className="text-sm">{fetchState.error}</p>
                                </div>
                            )}

                            {fetchState.status === "success" && fetchState.result && (
                                <div className="mt-6 space-y-6">
                                    {/* Stats Grid */}
                                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                                        <div className="text-center p-3 bg-[#c4b5fd] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#1a1a1a]">{fetchState.result.stats.items_extracted}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Items</div>
                                        </div>
                                        <div className="text-center p-3 bg-[#a5f3fc] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#1a1a1a]">{(fetchState.result.stats.duration_ms / 1000).toFixed(1)}s</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Duration</div>
                                        </div>
                                        {isTotEnabled && (
                                            <>
                                                <div className="text-center p-3 bg-[#fed7aa] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                                    <div className="text-xl font-extrabold text-[#1a1a1a]">{fetchState.result.stats.thoughts_generated || 0}</div>
                                                    <div className="text-xs text-[#444] font-bold uppercase">Strategies</div>
                                                </div>
                                                <div className="text-center p-3 bg-[#fbcfe8] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                                    <div className="text-xl font-extrabold text-[#1a1a1a]">{fetchState.result.stats.thoughts_tried || 0}</div>
                                                    <div className="text-xs text-[#444] font-bold uppercase">Tried</div>
                                                </div>
                                            </>
                                        )}
                                        <div className="text-center p-3 bg-[#bbf7d0] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-sm font-extrabold text-[#1a1a1a]">{isTotEnabled ? "ToT" : "LIVE"}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Mode</div>
                                        </div>
                                    </div>

                                    {/* Best Strategy */}
                                    {isTotEnabled && totInfo?.best_strategy && (
                                        <div className="p-4 bg-[#c4b5fd] border-[3px] border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a]">
                                            <div className="flex items-center gap-2 mb-2">
                                                <span className="text-lg">🏆</span>
                                                <h4 className="font-extrabold text-[#1a1a1a]">Best Strategy Selected</h4>
                                                <span className="text-xs px-2 py-0.5 bg-white border-2 border-[#1a1a1a] font-extrabold">
                                                    Score: {totInfo.best_strategy.score}
                                                </span>
                                            </div>
                                            <p className="text-sm text-[#1a1a1a] font-bold">{totInfo.best_strategy.strategy}</p>
                                            {totInfo.best_strategy.reasoning && (
                                                <p className="text-xs text-[#444] mt-1 font-medium">Why: {totInfo.best_strategy.reasoning}</p>
                                            )}
                                        </div>
                                    )}

                                    {/* All Strategies Toggle */}
                                    {isTotEnabled && totInfo?.all_strategies && totInfo.all_strategies.length > 1 && (
                                        <button
                                            onClick={() => setShowStrategies(!showStrategies)}
                                            className="text-sm text-[#8b5cf6] hover:text-[#6d28d9] flex items-center gap-1 font-extrabold uppercase"
                                        >
                                            {showStrategies ? "▼" : "▶"} View all {totInfo.all_strategies.length} strategies
                                        </button>
                                    )}

                                    {showStrategies && totInfo?.all_strategies && (
                                        <div className="space-y-2 max-h-48 overflow-y-auto">
                                            {totInfo.all_strategies.map((s, i) => (
                                                <div key={i} className={`p-3 text-sm border-[3px] border-[#1a1a1a] font-bold ${s.status === "succeeded" ? "bg-[#bbf7d0]" : s.status === "failed" ? "bg-[#fecaca]" : "bg-white"}`}>
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-[#1a1a1a]">{s.strategy}</span>
                                                        <span className="text-xs text-[#666]">{s.score} • {s.status}</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {/* Extracted Data */}
                                    {fetchState.result.data.length > 0 && (
                                        <div>
                                            <h4 className="text-lg font-extrabold text-[#1a1a1a] mb-3 flex items-center gap-2 uppercase">
                                                ✅ Extracted Data ({fetchState.result.data.length} items)
                                            </h4>
                                            <div className="grid gap-3 max-h-80 overflow-y-auto pr-2">
                                                {fetchState.result.data.map((item: ExtractedItem, index: number) => (
                                                    <div key={index} className="p-3 bg-[#f5f0e8] border-[3px] border-[#1a1a1a] hover:shadow-[4px_4px_0px_#1a1a1a] transition-shadow">
                                                        <div className="grid gap-1">
                                                            {Object.entries(item).map(([key, value]) => (
                                                                <div key={key} className="flex gap-2 text-sm">
                                                                    <span className="text-[#8b5cf6] font-extrabold min-w-20 uppercase text-xs">{key}:</span>
                                                                    <span className="text-[#444] font-medium">{renderValue(value)}</span>
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
                                        <summary className="cursor-pointer text-sm text-[#666] hover:text-[#1a1a1a] font-extrabold uppercase">
                                            View Raw JSON
                                        </summary>
                                        <div className="mt-3 p-3 bg-[#1a1a1a] border-[3px] border-[#1a1a1a]">
                                            <pre className="text-xs overflow-x-auto text-[#4ade80] max-h-48 overflow-y-auto font-mono">
                                                {JSON.stringify(fetchState.result.data, null, 2)}
                                            </pre>
                                        </div>
                                    </details>
                                </div>
                            )}
                        </>
                    )}

                    {/* ==================== SECURITY TAB ==================== */}
                    {activeTab === "security" && (
                        <div className="mt-6">
                            {securityState.status === "error" && (
                                <div className="p-4 bg-[#fecaca] border-[3px] border-[#dc2626] text-[#991b1b] font-bold">
                                    <p className="font-extrabold mb-1">⚠ Scan Failed</p>
                                    <p className="text-sm">{securityState.error}</p>
                                </div>
                            )}

                            {securityState.status === "success" && securityState.result && (
                                <div className="space-y-6">
                                    {/* Score */}
                                    <div className="flex items-center gap-6">
                                        <div className={`w-20 h-20 flex items-center justify-center text-3xl font-extrabold border-4 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] ${gradeStyles(securityState.result.grade)}`}>
                                            {securityState.result.grade}
                                        </div>
                                        <div>
                                            <div className="text-2xl font-extrabold text-[#1a1a1a]">Security Score: {securityState.result.score}/100</div>
                                            <div className="text-sm text-[#666] font-bold">Scanned in {securityState.result.duration_seconds}s</div>
                                        </div>
                                    </div>

                                    {/* Summary */}
                                    <div className="grid grid-cols-4 gap-3">
                                        <div className="text-center p-3 bg-[#fecaca] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#dc2626]">{securityState.result.summary.critical}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Critical</div>
                                        </div>
                                        <div className="text-center p-3 bg-[#fed7aa] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#ea580c]">{securityState.result.summary.warning}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Warning</div>
                                        </div>
                                        <div className="text-center p-3 bg-[#bfdbfe] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#2563eb]">{securityState.result.summary.info}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Info</div>
                                        </div>
                                        <div className="text-center p-3 bg-[#bbf7d0] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#15803d]">{securityState.result.summary.pass}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Passed</div>
                                        </div>
                                    </div>

                                    {/* Checks */}
                                    <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
                                        {securityState.result.checks.map((check, ci) => (
                                            <div key={ci}>
                                                <h4 className="text-sm font-extrabold text-[#1a1a1a] mb-2 flex items-center gap-2 uppercase tracking-wide">
                                                    <span className="w-3 h-3 bg-[#8b5cf6] border-2 border-[#1a1a1a]" />
                                                    {check.check}
                                                </h4>
                                                <div className="space-y-2">
                                                    {check.items.map((item, ii) => (
                                                        <div key={ii} className={`p-3 border-[3px] text-sm font-bold ${severityStyles(item.severity)}`}>
                                                            <div className="flex items-start gap-2">
                                                                <span>{severityIcon(item.severity)}</span>
                                                                <div>
                                                                    <div className="font-extrabold text-[#1a1a1a]">{item.message}</div>
                                                                    <div className="text-xs text-[#666] mt-0.5 font-medium">{item.detail}</div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {securityState.status === "idle" && (
                                <div className="text-center py-12 text-[#666]">
                                    <div className="text-5xl mb-3">🛡️</div>
                                    <p className="font-bold">Click &quot;Run Security Scan&quot; to check for vulnerabilities,<br />broken links, SSL issues, and more.</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* ==================== CHARTS TAB ==================== */}
                    {activeTab === "charts" && (
                        <div className="mt-6">
                            <div className="flex gap-0 mb-4 border-[3px] border-[#1a1a1a]">
                                {["dashboard", "bar", "pie", "completeness", "words"].map((type) => (
                                    <button
                                        key={type}
                                        onClick={() => {
                                            setChartState(prev => ({ ...prev, chartType: type }));
                                            if (fetchState.result?.data?.length) runVisualization(type);
                                        }}
                                        className={`flex-1 px-3 py-2 text-xs font-extrabold uppercase transition-all border-r-[3px] border-[#1a1a1a] last:border-r-0 ${chartState.chartType === type
                                            ? "bg-[#bbf7d0] text-[#1a1a1a]"
                                            : "bg-white text-[#666] hover:bg-[#f5f0e8]"
                                            }`}
                                    >
                                        {type}
                                    </button>
                                ))}
                            </div>

                            {chartState.status === "error" && (
                                <div className="p-4 bg-[#fecaca] border-[3px] border-[#dc2626] text-[#991b1b] font-bold">
                                    <p className="font-extrabold mb-1">⚠ Chart Generation Failed</p>
                                    <p className="text-sm">{chartState.error}</p>
                                </div>
                            )}

                            {chartState.status === "success" && chartState.result?.image_base64 && (
                                <div className="space-y-4">
                                    <div className="overflow-hidden border-[3px] border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] bg-white">
                                        <img
                                            src={`data:image/png;base64,${chartState.result.image_base64}`}
                                            alt={`${chartState.chartType} chart`}
                                            className="w-full h-auto"
                                        />
                                    </div>
                                    <div className="text-xs text-[#666] text-center font-bold uppercase">
                                        {chartState.result.items_analyzed} items analyzed • {chartState.chartType} chart
                                    </div>
                                </div>
                            )}

                            {chartState.status === "idle" && (
                                <div className="text-center py-12 text-[#666]">
                                    <div className="text-5xl mb-3">📊</div>
                                    <p className="font-bold">Extract data first, then generate visual charts<br />from the extracted data using Matplotlib.</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* ==================== STRUCTURE TAB ==================== */}
                    {activeTab === "structure" && (
                        <div className="mt-6">
                            {structureState.status === "error" && (
                                <div className="p-4 bg-[#fecaca] border-[3px] border-[#dc2626] text-[#991b1b] font-bold">
                                    <p className="font-extrabold mb-1">⚠ Analysis Failed</p>
                                    <p className="text-sm">{structureState.error}</p>
                                </div>
                            )}

                            {structureState.status === "success" && structureState.result && (
                                <div className="space-y-6">
                                    {/* Score */}
                                    <div className="flex items-center gap-6">
                                        <div className={`w-20 h-20 flex items-center justify-center text-3xl font-extrabold border-4 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] ${gradeStyles(structureState.result.grade)}`}>
                                            {structureState.result.grade}
                                        </div>
                                        <div>
                                            <div className="text-2xl font-extrabold text-[#1a1a1a]">Structure Score: {structureState.result.score}/100</div>
                                            <div className="text-sm text-[#666] font-bold">Analyzed in {structureState.result.duration_seconds}s</div>
                                        </div>
                                    </div>

                                    {/* DOM Stats */}
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        <div className="text-center p-3 bg-[#c4b5fd] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#1a1a1a]">{structureState.result.dom.total_elements}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Elements</div>
                                        </div>
                                        <div className="text-center p-3 bg-[#a5f3fc] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#1a1a1a]">{structureState.result.dom.max_depth}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Max Depth</div>
                                        </div>
                                        <div className="text-center p-3 bg-[#fed7aa] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#1a1a1a]">{structureState.result.links.total_links}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Links</div>
                                        </div>
                                        <div className="text-center p-3 bg-[#fbcfe8] border-[3px] border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a]">
                                            <div className="text-xl font-extrabold text-[#1a1a1a]">{structureState.result.headings.total_headings}</div>
                                            <div className="text-xs text-[#444] font-bold uppercase">Headings</div>
                                        </div>
                                    </div>

                                    {/* Link Distribution */}
                                    <div className="p-4 bg-[#f5f0e8] border-[3px] border-[#1a1a1a]">
                                        <h4 className="text-sm font-extrabold text-[#1a1a1a] mb-3 uppercase">Link Distribution</h4>
                                        <div className="flex gap-3">
                                            <div className="flex-1">
                                                <div className="flex justify-between text-xs text-[#666] mb-1 font-bold uppercase">
                                                    <span>Internal</span>
                                                    <span>{structureState.result.links.internal_links}</span>
                                                </div>
                                                <div className="h-4 bg-white border-2 border-[#1a1a1a]">
                                                    <div className="h-full bg-[#22c55e]" style={{
                                                        width: `${Math.min(100, (structureState.result.links.internal_links / Math.max(structureState.result.links.total_links, 1)) * 100)}%`
                                                    }} />
                                                </div>
                                            </div>
                                            <div className="flex-1">
                                                <div className="flex justify-between text-xs text-[#666] mb-1 font-bold uppercase">
                                                    <span>External</span>
                                                    <span>{structureState.result.links.external_links}</span>
                                                </div>
                                                <div className="h-4 bg-white border-2 border-[#1a1a1a]">
                                                    <div className="h-full bg-[#f97316]" style={{
                                                        width: `${Math.min(100, (structureState.result.links.external_links / Math.max(structureState.result.links.total_links, 1)) * 100)}%`
                                                    }} />
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Heading Hierarchy */}
                                    <div className="p-4 bg-[#f5f0e8] border-[3px] border-[#1a1a1a]">
                                        <h4 className="text-sm font-extrabold text-[#1a1a1a] mb-3 uppercase">Heading Hierarchy</h4>
                                        <div className="space-y-1 max-h-40 overflow-y-auto">
                                            {structureState.result.headings.headings.map((h, i) => (
                                                <div key={i} className="flex items-center gap-2 text-sm" style={{ paddingLeft: `${(h.level - 1) * 16}px` }}>
                                                    <span className="text-[#8b5cf6] font-mono text-xs font-extrabold bg-[#c4b5fd] px-1 border border-[#1a1a1a]">{h.tag}</span>
                                                    <span className="text-[#1a1a1a] font-medium truncate">{h.text}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* SEO Issues */}
                                    {structureState.result.metadata.seo_issues.length > 0 && (
                                        <div className="p-4 bg-[#f5f0e8] border-[3px] border-[#1a1a1a]">
                                            <h4 className="text-sm font-extrabold text-[#1a1a1a] mb-3 uppercase">SEO & Metadata Issues</h4>
                                            <div className="space-y-2">
                                                {structureState.result.metadata.seo_issues.map((issue, i) => (
                                                    <div key={i} className={`p-2 border-[3px] text-sm font-bold ${severityStyles(issue.severity)}`}>
                                                        <span>{severityIcon(issue.severity)}</span> {issue.message}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Resources */}
                                    <div className="p-4 bg-[#f5f0e8] border-[3px] border-[#1a1a1a]">
                                        <h4 className="text-sm font-extrabold text-[#1a1a1a] mb-3 uppercase">Page Resources</h4>
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center text-sm">
                                            <div className="bg-white border-2 border-[#1a1a1a] p-2">
                                                <div className="text-[#0891b2] font-extrabold text-lg">{structureState.result.resources.scripts.total}</div>
                                                <div className="text-xs text-[#666] font-bold uppercase">Scripts</div>
                                            </div>
                                            <div className="bg-white border-2 border-[#1a1a1a] p-2">
                                                <div className="text-[#8b5cf6] font-extrabold text-lg">{structureState.result.resources.stylesheets.total}</div>
                                                <div className="text-xs text-[#666] font-bold uppercase">Styles</div>
                                            </div>
                                            <div className="bg-white border-2 border-[#1a1a1a] p-2">
                                                <div className="text-[#be185d] font-extrabold text-lg">{structureState.result.resources.images.total}</div>
                                                <div className="text-xs text-[#666] font-bold uppercase">Images</div>
                                            </div>
                                            <div className="bg-white border-2 border-[#1a1a1a] p-2">
                                                <div className="text-[#15803d] font-extrabold text-lg">{structureState.result.resources.images.accessibility_score}</div>
                                                <div className="text-xs text-[#666] font-bold uppercase">Alt %</div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Sitemap */}
                                    {structureState.result.sitemap.tree.length > 0 && (
                                        <details className="group">
                                            <summary className="cursor-pointer text-sm text-[#666] hover:text-[#1a1a1a] font-extrabold uppercase">
                                                View Sitemap Tree ({structureState.result.sitemap.total_paths} paths)
                                            </summary>
                                            <div className="mt-3 p-3 bg-[#1a1a1a] border-[3px] border-[#1a1a1a] max-h-48 overflow-y-auto">
                                                {structureState.result.sitemap.tree.map((node, i) => (
                                                    <div key={i} className="text-xs font-mono text-[#4ade80] font-bold" style={{ paddingLeft: `${node.depth * 16}px` }}>
                                                        {node.depth > 0 ? "├── " : ""}{node.path || "/"}
                                                        {node.children_count > 0 && <span className="text-[#c4b5fd]"> ({node.children_count})</span>}
                                                    </div>
                                                ))}
                                            </div>
                                        </details>
                                    )}
                                </div>
                            )}

                            {structureState.status === "idle" && (
                                <div className="text-center py-12 text-[#666]">
                                    <div className="text-5xl mb-3">🌐</div>
                                    <p className="font-bold">Click &quot;Analyze Structure&quot; to examine DOM depth,<br />heading hierarchy, SEO metadata, and link graph.</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Feature Highlights */}
                <div className="mt-12 grid md:grid-cols-3 gap-6 text-center">
                    <div className="p-6 bg-white border-[3px] border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_#1a1a1a] transition-all">
                        <div className="text-3xl mb-3">🧠</div>
                        <h4 className="font-extrabold text-[#1a1a1a] mb-2 uppercase">Tree of Thought</h4>
                        <p className="text-sm text-[#666] font-medium">Generates multiple strategies, evaluates each, picks the best.</p>
                    </div>
                    <div className="p-6 bg-white border-[3px] border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_#1a1a1a] transition-all">
                        <div className="text-3xl mb-3">🛡️</div>
                        <h4 className="font-extrabold text-[#1a1a1a] mb-2 uppercase">Security Scanner</h4>
                        <p className="text-sm text-[#666] font-medium">Checks SSL, headers, broken links, mixed content, and exposed secrets.</p>
                    </div>
                    <div className="p-6 bg-white border-[3px] border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_#1a1a1a] transition-all">
                        <div className="text-3xl mb-3">📊</div>
                        <h4 className="font-extrabold text-[#1a1a1a] mb-2 uppercase">Data Visualization</h4>
                        <p className="text-sm text-[#666] font-medium">Generates charts from extracted data using Matplotlib.</p>
                    </div>
                </div>
            </div>
        </section>
    );
}
