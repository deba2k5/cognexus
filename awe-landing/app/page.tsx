"use client";

import { useState, useEffect } from "react";
import DemoSection from "./components/DemoSection";
import Image from "next/image";

// Icons as SVG components
const BrainIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-8 h-8">
    <path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3 2.5 2.5 0 0 0-1.32 4.24 3 3 0 0 0 .34 5.58 2.5 2.5 0 0 0 2.96 3.08A2.5 2.5 0 0 0 12 19.5a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 12 4.5" />
    <path d="m15.7 10.4-.9.4" />
    <path d="m9.2 13.2-.9.4" />
    <path d="m13.6 15.47-.6-.6" />
    <path d="m10.8 9.13-.6-.6" />
    <path d="m15.7 13.6-.9-.4" />
    <path d="m9.2 10.8-.9-.4" />
    <path d="M12 12v.01" />
  </svg>
);

const EyeIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-8 h-8">
    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const RefreshIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-8 h-8">
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
    <path d="M8 16H3v5" />
  </svg>
);

const BookIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-8 h-8">
    <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    <path d="M8 7h6" />
    <path d="M8 11h8" />
  </svg>
);

const NetworkIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-8 h-8">
    <circle cx="18" cy="5" r="3" />
    <circle cx="6" cy="12" r="3" />
    <circle cx="18" cy="19" r="3" />
    <path d="m8.59 13.51 6.83 3.98" />
    <path d="m8.59 10.49 6.83-3.98" />
  </svg>
);

const SparklesIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-8 h-8">
    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    <path d="M5 3v4" />
    <path d="M19 17v4" />
    <path d="M3 5h4" />
    <path d="M17 19h4" />
  </svg>
);

const ArrowRightIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-5 h-5">
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);

const GithubIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
  </svg>
);

const ChevronDownIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" className="w-6 h-6">
    <path d="m6 9 6 6 6-6" />
  </svg>
);

export default function Home() {
  const [scrolled, setScrolled] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const features = [
    {
      icon: <BrainIcon />,
      title: "Tree of Thought Reasoning",
      description: "Multi-path exploration enables smaller models to perform complex reasoning through systematic evaluation and backtracking.",
      tag: "Core Engine",
      color: "bg-[#c4b5fd]",
      iconColor: "text-[#6d28d9]",
    },
    {
      icon: <EyeIcon />,
      title: "Vision-First Approach",
      description: "Combines screenshot analysis with DOM understanding for superior page comprehension, not just HTML parsing.",
      tag: "Perception",
      color: "bg-[#a5f3fc]",
      iconColor: "text-[#0891b2]",
    },
    {
      icon: <RefreshIcon />,
      title: "Self-Correcting",
      description: "Observes failures and adapts strategies in real-time. 90%+ self-recovery rate on complex tasks.",
      tag: "Resilience",
      color: "bg-[#bbf7d0]",
      iconColor: "text-[#15803d]",
    },
    {
      icon: <BookIcon />,
      title: "Template Learning",
      description: "Automatically generates reusable Playwright extraction patterns from successful explorations.",
      tag: "Learning",
      color: "bg-[#fed7aa]",
      iconColor: "text-[#c2410c]",
    },
    {
      icon: <NetworkIcon />,
      title: "Knowledge Persistence",
      description: "Builds a persistent knowledge graph of learned approaches for zero-shot generalization.",
      tag: "Memory",
      color: "bg-[#fbcfe8]",
      iconColor: "text-[#be185d]",
    },
    {
      icon: <SparklesIcon />,
      title: "Discovery-Driven",
      description: "No hardcoded selectors or site-specific logic. Pure autonomous exploration and discovery.",
      tag: "Autonomy",
      color: "bg-[#fef08a]",
      iconColor: "text-[#a16207]",
    },
  ];

  const stats = [
    { value: "100%", label: "Profile Discovery", subtext: "vs 5% hardcoded", color: "bg-[#c4b5fd]" },
    { value: "95%+", label: "Extraction Accuracy", subtext: "Production-grade", color: "bg-[#a5f3fc]" },
    { value: "90%+", label: "Self-Recovery Rate", subtext: "Auto-healing", color: "bg-[#bbf7d0]" },
    { value: "85%+", label: "Zero-Shot Success", subtext: "New sites", color: "bg-[#fed7aa]" },
  ];

  return (
    <>
      {/* Background Effects */}
      <div className="bg-mesh" />
      <div className="bg-grid" />

      {/* Navigation */}
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled
          ? "bg-white/95 py-3 border-b-4 border-[#1a1a1a]"
          : "bg-transparent py-5"
          }`}
      >
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Image
              src="/cognexus-logo.png"
              alt="CogNexus Logo"
              width={44}
              height={44}
              className="logo-glow"
            />
            <div className="flex flex-col">
              <span className="text-xl font-extrabold tracking-tight text-[#1a1a1a]">CogNexus</span>
              <span className="text-[10px] text-[#666] tracking-widest uppercase -mt-1 font-bold">Web Extractor</span>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="nav-link">Features</a>
            <a href="#architecture" className="nav-link">Architecture</a>
            <a href="#performance" className="nav-link">Performance</a>
            <a href="#demo" className="nav-link">Live Fetch</a>
            <a href="#get-started" className="nav-link">Get Started</a>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/deba2k5/cognexus"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors"
            >
              <GithubIcon />
            </a>
            <a href="#demo" className="btn-primary hidden sm:flex items-center gap-2">
              Try Demo
              <ArrowRightIcon />
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="min-h-screen flex flex-col items-center justify-center px-6 pt-20 relative">
        <div className={`max-w-5xl mx-auto text-center hero-entrance ${mounted ? 'hero-visible' : ''}`}>
          {/* Logo Large */}
          <div className="mb-8 flex justify-center">
            <div className="hero-logo-container">
              <div className="inline-block bg-[#c4b5fd] border-4 border-[#1a1a1a] p-4 shadow-[8px_8px_0px_#1a1a1a]">
                <Image
                  src="/cognexus-logo.png"
                  alt="CogNexus Web Extractor"
                  width={120}
                  height={120}
                  className="float"
                  priority
                />
              </div>
            </div>
          </div>

          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#fef08a] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-8" style={{ border: '3px solid #1a1a1a' }}>
            <span className="w-3 h-3 bg-[#22c55e] border-2 border-[#1a1a1a]" />
            <span className="text-sm text-[#1a1a1a] font-bold uppercase tracking-wide">Powered by SLMs with ToT Reasoning</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-extrabold leading-[1.05] mb-6 tracking-tight text-[#1a1a1a]">
            <span className="block">Autonomous Web</span>
            <span className="text-[#8b5cf6]" style={{ textShadow: '4px 4px 0px #c4b5fd' }}>Exploration at Scale</span>
          </h1>

          {/* Subheadline */}
          <p className="text-lg md:text-xl text-[#444] max-w-3xl mx-auto mb-12 leading-relaxed font-medium">
            A production-grade multi-agent framework for autonomous data extraction.
            Works with <span className="bg-[#a5f3fc] px-1 border-b-3 border-[#0891b2] font-bold">small language models</span> through
            Tree of Thought reasoning — powered by <span className="text-[#8b5cf6] font-extrabold">CogNexus</span>.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <a href="#demo" className="btn-primary flex items-center gap-2 text-lg group">
              Try Live Demo
              <span className="group-hover:translate-x-1 transition-transform"><ArrowRightIcon /></span>
            </a>
            <a href="https://github.com/deba2k5/cognexus" target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center gap-2 text-lg">
              <GithubIcon />
              View on GitHub
            </a>
          </div>

          {/* Code Preview */}
          <div className="glass-card overflow-hidden max-w-3xl mx-auto text-left code-preview-card">
            <div className="flex items-center gap-2 px-4 py-3 border-b-3 border-[#333] bg-[#252525]" style={{ borderBottom: '3px solid #333' }}>
              <div className="w-4 h-4 bg-[#f87171] border-2 border-[#1a1a1a]" />
              <div className="w-4 h-4 bg-[#facc15] border-2 border-[#1a1a1a]" />
              <div className="w-4 h-4 bg-[#4ade80] border-2 border-[#1a1a1a]" />
              <span className="ml-4 text-xs text-gray-400 font-mono font-bold uppercase">quick_start.py</span>
            </div>
            <pre className="p-6 text-sm overflow-x-auto leading-relaxed bg-[#1a1a1a]">
              <code>
                <span className="text-[#c4b5fd]">from</span>{" "}
                <span className="text-[#67e8f9]">cognexus</span>{" "}
                <span className="text-[#c4b5fd]">import</span>{" "}
                <span className="text-white font-bold">WebExtractor</span>
                {"\n\n"}
                <span className="text-gray-500"># Define your goal</span>
                {"\n"}
                <span className="text-white">extractor</span> ={" "}
                <span className="text-[#67e8f9]">WebExtractor</span>
                <span className="text-white">(</span>
                {"\n"}
                {"    "}
                <span className="text-[#fb923c]">model</span>=
                <span className="text-[#4ade80]">&quot;llama-3.1-8b-instant&quot;</span>,
                {"\n"}
                {"    "}
                <span className="text-[#fb923c]">tot_enabled</span>=
                <span className="text-[#c4b5fd]">True</span>
                {"\n"}
                <span className="text-white">)</span>
                {"\n\n"}
                <span className="text-gray-500"># Extract data autonomously</span>
                {"\n"}
                <span className="text-white">results</span> ={" "}
                <span className="text-[#c4b5fd]">await</span>{" "}
                <span className="text-white">extractor.explore(</span>
                {"\n"}
                {"    "}
                <span className="text-[#fb923c]">url</span>=
                <span className="text-[#4ade80]">&quot;https://example.edu/faculty/&quot;</span>,
                {"\n"}
                {"    "}
                <span className="text-[#fb923c]">objective</span>=
                <span className="text-[#4ade80]">&quot;Extract all faculty profiles&quot;</span>
                {"\n"}
                <span className="text-white">)</span>
              </code>
            </pre>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 scroll-indicator">
          <a href="#features" className="flex flex-col items-center text-[#666] hover:text-[#1a1a1a] transition-colors font-bold">
            <span className="text-sm mb-2 uppercase tracking-widest">Explore</span>
            <ChevronDownIcon />
          </a>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-32 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#c4b5fd] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-6" style={{ border: '3px solid #1a1a1a' }}>
              <span className="text-sm text-[#1a1a1a] font-bold uppercase">✦ Capabilities</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-extrabold mb-6 tracking-tight text-[#1a1a1a]">
              Built for <span className="text-[#8b5cf6]">Real-World</span> Extraction
            </h2>
            <p className="text-xl text-[#666] max-w-2xl mx-auto font-medium">
              Everything you need to build production-grade autonomous web scrapers that actually work.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="glass-card feature-card p-8 cursor-pointer group"
              >
                <div className="flex items-center gap-3 mb-5">
                  <div className={`feature-icon w-14 h-14 flex items-center justify-center ${feature.color} border-3 border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a] ${feature.iconColor}`} style={{ border: '3px solid #1a1a1a', boxShadow: '3px 3px 0px #1a1a1a' }}>
                    {feature.icon}
                  </div>
                  <span className="text-[10px] font-extrabold uppercase tracking-widest text-[#1a1a1a] px-3 py-1.5 bg-[#fef08a] border-2 border-[#1a1a1a]">
                    {feature.tag}
                  </span>
                </div>
                <h3 className="text-xl font-extrabold mb-3 text-[#1a1a1a]">{feature.title}</h3>
                <p className="text-[#666] leading-relaxed text-sm font-medium">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Architecture Section */}
      <section id="architecture" className="py-32 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#a5f3fc] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-6" style={{ border: '3px solid #1a1a1a' }}>
              <span className="text-sm text-[#1a1a1a] font-bold uppercase">⚙ System Design</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-extrabold mb-6 tracking-tight text-[#1a1a1a]">
              Multi-Agent <span className="text-[#8b5cf6]">Architecture</span>
            </h2>
            <p className="text-xl text-[#666] max-w-2xl mx-auto font-medium">
              Six specialized agents working in harmony through a state machine orchestrator.
            </p>
          </div>

          {/* Architecture Diagram */}
          <div className="glass-card p-8 md:p-12">
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <div className="architecture-box text-center group">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">🔭</div>
                <h4 className="font-extrabold text-[#1a1a1a]">Observer</h4>
                <p className="text-sm text-[#666] font-medium">Vision + DOM Analysis</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">🧠</div>
                <h4 className="font-extrabold text-[#1a1a1a]">Planner</h4>
                <p className="text-sm text-[#666] font-medium">ToT Strategy</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">⚡</div>
                <h4 className="font-extrabold text-[#1a1a1a]">Executor</h4>
                <p className="text-sm text-[#666] font-medium">Playwright Actions</p>
              </div>
            </div>
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <div className="architecture-box text-center group">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">✅</div>
                <h4 className="font-extrabold text-[#1a1a1a]">Validator</h4>
                <p className="text-sm text-[#666] font-medium">Quality Assurance</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">📊</div>
                <h4 className="font-extrabold text-[#1a1a1a]">Extractor</h4>
                <p className="text-sm text-[#666] font-medium">Data Extraction</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">📚</div>
                <h4 className="font-extrabold text-[#1a1a1a]">Learner</h4>
                <p className="text-sm text-[#666] font-medium">Template Generation</p>
              </div>
            </div>

            {/* State Machine */}
            <div className="border-t-3 border-[#1a1a1a] pt-8" style={{ borderTop: '3px solid #1a1a1a' }}>
              <h4 className="text-center text-[#666] mb-6 text-sm uppercase tracking-widest font-bold">State Machine Flow</h4>
              <div className="flex flex-wrap items-center justify-center gap-2 md:gap-3 text-sm">
                <span className="state-pill bg-[#c4b5fd] text-[#1a1a1a]">Observe</span>
                <span className="text-[#1a1a1a] font-extrabold text-xl">→</span>
                <span className="state-pill bg-[#a5f3fc] text-[#1a1a1a]">Think</span>
                <span className="text-[#1a1a1a] font-extrabold text-xl">→</span>
                <span className="state-pill bg-[#93c5fd] text-[#1a1a1a]">Plan</span>
                <span className="text-[#1a1a1a] font-extrabold text-xl">→</span>
                <span className="state-pill bg-[#fed7aa] text-[#1a1a1a]">Act</span>
                <span className="text-[#1a1a1a] font-extrabold text-xl">→</span>
                <span className="state-pill bg-[#bbf7d0] text-[#1a1a1a]">Validate</span>
                <span className="text-[#1a1a1a] font-extrabold text-xl">→</span>
                <span className="state-pill bg-[#fef08a] text-[#1a1a1a]">Learn</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Performance Section */}
      <section id="performance" className="py-32 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#bbf7d0] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-6" style={{ border: '3px solid #1a1a1a' }}>
              <span className="text-sm text-[#1a1a1a] font-bold uppercase">📈 Benchmarks</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-extrabold mb-6 tracking-tight text-[#1a1a1a]">
              Proven <span className="text-[#8b5cf6]">Performance</span>
            </h2>
            <p className="text-xl text-[#666] max-w-2xl mx-auto font-medium">
              Benchmarked with Llama 3.1 8B on university faculty extraction tasks.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <div
                key={index}
                className={`glass-card p-8 text-center feature-card group ${stat.color}`}
              >
                <div className="stat-number mb-2 group-hover:scale-110 transition-transform inline-block">{stat.value}</div>
                <h4 className="text-lg font-extrabold mb-1 text-[#1a1a1a]">{stat.label}</h4>
                <p className="text-sm text-[#666] font-bold">{stat.subtext}</p>
              </div>
            ))}
          </div>

          {/* Comparison Table */}
          <div className="glass-card p-8 mt-12 overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr style={{ borderBottom: '3px solid #1a1a1a' }}>
                  <th className="pb-4 text-[#666] font-extrabold uppercase text-xs tracking-widest">Feature</th>
                  <th className="pb-4 font-extrabold text-[#8b5cf6] uppercase text-xs tracking-widest">CogNexus</th>
                  <th className="pb-4 text-[#999] font-extrabold uppercase text-xs tracking-widest">Traditional</th>
                  <th className="pb-4 text-[#999] font-extrabold uppercase text-xs tracking-widest">LLM-only</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr style={{ borderBottom: '2px solid #1a1a1a' }}>
                  <td className="py-4 font-bold">Works with SLMs</td>
                  <td className="py-4 text-[#15803d] font-bold">✓ ToT amplifies</td>
                  <td className="py-4 text-[#999]">N/A</td>
                  <td className="py-4 text-[#dc2626] font-bold">✗ Need GPT-4</td>
                </tr>
                <tr style={{ borderBottom: '2px solid #1a1a1a' }}>
                  <td className="py-4 font-bold">Generalizable</td>
                  <td className="py-4 text-[#15803d] font-bold">✓ Discovery-driven</td>
                  <td className="py-4 text-[#dc2626] font-bold">✗ Hardcoded</td>
                  <td className="py-4 text-[#a16207] font-bold">⚠ Prompt-dependent</td>
                </tr>
                <tr style={{ borderBottom: '2px solid #1a1a1a' }}>
                  <td className="py-4 font-bold">Self-correcting</td>
                  <td className="py-4 text-[#15803d] font-bold">✓ Observes & adapts</td>
                  <td className="py-4 text-[#dc2626] font-bold">✗ Fails silently</td>
                  <td className="py-4 text-[#a16207] font-bold">⚠ Limited</td>
                </tr>
                <tr>
                  <td className="py-4 font-bold">Template generation</td>
                  <td className="py-4 text-[#15803d] font-bold">✓ Auto Playwright</td>
                  <td className="py-4 text-[#999]">N/A</td>
                  <td className="py-4 text-[#dc2626] font-bold">✗ No</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Demo Section - Connected to Backend */}
      <DemoSection />

      {/* Get Started Section */}
      <section id="get-started" className="py-32 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="animated-border p-12 text-center">
            <div className="mb-6 flex justify-center">
              <div className="bg-[#c4b5fd] border-3 border-[#1a1a1a] p-3 shadow-[4px_4px_0px_#1a1a1a] inline-block" style={{ border: '3px solid #1a1a1a', boxShadow: '4px 4px 0px #1a1a1a' }}>
                <Image
                  src="/cognexus-logo.png"
                  alt="CogNexus"
                  width={64}
                  height={64}
                />
              </div>
            </div>
            <h2 className="text-4xl md:text-5xl font-extrabold mb-6 tracking-tight text-[#1a1a1a]">
              Ready to <span className="text-[#8b5cf6]">Get Started?</span>
            </h2>
            <p className="text-xl text-[#666] mb-10 max-w-2xl mx-auto font-medium">
              Install CogNexus Web Extractor and start extracting data autonomously in minutes.
            </p>

            {/* Install Command */}
            <div className="bg-[#1a1a1a] border-4 border-[#1a1a1a] shadow-[6px_6px_0px_#8b5cf6] p-4 mb-10 flex items-center justify-between max-w-xl mx-auto">
              <code className="text-[#4ade80] font-mono text-sm font-bold">pip install cognexus-extractor</code>
              <button className="text-white hover:bg-white/10 transition-colors px-3 py-1 text-sm font-bold uppercase border-2 border-white/30 hover:border-white">
                Copy
              </button>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button className="btn-primary flex items-center gap-2 text-lg group">
                Read Documentation
                <span className="group-hover:translate-x-1 transition-transform"><ArrowRightIcon /></span>
              </button>
              <a href="https://github.com/deba2k5/cognexus" target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center gap-2 text-lg">
                <GithubIcon />
                Star on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t-4 border-[#1a1a1a] bg-white">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <Image
              src="/cognexus-logo.png"
              alt="CogNexus"
              width={32}
              height={32}
            />
            <span className="font-extrabold text-[#1a1a1a]">CogNexus Web Extractor</span>
          </div>
          <p className="text-[#666] text-sm font-bold">
            Built with ❤️ for autonomous web exploration
          </p>
          <div className="flex items-center gap-6">
            <a href="#" className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors text-sm font-bold uppercase">Docs</a>
            <a href="https://github.com/deba2k5/cognexus" target="_blank" rel="noopener noreferrer" className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors text-sm font-bold uppercase">GitHub</a>
            <a href="#" className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors text-sm font-bold uppercase">Twitter</a>
          </div>
        </div>
      </footer>
    </>
  );
}
