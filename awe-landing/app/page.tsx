"use client";

import { useState, useEffect } from "react";
import DemoSection from "./components/DemoSection";
import Image from "next/image";

// Icons as SVG components
const BrainIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8">
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
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8">
    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const RefreshIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8">
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
    <path d="M8 16H3v5" />
  </svg>
);

const BookIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8">
    <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    <path d="M8 7h6" />
    <path d="M8 11h8" />
  </svg>
);

const NetworkIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8">
    <circle cx="18" cy="5" r="3" />
    <circle cx="6" cy="12" r="3" />
    <circle cx="18" cy="19" r="3" />
    <path d="m8.59 13.51 6.83 3.98" />
    <path d="m8.59 10.49 6.83-3.98" />
  </svg>
);

const SparklesIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8">
    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    <path d="M5 3v4" />
    <path d="M19 17v4" />
    <path d="M3 5h4" />
    <path d="M17 19h4" />
  </svg>
);

const ArrowRightIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
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
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-6 h-6">
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
    },
    {
      icon: <EyeIcon />,
      title: "Vision-First Approach",
      description: "Combines screenshot analysis with DOM understanding for superior page comprehension, not just HTML parsing.",
      tag: "Perception",
    },
    {
      icon: <RefreshIcon />,
      title: "Self-Correcting",
      description: "Observes failures and adapts strategies in real-time. 90%+ self-recovery rate on complex tasks.",
      tag: "Resilience",
    },
    {
      icon: <BookIcon />,
      title: "Template Learning",
      description: "Automatically generates reusable Playwright extraction patterns from successful explorations.",
      tag: "Learning",
    },
    {
      icon: <NetworkIcon />,
      title: "Knowledge Persistence",
      description: "Builds a persistent knowledge graph of learned approaches for zero-shot generalization.",
      tag: "Memory",
    },
    {
      icon: <SparklesIcon />,
      title: "Discovery-Driven",
      description: "No hardcoded selectors or site-specific logic. Pure autonomous exploration and discovery.",
      tag: "Autonomy",
    },
  ];

  const stats = [
    { value: "100%", label: "Profile Discovery", subtext: "vs 5% hardcoded" },
    { value: "95%+", label: "Extraction Accuracy", subtext: "Production-grade" },
    { value: "90%+", label: "Self-Recovery Rate", subtext: "Auto-healing" },
    { value: "85%+", label: "Zero-Shot Success", subtext: "New sites" },
  ];

  return (
    <>
      {/* Background Effects */}
      <div className="bg-mesh" />
      <div className="bg-grid" />

      {/* Floating Orbs */}
      <div className="floating-orb orb-1" />
      <div className="floating-orb orb-2" />
      <div className="floating-orb orb-3" />

      {/* Navigation */}
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${scrolled ? "glass-card py-3 shadow-2xl" : "py-5"
          }`}
        style={{ backdropFilter: scrolled ? "blur(24px)" : "none", borderRadius: scrolled ? "0" : "0" }}
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
              <span className="text-xl font-bold tracking-tight gradient-text">CogNexus</span>
              <span className="text-[10px] text-gray-500 tracking-widest uppercase -mt-1">Web Extractor</span>
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
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-white/70 hover:text-white transition-colors"
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
              <Image
                src="/cognexus-logo.png"
                alt="CogNexus Web Extractor"
                width={140}
                height={140}
                className="float"
                priority
              />
            </div>
          </div>

          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full border border-cyan-500/30 bg-cyan-500/5 mb-8 backdrop-blur-sm">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-sm text-cyan-300 font-medium">Powered by SLMs with ToT Reasoning</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold leading-[1.05] mb-6 tracking-tight">
            <span className="block text-white">Autonomous Web</span>
            <span className="gradient-text glow-text">Exploration at Scale</span>
          </h1>

          {/* Subheadline */}
          <p className="text-lg md:text-xl text-gray-400 max-w-3xl mx-auto mb-12 leading-relaxed">
            A production-grade multi-agent framework for autonomous data extraction.
            Works with <span className="text-cyan-400 font-semibold">small language models</span> through
            Tree of Thought reasoning — powered by <span className="gradient-text font-semibold">CogNexus</span>.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <a href="#demo" className="btn-primary flex items-center gap-2 text-lg group">
              Try Live Demo
              <span className="group-hover:translate-x-1 transition-transform"><ArrowRightIcon /></span>
            </a>
            <button className="btn-secondary flex items-center gap-2 text-lg">
              <GithubIcon />
              View on GitHub
            </button>
          </div>

          {/* Code Preview */}
          <div className="glass-card overflow-hidden max-w-3xl mx-auto text-left code-preview-card">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5 bg-black/30">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <div className="w-3 h-3 rounded-full bg-green-500/80" />
              <span className="ml-4 text-xs text-gray-500 font-mono">quick_start.py</span>
            </div>
            <pre className="p-6 text-sm overflow-x-auto leading-relaxed">
              <code>
                <span className="text-purple-400">from</span>{" "}
                <span className="text-cyan-400">cognexus</span>{" "}
                <span className="text-purple-400">import</span>{" "}
                <span className="text-white">WebExtractor</span>
                {"\n\n"}
                <span className="text-gray-500"># Define your goal</span>
                {"\n"}
                <span className="text-white">extractor</span> ={" "}
                <span className="text-cyan-400">WebExtractor</span>
                <span className="text-white">(</span>
                {"\n"}
                {"    "}
                <span className="text-orange-400">model</span>=
                <span className="text-green-400">&quot;llama-3.1-8b-instant&quot;</span>,
                {"\n"}
                {"    "}
                <span className="text-orange-400">tot_enabled</span>=
                <span className="text-purple-400">True</span>
                {"\n"}
                <span className="text-white">)</span>
                {"\n\n"}
                <span className="text-gray-500"># Extract data autonomously</span>
                {"\n"}
                <span className="text-white">results</span> ={" "}
                <span className="text-purple-400">await</span>{" "}
                <span className="text-white">extractor.explore(</span>
                {"\n"}
                {"    "}
                <span className="text-orange-400">url</span>=
                <span className="text-green-400">&quot;https://example.edu/faculty/&quot;</span>,
                {"\n"}
                {"    "}
                <span className="text-orange-400">objective</span>=
                <span className="text-green-400">&quot;Extract all faculty profiles&quot;</span>
                {"\n"}
                <span className="text-white">)</span>
              </code>
            </pre>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 scroll-indicator">
          <a href="#features" className="flex flex-col items-center text-gray-500 hover:text-white transition-colors">
            <span className="text-sm mb-2">Explore</span>
            <ChevronDownIcon />
          </a>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-32 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-purple-500/20 bg-purple-500/5 mb-6">
              <span className="text-sm text-purple-300">✦ Capabilities</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
              Built for <span className="gradient-text">Real-World</span> Extraction
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Everything you need to build production-grade autonomous web scrapers that actually work.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="glass-card feature-card p-8 cursor-pointer group"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center gap-3 mb-5">
                  <div className="feature-icon w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center text-purple-400 group-hover:text-cyan-400 transition-colors">
                    {feature.icon}
                  </div>
                  <span className="text-[10px] font-semibold uppercase tracking-widest text-cyan-400/60 px-2 py-1 rounded-full border border-cyan-500/20">
                    {feature.tag}
                  </span>
                </div>
                <h3 className="text-xl font-semibold mb-3 group-hover:text-white transition-colors">{feature.title}</h3>
                <p className="text-gray-400 leading-relaxed text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Architecture Section */}
      <section id="architecture" className="py-32 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-cyan-500/20 bg-cyan-500/5 mb-6">
              <span className="text-sm text-cyan-300">⚙ System Design</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
              Multi-Agent <span className="gradient-text">Architecture</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Six specialized agents working in harmony through a state machine orchestrator.
            </p>
          </div>

          {/* Architecture Diagram */}
          <div className="glass-card p-8 md:p-12">
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <div className="architecture-box text-center group">
                <div className="text-2xl mb-2 group-hover:scale-110 transition-transform">🔭</div>
                <h4 className="font-semibold text-purple-300">Observer</h4>
                <p className="text-sm text-gray-500">Vision + DOM Analysis</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-2xl mb-2 group-hover:scale-110 transition-transform">🧠</div>
                <h4 className="font-semibold text-cyan-300">Planner</h4>
                <p className="text-sm text-gray-500">ToT Strategy</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-2xl mb-2 group-hover:scale-110 transition-transform">⚡</div>
                <h4 className="font-semibold text-orange-300">Executor</h4>
                <p className="text-sm text-gray-500">Playwright Actions</p>
              </div>
            </div>
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <div className="architecture-box text-center group">
                <div className="text-2xl mb-2 group-hover:scale-110 transition-transform">✅</div>
                <h4 className="font-semibold text-green-300">Validator</h4>
                <p className="text-sm text-gray-500">Quality Assurance</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-2xl mb-2 group-hover:scale-110 transition-transform">📊</div>
                <h4 className="font-semibold text-pink-300">Extractor</h4>
                <p className="text-sm text-gray-500">Data Extraction</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-2xl mb-2 group-hover:scale-110 transition-transform">📚</div>
                <h4 className="font-semibold text-yellow-300">Learner</h4>
                <p className="text-sm text-gray-500">Template Generation</p>
              </div>
            </div>

            {/* State Machine */}
            <div className="border-t border-white/5 pt-8">
              <h4 className="text-center text-gray-400 mb-6 text-sm uppercase tracking-widest">State Machine Flow</h4>
              <div className="flex flex-wrap items-center justify-center gap-2 md:gap-3 text-sm">
                <span className="state-pill bg-purple-500/20 text-purple-300 border-purple-500/30">Observe</span>
                <span className="text-gray-600">→</span>
                <span className="state-pill bg-cyan-500/20 text-cyan-300 border-cyan-500/30">Think</span>
                <span className="text-gray-600">→</span>
                <span className="state-pill bg-blue-500/20 text-blue-300 border-blue-500/30">Plan</span>
                <span className="text-gray-600">→</span>
                <span className="state-pill bg-orange-500/20 text-orange-300 border-orange-500/30">Act</span>
                <span className="text-gray-600">→</span>
                <span className="state-pill bg-green-500/20 text-green-300 border-green-500/30">Validate</span>
                <span className="text-gray-600">→</span>
                <span className="state-pill bg-yellow-500/20 text-yellow-300 border-yellow-500/30">Learn</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Performance Section */}
      <section id="performance" className="py-32 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/20 bg-green-500/5 mb-6">
              <span className="text-sm text-green-300">📈 Benchmarks</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
              Proven <span className="gradient-text">Performance</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Benchmarked with Llama 3.1 8B on university faculty extraction tasks.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <div
                key={index}
                className="glass-card p-8 text-center feature-card group"
              >
                <div className="stat-number mb-2 group-hover:scale-110 transition-transform inline-block">{stat.value}</div>
                <h4 className="text-lg font-semibold mb-1">{stat.label}</h4>
                <p className="text-sm text-gray-500">{stat.subtext}</p>
              </div>
            ))}
          </div>

          {/* Comparison Table */}
          <div className="glass-card p-8 mt-12 overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="pb-4 text-gray-400 font-medium">Feature</th>
                  <th className="pb-4 font-medium">
                    <span className="gradient-text font-bold">CogNexus</span>
                  </th>
                  <th className="pb-4 text-gray-500 font-medium">Traditional</th>
                  <th className="pb-4 text-gray-500 font-medium">LLM-only</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr className="border-b border-white/5">
                  <td className="py-4">Works with SLMs</td>
                  <td className="py-4 text-green-400">✓ ToT amplifies</td>
                  <td className="py-4 text-gray-500">N/A</td>
                  <td className="py-4 text-red-400">✗ Need GPT-4</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-4">Generalizable</td>
                  <td className="py-4 text-green-400">✓ Discovery-driven</td>
                  <td className="py-4 text-red-400">✗ Hardcoded</td>
                  <td className="py-4 text-yellow-400">⚠ Prompt-dependent</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-4">Self-correcting</td>
                  <td className="py-4 text-green-400">✓ Observes & adapts</td>
                  <td className="py-4 text-red-400">✗ Fails silently</td>
                  <td className="py-4 text-yellow-400">⚠ Limited</td>
                </tr>
                <tr>
                  <td className="py-4">Template generation</td>
                  <td className="py-4 text-green-400">✓ Auto Playwright</td>
                  <td className="py-4 text-gray-500">N/A</td>
                  <td className="py-4 text-red-400">✗ No</td>
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
              <Image
                src="/cognexus-logo.png"
                alt="CogNexus"
                width={64}
                height={64}
                className="opacity-80"
              />
            </div>
            <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
              Ready to <span className="gradient-text">Get Started?</span>
            </h2>
            <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
              Install CogNexus Web Extractor and start extracting data autonomously in minutes.
            </p>

            {/* Install Command */}
            <div className="glass-card p-4 mb-10 flex items-center justify-between max-w-xl mx-auto">
              <code className="text-cyan-300 font-mono text-sm">pip install cognexus-extractor</code>
              <button className="text-gray-400 hover:text-white transition-colors px-3 py-1 rounded hover:bg-white/10 text-sm">
                Copy
              </button>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button className="btn-primary flex items-center gap-2 text-lg group">
                Read Documentation
                <span className="group-hover:translate-x-1 transition-transform"><ArrowRightIcon /></span>
              </button>
              <button className="btn-secondary flex items-center gap-2 text-lg">
                <GithubIcon />
                Star on GitHub
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-white/5">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <Image
              src="/cognexus-logo.png"
              alt="CogNexus"
              width={32}
              height={32}
            />
            <span className="font-semibold">CogNexus Web Extractor</span>
          </div>
          <p className="text-gray-500 text-sm">
            Built with ❤️ for autonomous web exploration
          </p>
          <div className="flex items-center gap-6">
            <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">Docs</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">GitHub</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">Twitter</a>
          </div>
        </div>
      </footer>
    </>
  );
}
