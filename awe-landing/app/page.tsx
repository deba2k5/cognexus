"use client";

import { useState, useEffect } from "react";
import DemoSection from "./components/DemoSection";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Brain, 
  Eye, 
  RefreshCcw, 
  BookOpen, 
  Network, 
  Sparkles, 
  ArrowRight, 
  ChevronDown, 
  Menu,
  X,
  Check
} from "lucide-react";

const GithubIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="0" className="w-[1em] h-[1em]">
    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
  </svg>
);

// Framer Motion Variants
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.15 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  show: { 
    opacity: 1, 
    y: 0,
    transition: { type: "spring" as const, stiffness: 300, damping: 24 }
  }
};

const TypewriterTerminal = () => {
  const codeLines = [
    'from cognexus import WebExtractor',
    '',
    '# Define your goal',
    'extractor = WebExtractor(',
    '    model="llama-3.1-8b-instant",',
    '    tot_enabled=True',
    ')',
    '',
    '# Extract data autonomously',
    'results = await extractor.explore(',
    '    url="https://example.edu/faculty/",',
    '    objective="Extract all faculty profiles"',
    ')'
  ];

  const [displayedText, setDisplayedText] = useState('');
  const [lineIndex, setLineIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (lineIndex < codeLines.length) {
      if (charIndex < codeLines[lineIndex].length) {
        const timeout = setTimeout(() => {
          setDisplayedText(prev => prev + codeLines[lineIndex][charIndex]);
          setCharIndex(c => c + 1);
        }, 15 + Math.random() * 20); // Faster, slightly variable typing speed
        return () => clearTimeout(timeout);
      } else {
        const timeout = setTimeout(() => {
          setDisplayedText(prev => prev + '\n');
          setLineIndex(l => l + 1);
          setCharIndex(0);
        }, 150); // Pause at end of line
        return () => clearTimeout(timeout);
      }
    }
  }, [lineIndex, charIndex]);

  const handleCopy = () => {
    navigator.clipboard.writeText(codeLines.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getLineLabel = (idx: number) => {
    return (idx + 1).toString().padStart(2, '0');
  };

  return (
    <motion.div 
      variants={itemVariants}
      className="glass-card overflow-hidden max-w-3xl mx-auto text-left code-preview-card"
    >
      <div className="flex items-center justify-between px-4 py-3 bg-[#1a1a1a] border-b-3 border-[#333]" style={{ borderBottom: '3px solid #333' }}>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-[#f87171] border-2 border-black rounded-full" />
          <div className="w-4 h-4 bg-[#facc15] border-2 border-black rounded-full" />
          <div className="w-4 h-4 bg-[#4ade80] border-2 border-black rounded-full" />
          <span className="ml-4 text-xs text-gray-400 font-mono font-bold uppercase">quick_start.py</span>
        </div>
        <button 
          onClick={handleCopy}
          className="text-gray-400 hover:text-white transition-colors flex items-center gap-1 text-xs font-bold bg-[#333] px-2 py-1 rounded"
        >
          {copied ? <Check size={14} className="text-green-400" /> : <div className="i-lucide-copy w-3 h-3" />}
          {copied ? 'COPIED!' : 'COPY'}
        </button>
      </div>
      
      <div className="p-6 text-sm overflow-x-auto leading-relaxed bg-[#111] font-mono text-gray-300 relative min-h-[320px]">
        <div className="flex">
          {/* Line Numbers */}
          <div className="flex flex-col text-gray-600 select-none pr-4 border-r border-[#333] mr-4 text-right">
            {codeLines.map((_, i) => (
              <span key={i}>{getLineLabel(i)}</span>
            ))}
          </div>
          
          {/* Code */}
          <div className="whitespace-pre">
            {displayedText}
            {lineIndex < codeLines.length && (
              <span className="animate-pulse bg-white w-2.5 h-4 inline-block align-middle ml-1" />
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default function Home() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const features = [
    {
      icon: <Brain size={28} />,
      title: "Tree of Thought Reasoning",
      description: "Multi-path exploration enables smaller models to perform complex reasoning through systematic evaluation and backtracking.",
      tag: "Core Engine",
      color: "bg-[#c4b5fd]",
      iconColor: "text-[#6d28d9]",
    },
    {
      icon: <Eye size={28} />,
      title: "Vision-First Approach",
      description: "Combines screenshot analysis with DOM understanding for superior page comprehension, not just HTML parsing.",
      tag: "Perception",
      color: "bg-[#a5f3fc]",
      iconColor: "text-[#0891b2]",
    },
    {
      icon: <RefreshCcw size={28} />,
      title: "Self-Correcting",
      description: "Observes failures and adapts strategies in real-time. 90%+ self-recovery rate on complex tasks.",
      tag: "Resilience",
      color: "bg-[#bbf7d0]",
      iconColor: "text-[#15803d]",
    },
    {
      icon: <BookOpen size={28} />,
      title: "Template Learning",
      description: "Automatically generates reusable Playwright extraction patterns from successful explorations.",
      tag: "Learning",
      color: "bg-[#fed7aa]",
      iconColor: "text-[#c2410c]",
    },
    {
      icon: <Network size={28} />,
      title: "Knowledge Persistence",
      description: "Builds a persistent knowledge graph of learned approaches for zero-shot generalization.",
      tag: "Memory",
      color: "bg-[#fbcfe8]",
      iconColor: "text-[#be185d]",
    },
    {
      icon: <Sparkles size={28} />,
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
    <div className="overflow-x-hidden">
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
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between">
          <div className="flex items-center gap-3 relative z-[60]">
            <Image
              src="/cognexus-logo.png"
              alt="CogNexus Logo"
              width={40}
              height={40}
              className="logo-glow sm:w-[44px] sm:h-[44px]"
            />
            <div className="flex flex-col">
              <span className="text-lg sm:text-xl font-extrabold tracking-tight text-[#1a1a1a]">CogNexus</span>
              <span className="text-[9px] sm:text-[10px] text-[#666] tracking-widest uppercase -mt-1 font-bold">Web Extractor</span>
            </div>
          </div>
          
          {/* Desktop Nav */}
          <div className="hidden lg:flex items-center gap-8">
            <a href="#features" className="nav-link">Features</a>
            <a href="#architecture" className="nav-link">Architecture</a>
            <a href="#performance" className="nav-link">Performance</a>
            <a href="#demo" className="nav-link">Live Fetch</a>
            <a href="#get-started" className="nav-link">Get Started</a>
          </div>
          
          <div className="hidden lg:flex items-center gap-4">
            <a
              href="https://github.com/deba2k5/cognexus"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors"
            >
              <GithubIcon />
            </a>
            <a href="#demo" className="btn-primary flex items-center gap-2">
              Try Demo
              <ArrowRight size={20} />
            </a>
          </div>

          {/* Mobile Hamburger Toggle */}
          <div className="lg:hidden relative z-[60]">
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 border-2 border-black bg-white shadow-[2px_2px_0px_#000] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none transition-all"
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 280, damping: 25 }}
            className="fixed inset-0 z-50 bg-[#e8e4dc] border-l-4 border-black pt-28 px-6 flex flex-col gap-6 lg:hidden"
          >
            <div className="bg-grid absolute inset-0 opacity-50 pointer-events-none" />
            <a href="#features" onClick={() => setMobileMenuOpen(false)} className="text-2xl font-black uppercase text-black border-b-2 border-black pb-2">Features</a>
            <a href="#architecture" onClick={() => setMobileMenuOpen(false)} className="text-2xl font-black uppercase text-black border-b-2 border-black pb-2">Architecture</a>
            <a href="#performance" onClick={() => setMobileMenuOpen(false)} className="text-2xl font-black uppercase text-black border-b-2 border-black pb-2">Performance</a>
            <a href="#demo" onClick={() => setMobileMenuOpen(false)} className="text-2xl font-black uppercase text-black border-b-2 border-black pb-2">Live Fetch</a>
            
            <div className="mt-8 flex flex-col gap-4">
              <a href="#demo" onClick={() => setMobileMenuOpen(false)} className="btn-primary flex items-center justify-center gap-2 text-xl py-4">
                Try Demo <ArrowRight />
              </a>
              <a href="https://github.com/deba2k5/cognexus" target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center justify-center gap-2 text-xl py-4">
                <GithubIcon /> GitHub
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hero Section */}
      <section className="min-h-screen flex flex-col items-center justify-center px-4 pt-32 pb-20 relative">
        <motion.div 
          className="max-w-5xl mx-auto text-center z-10 w-full"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
          {/* Logo Large */}
          <motion.div variants={itemVariants} className="mb-8 flex justify-center">
            <div className="hero-logo-container">
              <div className="inline-block bg-[#c4b5fd] border-4 border-[#1a1a1a] p-3 sm:p-4 shadow-[8px_8px_0px_#1a1a1a]">
                <Image
                  src="/cognexus-logo.png"
                  alt="CogNexus Web Extractor"
                  width={90}
                  height={90}
                  className="float sm:w-[120px] sm:h-[120px]"
                  priority
                />
              </div>
            </div>
          </motion.div>

          {/* Badge */}
          <motion.div variants={itemVariants} className="inline-flex items-center gap-2 px-4 sm:px-5 py-2.5 bg-[#fef08a] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-8" style={{ border: '3px solid #1a1a1a' }}>
            <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 bg-[#22c55e] border-2 border-[#1a1a1a]" />
            <span className="text-xs sm:text-sm text-[#1a1a1a] font-bold uppercase tracking-wide">Powered by SLMs with ToT Reasoning</span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1 variants={itemVariants} className="text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-extrabold leading-[1.05] mb-6 tracking-tight text-[#1a1a1a]">
            <span className="block">Autonomous Web</span>
            <span className="text-[#8b5cf6]" style={{ textShadow: '3px 3px 0px #1a1a1a' }}>Exploration at Scale</span>
          </motion.h1>

          {/* Subheadline */}
          <motion.p variants={itemVariants} className="text-base sm:text-lg md:text-xl text-[#444] max-w-3xl mx-auto mb-10 sm:mb-12 leading-relaxed font-medium px-2">
            A production-grade multi-agent framework for autonomous data extraction.
            Works with <span className="bg-[#a5f3fc] px-1 border-b-3 border-[#0891b2] font-bold text-black">small language models</span> through
            Tree of Thought reasoning.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6 mb-16 w-full px-4 sm:px-0">
            <a href="#demo" className="btn-primary flex items-center justify-center gap-2 text-base sm:text-lg group w-full sm:w-auto">
              Try Live Demo
              <motion.span 
                animate={{ x: [0, 4, 0] }} 
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                <ArrowRight size={20} />
              </motion.span>
            </a>
            <a href="https://github.com/deba2k5/cognexus" target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center justify-center gap-2 text-base sm:text-lg w-full sm:w-auto">
              <GithubIcon />
              View on GitHub
            </a>
          </motion.div>

          {/* Code Preview - Replaced with Typewriter */}
          <TypewriterTerminal />
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div 
          initial={{ opacity: 0 }} 
          animate={{ opacity: 1 }} 
          transition={{ delay: 1.5 }}
          className="absolute bottom-4 sm:bottom-8 scroll-indicator hidden sm:flex"
        >
          <a href="#features" className="flex flex-col items-center text-[#666] hover:text-[#1a1a1a] transition-colors font-bold z-10">
            <span className="text-xs sm:text-sm mb-2 uppercase tracking-widest">Explore</span>
            <ChevronDown size={24} />
          </a>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 sm:py-32 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            className="text-center mb-16 sm:mb-20"
          >
            <div className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 bg-[#c4b5fd] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-6" style={{ border: '3px solid #1a1a1a' }}>
              <span className="text-xs sm:text-sm text-[#1a1a1a] font-bold uppercase">✦ Capabilities</span>
            </div>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold mb-4 sm:mb-6 tracking-tight text-[#1a1a1a]">
              Built for <span className="text-[#8b5cf6]">Real-World</span> Extraction
            </h2>
            <p className="text-base sm:text-xl text-[#666] max-w-2xl mx-auto font-medium">
               Build production-grade autonomous web scrapers that actually work.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ delay: index * 0.1 }}
                className="glass-card feature-card p-6 sm:p-8 cursor-pointer group"
              >
                <div className="flex items-center justify-between mb-5">
                  <div className={`feature-icon w-12 h-12 sm:w-14 sm:h-14 flex items-center justify-center ${feature.color} border-3 border-[#1a1a1a] shadow-[3px_3px_0px_#1a1a1a] ${feature.iconColor}`} style={{ border: '3px solid #1a1a1a', boxShadow: '3px 3px 0px #1a1a1a' }}>
                    {feature.icon}
                  </div>
                  <span className="text-[9px] sm:text-[10px] font-extrabold uppercase tracking-widest text-[#1a1a1a] px-2 py-1 sm:px-3 sm:py-1.5 bg-[#fef08a] border-2 border-[#1a1a1a]">
                    {feature.tag}
                  </span>
                </div>
                <h3 className="text-lg sm:text-xl font-extrabold mb-3 text-[#1a1a1a]">{feature.title}</h3>
                <p className="text-[#666] leading-relaxed text-sm font-medium">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Architecture Section */}
      <section id="architecture" className="py-20 sm:py-32 px-4 sm:px-6 relative overflow-hidden">
        <div className="max-w-7xl mx-auto relative z-10">
          <motion.div 
            initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
            className="text-center mb-16 sm:mb-20"
          >
            <div className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 bg-[#a5f3fc] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-6" style={{ border: '3px solid #1a1a1a' }}>
              <span className="text-xs sm:text-sm text-[#1a1a1a] font-bold uppercase">⚙ System Design</span>
            </div>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold mb-4 sm:mb-6 tracking-tight text-[#1a1a1a]">
              Multi-Agent <span className="text-[#8b5cf6]">Architecture</span>
            </h2>
            <p className="text-base sm:text-xl text-[#666] max-w-2xl mx-auto font-medium">
              Six specialized agents working in harmony through a state machine orchestrator.
            </p>
          </motion.div>

          {/* Architecture Diagram */}
          <motion.div 
            initial={{ scale: 0.95, opacity: 0 }}
            whileInView={{ scale: 1, opacity: 1 }}
            viewport={{ once: true }}
            className="glass-card p-6 sm:p-8 md:p-12 w-full overflow-x-hidden"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mb-8">
              <div className="architecture-box text-center group">
                <div className="text-3xl sm:text-4xl mb-2 group-hover:scale-110 transition-transform">🔭</div>
                <h4 className="font-extrabold text-[#1a1a1a] text-lg">Observer</h4>
                <p className="text-xs sm:text-sm text-[#666] font-medium mt-1">Vision + DOM Analysis</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl sm:text-4xl mb-2 group-hover:scale-110 transition-transform">🧠</div>
                <h4 className="font-extrabold text-[#1a1a1a] text-lg">Planner</h4>
                <p className="text-xs sm:text-sm text-[#666] font-medium mt-1">ToT Strategy</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl sm:text-4xl mb-2 group-hover:scale-110 transition-transform">⚡</div>
                <h4 className="font-extrabold text-[#1a1a1a] text-lg">Executor</h4>
                <p className="text-xs sm:text-sm text-[#666] font-medium mt-1">Playwright Actions</p>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mb-12">
              <div className="architecture-box text-center group">
                <div className="text-3xl sm:text-4xl mb-2 group-hover:scale-110 transition-transform">✅</div>
                <h4 className="font-extrabold text-[#1a1a1a] text-lg">Validator</h4>
                <p className="text-xs sm:text-sm text-[#666] font-medium mt-1">Quality Assurance</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl sm:text-4xl mb-2 group-hover:scale-110 transition-transform">📊</div>
                <h4 className="font-extrabold text-[#1a1a1a] text-lg">Extractor</h4>
                <p className="text-xs sm:text-sm text-[#666] font-medium mt-1">Data Extraction</p>
              </div>
              <div className="architecture-box text-center group">
                <div className="text-3xl sm:text-4xl mb-2 group-hover:scale-110 transition-transform">📚</div>
                <h4 className="font-extrabold text-[#1a1a1a] text-lg">Learner</h4>
                <p className="text-xs sm:text-sm text-[#666] font-medium mt-1">Template Generation</p>
              </div>
            </div>

            {/* State Machine */}
            <div className="border-t-3 border-[#1a1a1a] pt-8 overflow-x-auto pb-4" style={{ borderTop: '3px solid #1a1a1a' }}>
              <h4 className="text-center text-[#666] mb-6 text-sm uppercase tracking-widest font-bold">State Machine Flow</h4>
              <div className="flex items-center min-w-[max-content] mx-auto justify-center gap-2 sm:gap-3 text-xs sm:text-sm px-4">
                <span className="state-pill bg-[#c4b5fd] text-[#1a1a1a]">Observe</span>
                <span className="text-[#1a1a1a] font-extrabold text-lg sm:text-xl">→</span >
                <span className="state-pill bg-[#a5f3fc] text-[#1a1a1a]">Think</span>
                <span className="text-[#1a1a1a] font-extrabold text-lg sm:text-xl">→</span >
                <span className="state-pill bg-[#93c5fd] text-[#1a1a1a]">Plan</span>
                <span className="text-[#1a1a1a] font-extrabold text-lg sm:text-xl">→</span >
                <span className="state-pill bg-[#fed7aa] text-[#1a1a1a]">Act</span>
                <span className="text-[#1a1a1a] font-extrabold text-lg sm:text-xl">→</span >
                <span className="state-pill bg-[#bbf7d0] text-[#1a1a1a]">Validate</span>
                <span className="text-[#1a1a1a] font-extrabold text-lg sm:text-xl">→</span >
                <span className="state-pill bg-[#fef08a] text-[#1a1a1a]">Learn</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Performance Section */}
      <section id="performance" className="py-20 sm:py-32 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div 
             initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
             className="text-center mb-16 sm:mb-20"
          >
            <div className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 bg-[#bbf7d0] border-3 border-[#1a1a1a] shadow-[4px_4px_0px_#1a1a1a] mb-6" style={{ border: '3px solid #1a1a1a' }}>
              <span className="text-xs sm:text-sm text-[#1a1a1a] font-bold uppercase">📈 Benchmarks</span>
            </div>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold mb-4 sm:mb-6 tracking-tight text-[#1a1a1a]">
              Proven <span className="text-[#8b5cf6]">Performance</span>
            </h2>
            <p className="text-base sm:text-xl text-[#666] max-w-2xl mx-auto font-medium">
              Benchmarked with Llama 3.1 8B on complex extraction tasks.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className={`glass-card p-6 sm:p-8 text-center feature-card group ${stat.color}`}
              >
                <div className="stat-number mb-2 transition-transform duration-300 group-hover:scale-110 inline-block">{stat.value}</div>
                <h4 className="text-base sm:text-lg font-extrabold mb-1 text-[#1a1a1a]">{stat.label}</h4>
                <p className="text-xs sm:text-sm text-[#666] font-bold">{stat.subtext}</p>
              </motion.div>
            ))}
          </div>

          {/* Comparison Table */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="glass-card p-4 sm:p-8 mt-12 overflow-x-auto w-full"
          >
            <table className="w-full text-left min-w-[600px]">
              <thead>
                <tr style={{ borderBottom: '3px solid #1a1a1a' }}>
                  <th className="pb-3 sm:pb-4 text-[#666] font-extrabold uppercase text-[10px] sm:text-xs tracking-widest pl-2">Feature</th>
                  <th className="pb-3 sm:pb-4 font-extrabold text-[#8b5cf6] uppercase text-[10px] sm:text-xs tracking-widest">CogNexus</th>
                  <th className="pb-3 sm:pb-4 text-[#999] font-extrabold uppercase text-[10px] sm:text-xs tracking-widest">Traditional</th>
                  <th className="pb-3 sm:pb-4 text-[#999] font-extrabold uppercase text-[10px] sm:text-xs tracking-widest pr-2">LLM-only</th>
                </tr>
              </thead>
              <tbody className="text-xs sm:text-sm">
                <tr style={{ borderBottom: '2px solid #1a1a1a' }}>
                  <td className="py-3 sm:py-4 font-bold pl-2">Works with SLMs</td>
                  <td className="py-3 sm:py-4 text-[#15803d] font-bold">✓ ToT amplifies</td>
                  <td className="py-3 sm:py-4 text-[#999]">N/A</td>
                  <td className="py-3 sm:py-4 text-[#dc2626] font-bold pr-2">✗ Need GPT-4</td>
                </tr>
                <tr style={{ borderBottom: '2px solid #1a1a1a' }}>
                  <td className="py-3 sm:py-4 font-bold pl-2">Generalizable</td>
                  <td className="py-3 sm:py-4 text-[#15803d] font-bold">✓ Discovery-driven</td>
                  <td className="py-3 sm:py-4 text-[#dc2626] font-bold">✗ Hardcoded</td>
                  <td className="py-3 sm:py-4 text-[#a16207] font-bold pr-2">⚠ Prompt-dependent</td>
                </tr>
                <tr style={{ borderBottom: '2px solid #1a1a1a' }}>
                  <td className="py-3 sm:py-4 font-bold pl-2">Self-correcting</td>
                  <td className="py-3 sm:py-4 text-[#15803d] font-bold">✓ Observes & adapts</td>
                  <td className="py-3 sm:py-4 text-[#dc2626] font-bold">✗ Fails silently</td>
                  <td className="py-3 sm:py-4 text-[#a16207] font-bold pr-2">⚠ Limited</td>
                </tr>
                <tr>
                  <td className="py-3 sm:py-4 font-bold pl-2">Template generation</td>
                  <td className="py-3 sm:py-4 text-[#15803d] font-bold">✓ Auto Playwright</td>
                  <td className="py-3 sm:py-4 text-[#999]">N/A</td>
                  <td className="py-3 sm:py-4 text-[#dc2626] font-bold pr-2">✗ No</td>
                </tr>
              </tbody>
            </table>
          </motion.div>
        </div>
      </section>

      {/* Demo Section */}
      <DemoSection />

      {/* Get Started Section */}
      <section id="get-started" className="py-20 sm:py-32 px-4 sm:px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div 
            initial={{ scale: 0.95, opacity: 0 }}
            whileInView={{ scale: 1, opacity: 1 }}
            viewport={{ once: true }}
            className="animated-border p-8 sm:p-12 text-center"
          >
            <div className="mb-6 flex justify-center">
              <div className="bg-[#c4b5fd] border-3 border-[#1a1a1a] p-3 shadow-[4px_4px_0px_#1a1a1a] inline-block" style={{ border: '3px solid #1a1a1a', boxShadow: '4px 4px 0px #1a1a1a' }}>
                <Image
                  src="/cognexus-logo.png"
                  alt="CogNexus"
                  width={56}
                  height={56}
                  className="sm:w-[64px] sm:h-[64px]"
                />
              </div>
            </div>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold mb-4 sm:mb-6 tracking-tight text-[#1a1a1a]">
              Ready to <span className="text-[#8b5cf6]">Get Started?</span>
            </h2>
            <p className="text-base sm:text-xl text-[#666] mb-8 sm:mb-10 max-w-2xl mx-auto font-medium">
              Start extracting data autonomously in minutes.
            </p>

            {/* Install Command */}
            <div className="bg-[#1a1a1a] border-4 border-[#1a1a1a] shadow-[4px_4px_0px_#8b5cf6] sm:shadow-[6px_6px_0px_#8b5cf6] p-3 sm:p-4 mb-8 sm:mb-10 flex flex-col sm:flex-row items-start sm:items-center justify-between max-w-xl mx-auto gap-4">
              <code className="text-[#4ade80] font-mono text-xs sm:text-sm font-bold text-left w-full sm:w-auto px-2">pip install cognexus-extractor</code>
              <button 
                onClick={() => navigator.clipboard.writeText('pip install cognexus-extractor')}
                className="text-white hover:bg-white/10 active:scale-95 transition-all px-4 py-2 text-xs font-bold uppercase border-2 border-white/30 hover:border-white w-full sm:w-auto flex items-center justify-center gap-2"
              >
                <div className="i-lucide-copy" /> Copy
              </button>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button className="btn-primary flex items-center justify-center gap-2 text-base sm:text-lg group w-full sm:w-auto">
                Read Documentation
                <span className="group-hover:translate-x-1 transition-transform"><ArrowRight size={20} /></span>
              </button>
              <a href="https://github.com/deba2k5/cognexus" target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center justify-center gap-2 text-base sm:text-lg w-full sm:w-auto">
                <GithubIcon />
                Star on GitHub
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 sm:py-12 px-4 sm:px-6 border-t-4 border-[#1a1a1a] bg-white">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left">
          <div className="flex items-center gap-3">
            <Image
              src="/cognexus-logo.png"
              alt="CogNexus"
              width={28}
              height={28}
              className="sm:w-[32px] sm:h-[32px]"
            />
            <span className="font-extrabold text-[#1a1a1a] text-sm sm:text-base">CogNexus Web Extractor</span>
          </div>
          <p className="text-[#666] text-xs sm:text-sm font-bold">
            Built with ❤️ for autonomous web exploration
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-6">
            <a href="#" className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors text-xs sm:text-sm font-bold uppercase">Docs</a>
            <a href="https://github.com/deba2k5/cognexus" target="_blank" rel="noopener noreferrer" className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors text-xs sm:text-sm font-bold uppercase">GitHub</a>
            <a href="#" className="text-[#1a1a1a] hover:text-[#8b5cf6] transition-colors text-xs sm:text-sm font-bold uppercase">Twitter</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
