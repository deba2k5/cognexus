# AWE Extractor Agent

The **Extractor Agent** is responsible for the core mission of the framework: turning raw HTML into structured, usable data. It employs a multi-layered strategy to ensure high accuracy and resilience, falling back to more expensive methods only when necessary.

## 🏗️ Extraction Strategy (The "waterfall")

The agent attempts to extract data using three methods in order of preference:

1.  **Learned Patterns** (Fastest & Cheapest):
    *   If the **Learner Agent** has previously identified a successful pattern for this site, it is applied first.
    *   This uses cached CSS selectors mapped to specific fields.
    *   It is extremely fast and costs zero tokens.

2.  **Heuristics** (Fast & Reliable):
    *   If no pattern exists, it uses the `HeuristicExtractor`.
    *   This relies on a curated list of common CSS selectors for faculty/profile pages (e.g., `.profile-name`, `h1.title`, `a[href^="mailto:"]`).
    *   It also uses JavaScript-based logic to parse complex lists like "Education" or "Research Interests".

3.  **LLM Extraction** (Most Capable & Costly):
    *   If fields are still missing, the HTML is truncated to the main content area.
    *   It is sent to a Small Language Model (SLM) or LLM with a specific prompt to "extract valid JSON".
    *   This handles complex, non-standard layouts that defy regex or simple selectors.

## 🛠️ Technical Implementation

The Extractor is implemented in `extractor.py`.

### 1. `HeuristicExtractor`
A standalone class containing the rule-based logic.
*   **Field Selectors**: A dictionary mapping fields (`name`, `email`, `phone`) to lists of potential CSS selectors.
*   **Education Parsing**: specialized JS that looks for "Education" headers and parses following `<ul>` or `<p>` tags.
*   **Research Areas**: Similar JS logic to find and split research interest lists.

### 2. `ExtractorAgent`
The main agent that orchestrates the waterfall process.
*   **`extract()`**: The primary entry point. It aggregates results from all three methods.
*   **Validation**:
    *   Checks if essential fields (like `name`) are present.
    *   Validates formats (e.g., email contains `@`).
    *   Calculates a "confidence score" based on the ratio of found vs. expected fields.
*   **Truncation**: Smartly truncates HTML before sending to LLM to save context window, focusing on `<main>`, `<article>`, or content divs.

## 📊 Data Schema
The agent targets the following standard fields by default, though this is configurable:

*   `name` (String)
*   `title` (String)
*   `email` (String)
*   `phone` (String)
*   `department` (String)
*   `office` (String)
*   `education` (List of Strings)
*   `research_areas` (List of Strings)
*   `bio` (String)

## 📦 Libraries & Dependencies

| Library | Purpose |
| :--- | :--- |
| **`playwright.async_api`** | Querying selectors and evaluating extraction JS. |
| **`json`** | Parsing LLM responses. |
| **`re`** | HTML truncation and pattern matching. |
| **`urllib.parse`** | fixing relative URLs in images/links. |

## ✅ Validation Logic
Data is only considered "valid" if:
1.  A `name` is found (length 3-100 chars).
2.  If an `email` is found, it must contain `@`.
3.  At least **30%** of the requested fields were successfully extracted.
