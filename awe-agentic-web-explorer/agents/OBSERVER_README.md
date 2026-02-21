# AWE Observer Agent

The **Observer Agent** is the "eyes" of the Agentic Web Explorer (AWE) framework. It is responsible for perceiving, analyzing, and understanding web pages using a combination of **computer vision (screenshots)** and **DOM analysis**.

## 🚀 Key Responsibilities

1.  **Visual Perception**: Takes screenshots of pages to understand layout and context.
2.  **DOM Analysis**: Parses the HTML structure to identify key elements (cards, profiles, pagination).
3.  **Page Classification**: Determines if a page is a Profile, Directory, Search, Login, etc.
4.  **Pattern Detection**: Identifies AJAX endpoints, lazy loading mechanisms, and repeated structures.
5.  **Semantic Representation**: Converts complex HTML into a simplified "Semantic DOM" for LLM consumption.

## 🛠️ Technical Implementation

The Observer is implemented in `observer.py` and consists of two main classes:

### 1. `DOMAnalyzer`
This class handles the parsing and low-level analysis of the page HTML.

*   **Page Classification**: Uses regex patterns on URLs (`/faculty/`, `/profile/`) and specific content keywords.
*   **Loading Detection**: Analyzes HTML for indicators of infinite scroll, "Load More" buttons, or AJAX data attributes (`data-src`).
*   **Pattern Recognition**:
    *   **Cards**: Identifies repeated container classes (e.g., `.card`, `.profile-item`).
    *   **Pagination**: Detects numbered lists or "Next" buttons using CSS selectors.
    *   **AJAX**: Extracts potential API endpoints from `data-*` attributes and script tags.
*   **Element Extraction**: Uses a JavaScript injection to score and extract links that look like people's names.

### 2. `ObserverAgent`
This is the main agent class that orchestrates the observation process.

*   **Screenshotting**: Uses Playwright to capture screenshots in base64 format.
*   **Vision Analysis**: Sends the screenshot to a Vision LLM (like Gemma-2-9B-It or GPT-4o) with a specific prompt to get high-level layout understanding.
*   **Fusion**: Combines the DOM analysis results with the Vision LLM's insights to create a comprehensive `PageObservation`.
*   **State Management**: Saves the observation to the shared agent state.

## 📦 Libraries & Dependencies

The Observer relies on the following key libraries:

| Library | Purpose |
| :--- | :--- |
| **`playwright.async_api`** | Browser automation, navigating pages, taking screenshots, evaluating JavaScript. |
| **`re`** | Regular expressions for pattern matching in URLs and HTML. |
| **`base64`** | Encoding screenshots for LLM consumption. |
| **`urllib.parse`** | URL manipulation and joining. |
| **`logging`** | tailored logging of observation events. |
| **`dataclasses`** | structuring the observation data objects. |
| **`asyncio`** | Asynchronous execution handling. |

## 🧠 Workflows

### The `observe` Workflow
1.  **Capture**: Playwright captures the page title, full HTML, and a viewport screenshot.
2.  **DOM Analysis**:
    *   Regex checks on URL.
    *   CSS selector counts for cards and pagination.
    *   JavaScript evaluation to find "person-like" links.
3.  **Vision Analysis** (if enabled):
    *   Screenshot is sent to the Vision Model.
    *   Model answers questions about page type, layout, and visible items.
4.  **Synthesis**:
    *   `DOMAnalyzer` creates a `semantic_dom` XML-like string.
    *   Results are merged into a `PageObservation` object.
5.  **Persistence**: Observation is stored in the `AgentState`.

## 🔍 Semantic DOM
Instead of feeding raw HTML to the planning agents, the Observer generates a **Semantic DOM**:

```xml
<page url="https://example.edu/faculty" type="directory">
  <content_loading>static</content_loading>
  <cards_found patterns="1">
    <pattern selector=".profile-card" count="24"/>
  </cards_found>
  <pagination type="numbered" total="12" current="1"/>
  <profile_links count="24">
    <link href="/faculty/jane-doe" text="Dr. Jane Doe"/>
    <!-- ... -->
  </profile_links>
</page>
```

This compact format allows smaller LLMs (SLMs) to reason effectively about the page without context window overflow.
