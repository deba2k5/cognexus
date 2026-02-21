# AWE Planner Agent

The **Planner Agent** is the "brain" of the Agentic Web Explorer (AWE) framework. It is responsible for devising high-level exploration strategies based on the observations provided by the Observer Agent. It uses **Tree of Thought (ToT)** reasoning to evaluate multiple potential approaches before committing to an action plan.

## 🧠 Key Responsibilities

1.  **Hypothesis Generation**: Formulates different ways to extract data (e.g., "Use AJAX", "Crawl Pagination", "Scroll & Scrape").
2.  **ToT Evaluation**: Systematically evaluates each hypothesis by simulating its execution and scoring its likelihood of success.
3.  **Strategy Selection**: Picks the best-performing strategy based on confidence scores and past history.
4.  **Action Planning**: Converts the selected high-level strategy into a concrete sequence of Playwright actions (NAVIGATE, CLICK, EXTRACT).
5.  **Failure Recovery**: Maintains a history of failed strategies to avoid repeating mistakes during replanning.

## 🛠️ Technical Implementation

The Planner is implemented in `planner.py` and centers around the `PlannerAgent` class.

### 1. `ExplorationStrategy`
A data class that represents a candidate approach. It includes:
*   `name`: Unique identifier (e.g., "ajax_all").
*   `actions`: A list of `Action` objects to be executed.
*   `confidence`: A float (0.0 - 1.0) indicating estimated success rate.
*   `reasoning`: Textual explanation of why this strategy was chosen.

### 2. `PlannerAgent`
The main agent class. It supports two modes of planning:

#### A. Tree of Thought (ToT) Planning (`_plan_with_tot`)
When enabled, the Planner uses the `ToTEngine` to:
1.  **Generate Thoughts**: The LLM suggests possible strategies given the `PageObservation`.
2.  **Evaluate Thoughts**: The Planner simulates the execution of each thought against the page's capabilities (e.g., "Can we use AJAX?" -> "Yes, endpoint found").
3.  **Search**: It explores the tree of possibilities to find the optimal path.
4.  **Reflection**: Optionally reflects on the decision to improve future reasoning.

#### B. Heuristic Planning (`_plan_heuristic`)
A fast, rule-based fallback when ToT is disabled or unavailable. It prioritizes strategies in this order:
1.  **AJAX**: If endpoints are detected (highest reliability).
2.  **Pagination**: If page numbers or "Next" buttons are found.
3.  **Scroll**: If infinite scroll is detected.
4.  **Direct Extraction**: Fallback to scraping visible items.
5.  **Search**: If a search bar is present.

## 📋 Strategy Templates

The Planner uses pre-defined templates to standardize common scraping patterns:

| Strategy | Description | Typical Actions |
| :--- | :--- | :--- |
| **`ajax_all`** | Check for hidden API endpoints. | `NAVIGATE` to API, `EXTRACT` JSON. |
| **`pagination_crawl`** | Standard multi-page scraping. | `EXTRACT` current, `CLICK` Next, Repeat. |
| **`scroll_load`** | Infinite scroll pages. | `SCROLL` to bottom, `WAIT`, `EXTRACT`. |
| **`direct_extract`** | Simple single-page scrape. | `EXTRACT` visible elements. |
| **`search_browse`** | Search-driven exploration. | `TYPE` query, `CLICK` search, `EXTRACT` results. |

## 📦 Libraries & Dependencies

The Planner relies on:

| Library | Purpose |
| :--- | :--- |
| **`..reasoning.tot`** | The core Tree of Thought engine. |
| **`..core.types`** | Definitions for `Action`, `PageObservation`, `Thought`. |
| **`dataclasses`** | Structuring strategy and action objects. |
| **`logging`** | Tracing the decision-making process. |

## 🔄 Planning Workflow

1.  **Receive Context**: The Planner gets the `AgentContext` containing the latest `PageObservation` from the Observer.
2.  **Check Capabilities**: It verifies what features the page has (AJAX, Pagination, etc.).
3.  **Formulate Plan**:
    *   *If ToT*: LLM generates thoughts -> Thoughts are scored -> Best thought becomes strategy.
    *   *If Heuristic*: Checks conditions hierarchically -> Selects first matching strategy.
4.  **Create Actions**: The abstract strategy is converted into specific `Action` objects (e.g., `CLICK` on `.next-page`).
5.  **Output**: Returns the `ExplorationStrategy` to the Orchestrator for execution.
