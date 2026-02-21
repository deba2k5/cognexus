# AWE Executor Agent

The **Executor Agent** is the "hands" of the Agentic Web Explorer (AWE) framework. It is responsible for carrying out the concrete actions planned by the Planner Agent. It interfaces directly with the **Playwright** browser automation library to interact with web pages and extract data.

## 🛠️ Key Responsibilities

1.  **Action Execution**: Performs browser actions like `NAVIGATE`, `CLICK`, `TYPE`, `SCROLL`, and `WAIT`.
2.  **Data Extraction**: Runs specific extraction routines (JSON, Profile Links, CSS Selectors) to pull data from the page context.
3.  **Error Handling**: Catches Playwright exceptions (e.g., TimeoutError, ElementNotFound) and reports them without crashing the agent.
4.  **Result Reporting**: Returns detailed `ActionResult` objects containing success status, data, duration, and error messages.
5.  **State Management**: Updates the browser state (current URL, page content) through its actions.

## ⚙️ Technical Implementation

The Executor is implemented in `executor.py` and is built around the `ExecutorAgent` class.

### 1. `ActionResult`
A data class that standardizes the output of any action:
*   `success`: Boolean indicating if the action worked.
*   `data`: The return value (e.g., extracted text, screenshot base64).
*   `error`: Error message if failed.
*   `duration_ms`: Execution time in milliseconds.

### 2. `ExecutorAgent`
The main class that wraps Playwright's `Page` object.

#### distinct Capabilities:
*   **Navigation**: `_execute_navigate` - Handles `goto` with network idle waits.
*   **Interaction**:
    *   `_execute_click`: Clicks elements.
    *   `_execute_type`: Fills input fields.
    *   `_execute_scroll`: Scrolls to top/bottom/element or by pixel amount.
    *   `_execute_hover` & `_execute_select`: For menus and dropdowns.
*   **Extraction**:
    *   `_extract_json`: Intelligently finds JSON blobs in `<pre>` tags or body text (common in raw API responses).
    *   `_extract_profile_links`: Uses heuristic JS to find links that look like people/faculty.
    *   `_extract_with_selector`: Standard CSS selector extraction.

## 🧠 Smart Extraction Logic
The Executor isn't just a dumb wrapper; it contains logic to handle messy web data:

*   **JSON Unescaping**: Automatically fixes escaped slashes (`\/`) in raw JSON responses.
*   **Link Deduplication**: Removes duplicate URLs during extraction to keep data clean.
*   **Heuristic Fallbacks**: If a specific extraction fails, it can fallback to broader strategies (e.g., extracting all links).

## 📦 Libraries & Dependencies

The Executor relies heavily on:

| Library | Purpose |
| :--- | :--- |
| **`playwright.async_api`** | The core browser automation engine. |
| **`re`** | Regex for finding JSON patterns and filtering URLs. |
| **`urllib.parse`** | Joining relative URLs to make them absolute. |
| **`asyncio`** | Handling asynchronous browser operations. |

## 🔄 Execution Flow

1.  **Receive Plan**: The Orchestrator sends a list of `Action` objects (from the Planner).
2.  **Iterate**: The Executor loops through each action.
3.  **Execute**:
    *   Maps `ActionType` (e.g., `CLICK`) to the corresponding private method (`_execute_click`).
    *   Runs the Playwright command with a timeout.
4.  **Capture Result**: Records success/failure and any extracted data.
5.  **Stop on Failure**: If configured, it can halt the sequence immediately upon an error (useful for dependent chains).
6.  **Return**: Sends the list of `ActionResult`s back to the Orchestrator.
