# AWE Validator Agent

The **Validator Agent** acts as the quality assurance (QA) layer of the framework. It ensures that the data extracted by the **Extractor Agent** meets specific quality standards before it is accepted into the final dataset or used for learning.

## 🛡️ Core Responsibilities

1.  **Data Validation**: Checks every extracted field against a set of predefined rules (regex, length, format).
2.  **Quality Scoring**: Assigns a numeric quality score (0.0 - 1.0) to each extraction result.
3.  **Cross-Validation**: Performs logical checks across multiple fields (e.g., ensuring the "Bio" isn't just a copy of the "Name").
4.  **Feedback generation**: Produces actionable suggestions for the **Planner** or **Extractor** to improve future runs (e.g., "Name too short, try a different selector").

## 🛠️ Technical Implementation

The Validator is implemented in `validator.py`.

### 1. `ValidationReport`
The output artifact for every validation attempt.
*   `is_valid`: Boolean flag (Pass/Fail).
*   `quality_score`: Aggregate float score.
*   `errors`: Critical issues that cause failure (e.g., "Missing Name").
*   `warnings`: Non-critical issues (e.g., "Bio is very short").
*   `suggestions`: Hints for fixing the issues.

### 2. `ValidatorAgent`
The agent class containing the logic.

*   **Field Rules**: A dictionary defining constraints for each field type:
    *   `name`: Regex for proper names, min length 3.
    *   `email`: Standard email regex.
    *   `phone`: Phone number pattern.
    *   `bio`: Min/Max length constraints.
*   **Validation Logic**:
    *   `validate()`: The main pipeline.
    *   `_cross_validate()`: Checks relational consistency (e.g., if `title == name`, it's likely an extraction error).
    *   `_calculate_quality()`: Computes the final score based on field completeness, error penalties, and valid field counts.

## 📊 Quality Scoring Algorithm
 The quality score is calculated as:
$$
Quality = \text{Avg(Field Scores)} - \text{Penalty(Errors)} + \text{Bonus(Completeness)}
$$

*   **Field Scores**: unique scores for each field (1.0 = perfect, <1.0 = warning/error).
*   **Completeness Bonus**: Rewards records that have more non-empty fields.
*   **Threshold**: A record is considered "Valid" if it has **0 errors** and a quality score **>= 0.5**.

## 📦 Libraries & Dependencies

| Library | Purpose |
| :--- | :--- |
| **`re`** | Regular expressions for pattern matching (email, phones, names). |
| **`dataclasses`** | Structuring the validation report. |
| **`logging`** | Tracking validation metrics. |
