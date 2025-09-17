# To-Do: Caching Plan

## 1. Exact Key Cache (same question → instant answer)
- Remember the final search results for a specific question.
- If the user asks the exact same question again (with the same top-K and same filters), return the cached answer immediately.
- No database work needed in this case.

## 2. Semantic Cache (similar question → reuse answer)
- Detect when a new question has different wording but the same meaning as a previous one.
- Reuse the earlier cached answer instead of doing a full database query.
- Saves computation and speeds up responses.