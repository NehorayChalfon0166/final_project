---
name: code-review
description: Comprehensive code review for staged or recent changes. Reviews for bugs, security issues, performance, and best practices.
---

Review the code changes for $ARGUMENTS (default: staged changes via `git diff --cached`, or `git diff HEAD~1` if nothing is staged).

## Instructions

1. First, determine what to review:
   - If arguments are provided, use them (e.g., a file path, directory, or commit range)
   - Otherwise, check `git diff --cached` for staged changes
   - If nothing is staged, fall back to `git diff HEAD~1` for the last commit

2. Read the diff, then read the full context of each changed file to understand the surrounding code.

3. Analyze changes against this checklist:

### Correctness
- Logic errors, off-by-one errors, wrong comparisons
- Null/undefined/empty handling
- Race conditions or concurrency issues
- Incorrect error handling or swallowed exceptions

### Security
- Injection vulnerabilities (SQL, command, XSS)
- Hardcoded secrets, API keys, or credentials
- Insecure data handling or storage
- Missing input validation at system boundaries

### Performance
- Unnecessary allocations or copies
- N+1 queries or missing batch operations
- Missing caching for expensive operations
- Inefficient algorithms or data structures

### Edge Cases
- Empty inputs, zero values, negative numbers
- Boundary conditions
- Error paths and failure modes
- Unicode, encoding, or locale issues

### Code Quality
- Unclear or misleading naming
- Duplicated logic that should be extracted
- Dead code or unreachable branches
- Missing error messages that would aid debugging

## Output Format

Start with a one-line summary: "Reviewed N files, found X issues (Y critical, Z warnings)."

Then list each finding:

**[CRITICAL/WARNING/NIT]** `file_path:line_number`
> One-line description of the issue.
> **Fix:** Concrete suggestion for how to resolve it.

Group findings by file. If no issues are found, say so explicitly — don't invent problems.

End with a brief summary of the overall code quality and any patterns you noticed.
