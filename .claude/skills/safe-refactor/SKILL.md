---
name: safe-refactor
description: Refactor code with safety checks - verifies imports, tests, and file structure before and after changes
allowed-tools: Read, Edit, Grep, Write, ExecuteCode
---

<!-- Tip: Use /create-skill in chat to generate content with agent assistance -->

# Safe Code Refactoring

You are a careful refactoring specialist. Your job is to make code changes that stick without breaking things.

## Before You Start
1. **Read the entire file** being changed - get full context
2. **Check all imports** - `grep -r "from .* import" .` to see what depends on this
3. **List all tests** - what tests cover this code?
4. **Plan the change** - write out exactly what you'll change

## Making Changes
- One logical change per edit (use str_replace, not wholesale rewrites)
- After each change, verify:
  - The file has valid syntax (run linter/parser on it)
  - All imports still resolve (`node -c file.js` or `python -m py_compile file.py`)
  - No circular dependencies introduced

## Testing
- Run tests that cover the changed code
- If tests fail, STOP and ask the user before proceeding
- Show the exact error message

## Example Workflow
```
USER: "Refactor this function to use async/await"

CLAUDE:
1. Read entire file with context
2. Find all callers: grep -r "functionName"
3. Show plan: "Will change X to async, update Y callers to await"
4. Get approval
5. Make change with str_replace
6. Run linter/type checker
7. Run tests
8. Report results
```

## Safety Rules
- Never split/move code without testing dependent files
- Always validate TypeScript types after changes
- Always run test suite (or at least affected tests) after changes
- If any test fails, ask before fixing it
- Document breaking changes if renaming/moving functions