---
name: validate-project
description: Validate project health - check file structure, dependencies, imports, and config syntax
allowed-tools: Read, Grep, Glob, ExecuteCode
auto-invoke: false
---

<!-- Tip: Use /create-skill in chat to generate content with agent assistance -->

# Project Validation Skill

Validate your entire project for common issues that cause breakages.

## What This Checks
1. **File Structure** - confirms all key files exist and are in expected locations
2. **Dependencies** - verifies package.json/requirements.txt matches what's actually installed
3. **Imports** - checks that all import statements resolve to real files
4. **Config Files** - validates JSON/YAML syntax (tsconfig, webpack, etc.)
5. **Git Status** - shows any uncommitted changes that might conflict

## How to Invoke
When Claude breaks something, you ask:
```
/validate-project
```

Or to check a specific area:
```
/validate-project --imports
/validate-project --deps
/validate-project --config
```

## Output Format
For each check, show:
- ✅ PASS - specific file or dependency verified
- ❌ FAIL - exact error with line number/location
- ⚠️  WARNING - potential issue that might cause problems later

Example output:
```
FILE STRUCTURE:
✅ src/index.js exists
✅ src/components/ directory exists
❌ FAIL: src/types/index.ts missing (2 files import from it)

IMPORTS:
❌ FAIL: components/Button.jsx imports { useHook } from '../hooks' 
         but hooks.js not found in that path
✅ All other 24 imports resolve correctly

DEPENDENCIES:
✅ React 18.2.0 installed, package.json requires ^18.0.0
❌ FAIL: lodash listed in package.json but not installed
         (npm list shows: not installed)
```

## When to Use
- **Before any major refactor** - ensure baseline is healthy
- **When Claude says "project is broken"** - diagnose the real issue
- **After Claude makes changes** - verify nothing broke
- **Daily in long sessions** - catch drift before it compounds