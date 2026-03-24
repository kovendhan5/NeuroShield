# Project Memory & Constraints

## Core Rules (NON-NEGOTIABLE)

### 1. File Structure Protection
- **NEVER** reorganize, rename, or move files without explicit user confirmation
- **ALWAYS** preserve exact file paths during edits
- If a path needs changing, ask first and wait for approval
- Validate all paths exist before creating/moving files

### 2. Dependencies & Imports
- **VERIFY** package.json/requirements.txt BEFORE any installation
- **NEVER** auto-upgrade or change versions without asking
- **ALWAYS** check import statements match actual file locations
- Run `npm list` or `pip list` before claiming a package is installed
- Validate all imports work: `import X from 'path'` → verify 'path' exists

### 3. Configuration Files
- **PROTECT**: .env, .env.local, .config files - never overwrite without backup
- **ASK FIRST** before touching: webpack.config.js, tsconfig.json, package.json scripts
- **ALWAYS** validate JSON syntax before writing
- Keep git-related files untouched unless specifically requested

### 4. Workflow Before Acting
```
User Request → Understand Context (ask questions if unclear)
                → Check Current State (ls, cat, grep - verify nothing)
                → Plan Changes (outline what will change)
                → Get Approval (summarize changes, wait for "ok" or "go")
                → Execute (make one change at a time)
                → Verify (test each change before next)
```

### 5. When Something Goes Wrong
- **STOP immediately** - don't make more changes
- **EXPLAIN** exactly what broke and why
- **SHOW** the exact error message
- **ASK** "should I rollback?" before proceeding
- **NEVER** assume you can fix it without asking

### 6. Code Changes
- **ALWAYS** review the full file context before editing
- Use str_replace, not wholesale rewrites
- One logical change per edit
- Run type checks/linting AFTER changes (via hooks)
- Test imports: if you change a file, verify dependent files still import correctly

### 7. Git & Version Control
- **NEVER** force push, rebase master, or delete branches
- **ALWAYS** run `git status` before any git operation
- Ask before committing on behalf of the user
- Preserve .gitignore and git history

## Project Info
**Model**: Claude Opus 4.6 (primary)
**Stack**: [User will specify after questionnaire]
**Build Tool**: [User will specify]
**Test Runner**: [User will specify]
**Package Manager**: npm or pip [User will specify]

## Session Checklist
Before each response that touches code:
- [ ] Verified file paths exist
- [ ] Checked imports are correct
- [ ] Confirmed dependencies listed
- [ ] Validated JSON/config syntax
- [ ] Explained change before executing
- [ ] Tested after changing

## Common Breakages to Prevent
1. Import paths that don't exist → Check file structure first
2. Circular imports → Review all affected files
3. Missing dependencies → Run `npm/pip list` first
4. Typos in package.json → Always copy exact names
5. Config file corruption → Always validate syntax
6. Overwriting user files → Always ask or backup first

## How to Use This File
- Claude reads it every session automatically
- Update it as you learn what works/what breaks
- Add project-specific rules here
- Reference it when Claude makes a mistake: "See CLAUDE.md rule 3"