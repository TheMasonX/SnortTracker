---
name: council
description: Run an LLM council review with 6 diverse subagent personalities. Use for architecture review, implementation plan critique, risk assessment, or quality audit. Produces a structured council report with ratings, MUST-FIX items, and a consolidated verdict.
---

# Council Review Skill

Convene a multi-perspective LLM council to review a project, plan, or implementation. Six subagents with distinct personalities independently audit the target and return structured findings. Their reports are compiled into a single council review document with a unified verdict, MUST-FIX list, and revised roadmap.

## When to Use

- After completing a major implementation phase — get independent critique
- Before starting a new project phase — validate the plan
- When the user asks for a "review," "audit," "second opinion," or "council"
- When the user says "what are we missing?" or "is this the right approach?"
- During architecture decision-making — get diverse perspectives

## Council Members

Launch all six in parallel. Each is an Explore subagent with a distinct personality and scope.

| Role | Personality | Focus |
|------|-------------|-------|
| **The Architect** | Systems thinker, values clean abstractions and long-term maintainability | Architecture, coupling, separation of concerns, abstraction levels |
| **The Pragmatist** | Battle-scarred shipper, values working software over perfect design | Scope realism, shippability, timeline, what to cut |
| **The Tester** | Quality obsessive, believes untested code is broken | Test coverage, edge cases, failure modes, what's untested |
| **The Guardian** | Privacy-first, treats user data as radioactive | Data hygiene, local-first enforcement, leak vectors, retention |
| **The Perf Engineer** | Optimizes everything, thinks in nanoseconds | CPU/memory/GC pressure, target hardware readiness, hidden allocations |
| **The User Advocate** | Represents the actual human user | UX, CLI design, on-ramp friction, debuggability, real-world usefulness |

## Prompt Template

For each subagent, use this structure (customize the file list per role):

```
You are "[ROLE]" — [PERSONALITY DESCRIPTION].

Your task: Review the [PROJECT NAME] and provide critical findings.

Read these files in full:
- [FILE LIST — tailored to role]

Then provide a concise review covering:
1. [QUESTION 1]
2. [QUESTION 2]
3. [QUESTION 3]
4. [QUESTION 4]
5. [QUESTION 5]
6. Rate [DIMENSION] 1-10 and give one MUST-FIX item.

Be [TONE INSTRUCTION]. Return your findings as structured markdown.
```

## Compilation

After all six return, compile into a single council review document at:

```
Data/Pages/Council/council-review-{DD-MM-YY-HH-MM-SS}.md
```

The compiled document should contain:

1. **Header** — date, scope, council composition table with ratings
2. **Unanimous Findings** — items all 6 agreed on (or 5+)
3. **Per-Member Findings** — each member's report condensed
4. **Consolidated MUST-FIX List** — priority-ordered table
5. **Revised Roadmap/Plan** — if reordering is recommended
6. **Verdict** — 1-paragraph summary

Additionally, update or create `ROADMAP.md` in the repo root with the refined plan (pure roadmap, no mention of council methodology).

## Customization

- **Change the council size:** Use 3 for quick reviews, 6 for thorough audits
- **Swap roles:** Replace any role with a domain-specific one (e.g., "ML Engineer" for model review)
- **Adjust depth:** Add "thorough" for critical milestones, "quick" for routine checkpoints
- **Focus scope:** Limit file list to specific modules for targeted reviews

## Example Usage

User says: "Review our implementation so far."
→ Launch 6 subagents with tailored file lists.
→ Compile findings into `council-review-{timestamp}.md`.
→ Update `ROADMAP.md` with refined plan.
→ Present the consolidated MUST-FIX list to the user.
