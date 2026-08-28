# Proposal 001: Harness gene improvements derived from build-pmars (2026-08-28)

- **Status:** proposal
- **Date:** 2026-08-28
- **Source trace:** `jobs/2026-08-28__05-06-22/build-pmars__GdorEsV`
- **Harness under test:** `meta-v10` (parent `meta-v9`)
- **Outcome of that run:** `reward = 0` (3/4 verifier tests passed, 1 failed)

## Context

The `terminal-bench/build-pmars` task requires building pMARS from **Debian
source** (no X11), placing the source under `/app`, and installing the binary to
`/usr/local/bin/pmars`. The agent functionally succeeded (binary ran, output
`Results: 12 30 8`, no X11 deps), but the verifier failed it on
`test_debian_source_used`, which asserts that `/app/pmars-*/debian` exists.

Root cause in the trace: the agent correctly ran `apt-get source pmars` (which
already extracts to `/app/pmars-0.9.4/debian`), then **manually re-extracted
the tarballs**, which placed `debian/` at `/app/debian` instead of inside
`/app/pmars-0.9.4`. The build worked, but the verifier's directory check failed.

This is not a task-specific bug — it is a class of failure that a generic agent
harness should prevent. Three generalizable improvements are proposed below,
each mapped to one of the seven harness genes
(`prompt`, `tools`, `control`, `memory`, `recovery`, `verification`, `skills`).

## Proposed mutations

### 1. Pre-termination acceptance-criteria check → `verification` gene

- **Current state:** `verification.enabled: false`. The gene is recorded in
  lineage but not wired into the runtime loop; it is a dormant boolean.
- **Problem:** The agent declared success after producing `Results: 12 30 8`
  and never checked the other explicit acceptance criterion (the `debian/`
  directory location). No mechanism forced it to verify every stated criterion.
- **Proposed mutation:** Promote `verification` from a boolean to a real
  **pre-termination acceptance-criteria checklist**: before the agent may
  finish, it must re-read every explicit success criterion in the task and
  produce a concrete evidence command for each (e.g. `ls /app/pmars-0.9.4/debian`).
- **Why this gene:** "verify work before done" is exactly the `verification`
  gene's purpose; keeping it generic makes it reusable across all tasks.

### 2. State-preservation rule → `prompt` gene

- **Current state:** Global rules 1–4 live in `prompt.system`. Rule 3 only
  bans repeating a command that has **already failed** with the same parameters.
- **Problem:** This run is the inverse — a **successful** `apt-get source`
  already produced the correct state, yet the agent re-ran an equivalent
  extraction that relocated/overwrote that state and broke it.
- **Proposed mutation:** Add **rule 5 (state non-destruction)**: *inspect the
  current environment state before acting; never re-run a step that has already
  succeeded and could overwrite or relocate an existing correct state.*
- **Why this gene:** It is a behavioral heuristic, identical in nature to the
  existing global rules, so it belongs in `prompt.system`.

### 3. Prefer ecosystem-native tooling → `prompt` gene

- **Current state:** No guidance exists about preferring package/ecosystem
  build tooling over manual reconstruction.
- **Problem:** The agent hand-edited the upstream `Makefile` and manually
  untarred instead of using Debian's own build chain (`dpkg-buildpackage` /
  `debian/rules`), which would have preserved `debian/` in place, applied
  Debian patches, and handled no-X11 automatically.
- **Proposed mutation:** Add **rule 6 (native tooling first)**: *for
  extraction/compilation/installation, prefer the ecosystem's own build and
  package-manager commands (e.g. source-package build CLIs, framework CLIs)
  over manually re-implementing extraction and compilation steps.*
- **Why this gene:** Also a behavioral heuristic → `prompt.system` (can be
  merged with rule 5 as a single `prompt` mutation, or split into two derives).

## Gene summary

| Improvement | Gene | Mutation |
|-------------|------|----------|
| 1. Acceptance-criteria check | `verification` | boolean → pre-termination checklist requiring evidence per criterion |
| 2. State non-destruction | `prompt` | add rule 5 (inspect state; don't re-run successful steps) |
| 3. Native tooling first | `prompt` | add rule 6 (prefer ecosystem CLI over manual reconstruction) |

Per the lineage model (mutate one gene at a time), recommend:
- `meta-v11` ← derive from `meta-v10`, mutate `verification`
- `meta-v12` ← derive, mutate `prompt` (rules 5 + 6)

## Out of scope (genes not changed)

`tools`, `control`, `memory`, `recovery`, `skills` require no change for these
improvements. Note: `memory.strategy: llm_summary` was correct but never fired
because the conversation (92 messages / 486s) did not cross the compaction
trigger threshold — that is a threshold tuning issue, not a gene-selection one.
