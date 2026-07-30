# ADR 0007 — Gold labels are machine-generated, not human-reviewed

**Status:** Accepted, with a mandated follow-up
**Date:** 2026-07-29

## Context

The occasion field in the source catalogue was rule-derived. Scoring a rule
engine against labels produced by the same rules is circular, so an
independent reference set was required.

The intended design (ADR 0004) was a 300-row held-out set annotated by a human
with no machine assistance. Under project time constraints this was not
possible.

## Decision

The gold sets were labelled by an **archetype-based machine annotator**
(`src/gift_recommender/labeling/annotate_v2.py`) rather than by a human.

Independence from the rule engine is preserved structurally:

| | Rule engine (`taxonomy.yaml`) | Annotator (`annotate_v2.py`) |
|---|---|---|
| Category (14 values) | ✅ used | ❌ never read |
| Sub-category (70 values) | ❌ | ✅ used |
| Product name text | ❌ | ✅ used (bilingual) |
| Price | partially | ✅ social price gates |
| Gender / age fields | ✅ | ❌ |

The two methods therefore share no input signal, and agreement between them is
evidence rather than tautology. But both are machine heuristics, so the
resulting figures measure **inter-method agreement**, not accuracy against
human judgement.

## Consequences

**Accepted.** Every report, README table and presentation slide citing these
numbers must state that labels are machine-generated. Claiming human ground
truth would be a misrepresentation and is trivially exposed by the question
"who labelled it?".

**Mandated follow-up.** A human must label **100 rows** of `gold_test.csv`
unassisted and report agreement with the machine annotator. One hour of work
converts the artefact from "AI-generated" to "AI-generated, human-validated at
0.XX agreement" — a materially different claim.

Until that is done, the honest phrasing is:

> Macro-F1 0.777 against an independently constructed machine-annotated
> reference set. Human validation pending.

## Alternatives rejected

- **Skip evaluation.** Leaves Precision@K unmeasurable and the entire
  evaluation chapter unsupported.
- **Evaluate against the source catalogue's own occasion field.** Purely
  circular; the field is what we are replacing.
- **Label a smaller human set (~50 rows).** Confidence intervals too wide to
  distinguish 0.70 from 0.85 at ten labels.
