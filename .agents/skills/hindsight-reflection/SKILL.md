---
name: hindsight-reflection
description: Evaluate a working implementation against a clean-slate design to eliminate exploratory accretion, rebuild cleanly, or drop the direction before review.
---

# Hindsight reflection

The first working implementation is often an exploratory draft shaped by the journey of debugging: compensatory layering, patch-on-patch overrides, and structural compromises made to get tests to pass. Equally often, an implementation reveals that the original hypothesis or premise was flawed—that the anticipated savings or benefits are an illusion, dwarfed by new incidental complexity.

Once tests or verification scripts are passing, you possess full knowledge of the problem that was missing at the start. Do not fall into the sunk-cost trap: passing tests are your **safety ratchet**, not a reason to preserve an accreted draft or defend a questionable premise. Because the tests already capture the discovered requirements and edge cases, discarding draft implementation files or abandoning the direction entirely carries zero risk of regression.

At the completion of an implementation, ensure the upstream PR is active, launch the review loop in parallel, execute this protocol, post the evaluation in dialogue, and yield the turn.

## 1. Discoveries & Learnings

Name what was learned during implementation that was not obvious at the outset (e.g. unexpected interactions, domain nuances, hidden bottlenecks, color detection edge cases, or cleaner mathematical/geometric models).

## 2. Greenfield Design (The Clean Slate)

With the benefit of hindsight and everything learned, sketch the simplest, most direct architecture you would build from a blank slate. What is the unified model or representation that makes special-casing and compensatory glue unnecessary? Apply the restraint check: simpler and more direct, not more clever.

## 3. Hostile Falsification & Reality Audit

Do not treat this reflection as a justification exercise or victory lap. Actively attempt to falsify the value of the change before evaluating its code structure:

- **Hypothesis vs. Empirical Reality:** What was the original premise or anticipated win (e.g., solver speed, cleaner notation, detection accuracy, visual responsiveness)? What does the actual execution timing or practical verification reveal?
- **Steelmanned Case to Drop It:** Construct the strongest adversarial argument against landing this change. What ongoing cognitive burden, statefulness, dual concepts, or maintenance tax does it impose on future agents?
- **Zero-Sum / Negative-Sum Check:** Did solving this problem shift friction elsewhere or increase the total number of moving parts, states, or steps in the end-to-end workflow?
- **Premise Independence:** Treat all ideas—whether originated by the user or the agent—as empirical hypotheses to be challenged with data, not mandates to rationalize. A clean, passing implementation of a negative-value feature is still negative value.

## 4. Delta Check (Structural Audit)

Compare the working draft against the greenfield design, auditing for exploratory accretion:

- **Compensatory layering vs. unified model:** Are overrides, extra branches, or wrappers stacked to handle newly discovered cases instead of finding the underlying model that makes those cases natural?
- **Sediment of the journey:** Does the diff reflect the chronological order of bug fixes and patches rather than an intentional, coherent design?
- **Downstream symptom relief:** Does the change work around awkward upstream structures from the outside rather than fixing the contract or structure at the root?
- **Inverted signal-to-noise:** Do glue, overrides, special cases, and coordination dominate the core intent?
- **Partial surface alignment:** Did the change update the happy path while leaving adjacent edge cases, face orientations, or input modes with residual inconsistency?

### Recommendation

In all cases, ensure the upstream PR is active, launch the review loop in parallel, post the evaluation and recommendation in dialogue, and yield the turn:

1. **Drop it:** Evaluate this first. Does the change truly pull its weight against the hostile falsification audit? If even the clean greenfield design introduces ongoing cognitive burden or complexity disproportionate to the value delivered, or if the original premise did not survive empirical testing, recommend dropping the direction. Abandoning a spike after proving its cost is a high-value engineering win that protects Eigencube's simplicity ("When in doubt, leave it out"), not wasted effort.
   - _Triggers:_ Flawed premise, unproven or illusionary benefits, disproportionate complexity for marginal value, existing primitives are sufficient, or cognitive burden outweighs the benefit.
   - _Guardrail:_ Root-cause fixes for regressions or broken core solver/cube invariants must be solved, not dropped.
2. **Rebuild cleanly:** (Expected for non-trivial work, at most once). The value justifies the change and survives the falsification audit, but the draft exhibits exploratory accretion, journey sediment, or structural divergence. Outline the greenfield plan in dialogue and recommend rebuilding cleanly. Upon user confirmation, keep the newly written tests (the safety ratchet), discard the draft implementation files (`git restore --source origin/main <files>` and clean untracked draft files), and implement the clean greenfield design in one direct pass.
3. **Proceed:** (High bar). Choose this only if the change is proven justified by the hostile falsification audit, the steelmanned case to drop it is conclusively refuted with concrete evidence, and the draft already matches the greenfield design with zero compensatory layering or journey sediment.
