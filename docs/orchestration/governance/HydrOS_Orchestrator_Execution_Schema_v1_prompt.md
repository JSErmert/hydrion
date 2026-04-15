# HydrOS — AROD (AlignFlow Recursive Orchestration Distillation) Schema v1 

First, write this prompt content to:
`docs/orchestration/governance/HydrOS_Orchestrator_Execution_Schema_v1_prompt.md`

If the folder path does not yet exist, create it first.

Then execute the task exactly as written below.

Operate under HydrOS x AlignFlow Orchestration Contract v3.1 at:
`docs/orchestration/governance/HydrOS_x_AlignFlow_Orchestration_Contract_v3.1.md`

## Objective

Create the formal orchestration execution schema for HydrOS milestone work.

Write the output to:
`docs/orchestration/governance/HydrOS_Orchestrator_Execution_Schema_v1.md`

Also create one required insight file for this pass at:
`docs/orchestration/insights/Governance/HydrOS_Orchestrator_Schema_v1_insight_<short_slug>.md`

If the insights folder path does not yet exist, create it first.

After creating each markdown artifact, also print the full markdown in chat per Contract v3.1.

## Purpose

This schema is not a milestone-specific artifact.

It is a reusable execution template that defines how HydrOS orchestration artifacts should be structured, sequenced, stored, inherited, and returned.

It should function as:
- an orchestration template
- a prompt-generation template
- a milestone sequencing template
- a runtime handoff template
- a reusable in-repo workflow reference for future HydrOS milestone generation

This schema is intended to live in the repository so future milestone prompts can explicitly reference it.

## Important framing

This schema should be written as a **reference-first orchestration scaffold**, not as a rigid mandatory dependency.

It is being introduced because the M6–M8 orchestration pattern proved high-value, repeatable, and governance-safe.

It may be consulted and used throughout future milestone work when it improves:
- prompt generation
- artifact sequencing
- runtime handoff clarity
- return formatting
- inheritance structure
- workflow consistency

However, it should **not** be written as if every future artifact must mechanically depend on it in all cases.

The intention is:
- store it in the repo
- use it as a reusable orchestration reference
- test it in practice during M9 and later milestones
- promote it to stronger dependency status only if it proves quality through repeated use

This test-first status should be stated explicitly in the schema.

## Recursive distillation requirement

This schema must be created through **recursive orchestration distillation**.

That means it should be derived not only from abstract governance goals, but from the actual historical milestone prompt lineage already stored in the repository.

Use prior M6–M8 orchestration artifacts as evidence of the proven orchestration pattern this schema is formalizing.

The schema must therefore be written with enough structural depth that it can be used to generate future HydrOS orchestration prompts of the same general class as the prompts that created it.

That includes prompts for:
- refinement
- research
- source traceability
- execution documents
- implementation plans
- execution
- runtime updates
- closeout
- achievements
- future orchestration refinements

The schema should therefore function not only as a descriptive reference, but as a **recursively applicable prompt-generation framework**.

However, this recursive capability must remain governed:
- it should support reuse and regeneration of the orchestration pattern
- it should not imply uncontrolled or automatic self-replication
- it should be applied intentionally when it improves milestone quality, sequencing, or workflow efficiency

The schema should explicitly acknowledge that it was distilled from the M6–M8 orchestration pattern and is intended to be capable of recreating that class of structured milestone prompt chain in future work.

## Historical prompt-lineage grounding

Use the actual stored orchestration prompt/output lineage from prior milestones as a reference base.

At minimum, infer and formalize patterns such as:
- alternating prompt/output numbering
- standard file path behavior
- insight generation timing
- research-to-execution sequencing
- runtime-handoff handling
- closeout vs achievements separation
- inheritance patterns across iterations
- return formatting conventions
- when manual runtime became necessary
- how orchestration refinement improved over time

The schema should be presented as the formalization of a **proven historical pattern**, not as a hypothetical workflow design.

## Governing relationship

This schema does **not** replace:
`docs/orchestration/governance/HydrOS_x_AlignFlow_Orchestration_Contract_v3.1.md`

Instead:
- **Contract v3.1** governs reasoning, standards, truthfulness, insight discipline, and milestone philosophy
- **Execution Schema v1** governs prompt structure, artifact sequencing, storage, runtime class behavior, inheritance structure, and return formatting

The schema must explicitly explain this relationship.

## Required contents of HydrOS_Orchestrator_Execution_Schema_v1.md

### 1. Identity and purpose
State what this schema is and why it exists.

### 2. Governing relationship to Contract v3.1
Explain how this schema works together with the existing governing contract.

### 3. Reference-first usage status
State explicitly that this schema is:
- intended for iterative use throughout milestone work
- available as an in-repo orchestration reference
- designed to improve speed, consistency, and sequencing quality
- not yet a rigid mandatory dependency until validated through repeated practice

### 4. Recursive self-application role
Explain that the schema is intended to support generation of future prompts and artifact chains of the same general class as those that created it, while remaining governed and intentionally applied.

### 5. Standard artifact classes
Define the main HydrOS orchestration artifact types, such as:
- refinement prompt
- refinement output
- research brief prompt
- research brief
- sources map prompt
- sources map
- execution document prompt
- execution document
- implementation plan prompt
- implementation plan
- execution prompt
- execution report
- runtime handoff / runtime update
- closeout prompt
- closeout record
- achievements prompt
- achievements document
- insight artifact

For each class, briefly state its purpose.

### 6. Standard phase sequencing
Define the standard milestone progression pattern and explain that not every milestone must use every phase, but the sequence is the default template.

### 7. Artifact identity schema
Define the required identity fields for orchestration artifacts, including:
- milestone
- iteration
- phase
- prompt path
- output path
- insight path
- governing contract path
- governing schema path
- status
- runtime class

### 8. Runtime class system
Define the standard runtime classes:

- `fully_autonomous`
- `autonomous_until_manual_runtime`
- `manual_runtime_required`
- `post_runtime_update_required`

Explain what each means and how each should affect planning, execution prompts, execution reports, and handoffs.

### 9. Inheritance input structure
Define how each artifact should specify the prior files, insights, and repo artifacts it inherits from.

### 10. Truth-status / validity-status structure
Define a standard truth-status framing for milestone work, such as:
- source-grounded
- internal architecture decision
- implemented
- runtime-verified
- confounded / unresolved

Explain how this should be surfaced across major milestone documents.

### 11. Visual validity bridge
State explicitly that HydrOS milestones M6 and beyond should be evaluated not only for backend realism or control performance, but also for whether they improve the future visual simulation console's **visual validity**.

Define visual validity as the degree to which future rendered telemetry, truth-vs-sensor comparisons, policy-visible observations, and action traces are scientifically grounded, interpretable, and not misleading.

Require future milestone refinement to assess:
- whether the milestone improves future truth-vs-sensor visualization fidelity
- whether it improves the scientific meaning of displayed telemetry
- whether it reduces confounds that would make a future visual console misleading
- whether it strengthens the legitimacy of future policy-view and action-trace rendering

### 12. Confound-priority structure
Define that refinement outputs should rank inherited confounds as:
- primary
- secondary
- tertiary

And identify which confound the next milestone should reduce first.

### 13. System-model change requirement
Define that major refinement and closeout artifacts should state:
- what changed in the system model
- what assumption was corrected
- what future milestones must no longer assume

### 14. Manual-runtime handoff pattern
Define the standard pattern for milestones that require a user-run command, including:
- exact commands
- expected outputs
- what the user should paste back
- what report is updated after runtime
- what counts as success/failure

### 15. Standard return structure
Define the standard return pattern for orchestration prompts:
- prompt path
- output path
- insight path
- short summary
- next artifact

### 16. Storage discipline
Define where artifacts should live in the repo and reinforce that markdown artifacts should be printed in chat after creation per Contract v3.1.

### 17. M6–M8 proven pattern note
State that this schema is derived from the successfully executed M6–M8 orchestration pattern and is being formalized because that pattern proved high-value, repeatable, and governance-safe.

### 18. Validation-through-use note
State explicitly that the schema is expected to be tested during M9 and later milestones, and that its long-term role should be determined by whether it improves actual orchestration quality in practice.

## Style requirements

- concise but complete
- structured
- reusable
- written like a real operating reference
- not milestone-specific beyond the M6–M8 derivation note
- clear that it is a strong reference, not yet a rigid law
- clear that it is recursively generative but governed

## Required return

Return:
1. `docs/orchestration/governance/HydrOS_Orchestrator_Execution_Schema_v1_prompt.md`
2. `docs/orchestration/governance/HydrOS_Orchestrator_Execution_Schema_v1.md`
3. the insight file path created under `docs/orchestration/insights/Governance/`
4. short summary of what this schema adds beyond Contract v3.1
5. confirmation that this schema is now available as a reusable orchestration reference for future milestone prompt generation
