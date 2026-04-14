# HydrOS x AlignFlow Orchestration Contract v3.1 — Bottom-Append Refinement Prompt

First, write this prompt content to:
`docs/orchestration/governance/HydrOS_x_AlignFlow_Orchestration_Contract_v3.1_refinement_prompt.md`

If the folder path does not yet exist, create it first.

Then execute the task exactly as written below.

Operate under HydrOS x AlignFlow Orchestration Contract v3 at:
`docs/orchestration/governance/HydrOS_x_AlignFlow_Orchestration_Contract_v3.md`

## Objective

Refine Contract v3 into Contract v3.1 by appending a new required chat-echo rule to the **bottom of the contract**.

I am not asking for a broad rewrite of the contract.
Do not restructure existing sections unless a tiny consistency edit is truly required.

## Required file outputs

Write the updated governing contract to:
`docs/orchestration/governance/HydrOS_x_AlignFlow_Orchestration_Contract_v3.1.md`

Also create one insight file for this refinement pass at:
`docs/orchestration/insights/Governance/HydrOS_v3.1_insight_chat_echo_rule_improves_portability.md`

If the insights folder path does not yet exist, create it first.

## Required refinement behavior

Use Contract v3 as the base document.

Create Contract v3.1 by:
1. preserving the existing contract content
2. appending the new rule block **at the bottom of the contract**
3. not removing or rewriting prior sections unless absolutely necessary for consistency

## Exact rule to append at the bottom of Contract v3.1

Append this as a new final section at the bottom:

### Required Chat Echo for New Markdown Files

Whenever a new markdown file is created as part of HydrOS orchestration, also print the full markdown contents in chat output after writing the file.

This printed markdown must be:
- complete
- copy-pastable
- clearly labeled with its intended repo path
- materially identical to the stored file

This rule applies to markdown artifacts such as:
- refinement prompts
- research briefs
- source maps
- execution documents
- implementation plans
- execution reports
- closeout records
- achievements documents
- insight files
- other milestone markdown artifacts

This rule does **not** automatically apply to non-markdown files such as:
- `.py`
- `.json`
- `.pkl`
- `.zip`
- other binary or code artifacts

unless explicitly requested by the user.

### Purpose of the rule

The chat echo requirement exists to:
- preserve a portable backup of each markdown artifact
- make rapid inspection easier
- support manual copy/paste recovery
- reduce risk from path mistakes, overwrite mistakes, or storage-only dependence

## Required output behavior

When generating Contract v3.1:
1. store the new file in the repo
2. append the new chat-echo section at the bottom of the contract
3. print the full markdown of Contract v3.1 in chat
4. create and store the required insight file
5. print the full markdown of the insight file in chat

## Required insight contents

The insight file should capture:
- context of the refinement
- what was learned
- why the bottom-append rule matters
- workflow implications
- recommended carry-forward action

## Return

Return:
1. `docs/orchestration/governance/HydrOS_x_AlignFlow_Orchestration_Contract_v3.1_refinement_prompt.md`
2. `docs/orchestration/governance/HydrOS_x_AlignFlow_Orchestration_Contract_v3.1.md`
3. `docs/orchestration/insights/Governance/HydrOS_v3.1_insight_chat_echo_rule_improves_portability.md`
4. short summary of what changed from v3 to v3.1
5. confirmation that the new rule was appended at the bottom of the contract
6. confirmation that Contract v3.1 is now the recommended governing contract going forward
