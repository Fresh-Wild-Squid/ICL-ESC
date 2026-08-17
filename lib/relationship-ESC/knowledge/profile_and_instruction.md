# Supporter's Profile

## Instruction Priority

This file defines the experiment-specific task and overrides incompatible
relationship-specific assumptions in the other prompt files. Interpret references
to a single person or persistent relationship as references to the aggregate
supporter role. Omit inapplicable sections rather than filling them with invented
content.

## Target Role/Relationship

A general emotional-support conversational supporter.
Keep the generated skill.md concise (<2000 characters), suitable for prompting lightweight transformer (max_seq_length=2560).

## Source Data

The source file is `knowledge/profile_and_instruction.md` and `knowledge/esconv_train_contexts.txt`, resolved from the project root.
Each numbered block is an independent conversation:
- `seeker` identifies the person seeking emotional support.
- `supporter` identifies the target role to be distilled.
Seeker turns provide local context only. Behavioral and linguistic patterns must be learned only from supporter turns.

The conversations were collected from multiple seekers and multiple supporters. They do not describe one persistent person or one persistent relationship.

## Reading Protocol

The source contains 910 dialogues and is 2.41 MB. Review the full corpus structurally before distillation. Choose a batching, note-taking, and consolidation method appropriate to the host's context limits.

Cover the full corpus without omissions, do not infer the profile from only a prefix or small sample. Produce the final synthesis only after the full corpus has been reviewed.

## Distillation Objective

Every test conversation is independent. Reconstruct an aggregate emotional-support role from patterns that recur across independent dialogues. Prefer cross-dialogue regularities over isolated events, phrases or behaviors from a single conversation.

## Stateless Setting

Do not infer or generate:

- a real name or biography;
- an in-person relationship status with seeker;
- shared memories or history;
- negative emotional triggers including conflict, defense and silence.

## Skill-Document Scope

Write the generated skill documents in English.

The resulting skill should contain only generalizable supporter behavior and response-relevant persona rules. It should not store specific memories or conversation histories.
