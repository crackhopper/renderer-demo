# Notes Writing Style

## Purpose

Define the house style for human-facing documentation under `notes/`, especially the `概念` section.

This spec is not about code behavior. It is about how we explain the engine to human readers so that notes stay readable, consistent, and aligned with the project voice.

## Scope

This spec applies to:

- `notes/concepts/`
- `notes/tutorial/`
- user-facing explanatory sections inside `notes/subsystems/` when they link back to concepts

It does not replace technical specs under `openspec/specs/`, which remain normative for implementation behavior.

## Requirement: Concepts use an explanatory narrative, not a reference-manual voice

Concept documents SHALL read like guided explanations, not API reference pages or product manuals.

They SHALL answer, in natural reading order:

- what this system is in the current engine
- what problem it solves
- how it is typically used in the current codebase
- where its boundaries are
- how it connects to the implementation and deeper docs

Concept documents SHOULD introduce ideas in paragraphs first, and use lists only where lists materially improve readability.

Concept documents SHALL avoid template-like headings such as:

- `这个系统是什么`
- `它有什么用`
- `你通常怎么用它`
- `当前项目做到哪了`
- `底层现在怎么实现`

when used mechanically across every page.

Instead, headings SHOULD be phrased as meaningful reading cues that match the topic, for example:

- `从蓝图到实例`
- `从文件到运行时对象`
- `一条 pipeline 是怎样被确定的`
- `资源怎样成为场景里的对象`

## Requirement: Use "我们" voice, not second-person instruction voice

Human-facing notes SHALL use first-person plural narration (`我们`) as the default voice.

They SHALL NOT use second-person instructional voice as the primary narrative style, including patterns such as:

- `你会在什么场景接触它`
- `你通常怎么用它`
- `如果你想...`
- `你可以继续看...`

This rule exists to keep notes aligned with the existing tutorial voice: collaborative, explanatory, and engineering-oriented.

Short direct instructions are allowed inside code snippets, warnings, or migration notes when needed, but the surrounding prose SHALL remain in `我们` voice.

## Requirement: Concepts are capability-oriented, not category checklists

Concept pages SHALL organize around the reader's mental model of the engine, not around a checklist copied from planning notes.

This means:

- page titles SHOULD describe the underlying idea or flow, not merely repeat a category label
- sections SHOULD be ordered for understanding, not for requirement bookkeeping
- implementation status MAY be included, but it SHALL support understanding rather than dominate the page

For example, a good concept title explains a mechanism or role:

- `资产如何进入引擎`
- `材质如何决定一条渲染路径`
- `相机怎样进入一帧渲染`

A weaker title merely repeats a bucket name with no reader guidance.

## Requirement: Concepts explain current reality first

Concept pages SHALL describe the current engine as it exists today.

They MAY mention future or missing capability, but only after current behavior is made clear.

When a capability is incomplete or missing:

- the document SHALL clearly label it as partial or not yet implemented
- the document SHALL link to an active requirement when one exists
- if no active requirement exists and the capability is important to the concept map, a new requirement SHOULD be created before the concept claims the feature as planned

Concept pages SHALL NOT present future architecture as if it already exists.

## Requirement: Cross-links must preserve the concept-to-design ladder

Concept pages SHALL link readers downward into deeper implementation documents.

The usual ladder is:

1. concept page explains the role and usage
2. subsystem page explains current implementation shape
3. spec explains normative behavior

Concept pages SHOULD therefore end or conclude with references to the most relevant:

- `notes/subsystems/*.md`
- `openspec/specs/*/spec.md` when helpful
- `docs/requirements/*.md` for incomplete capability

## Requirement: Tutorials and concepts should share tone, not structure

`notes/tutorial/` is the reference tone for approachable documentation.

Concept pages SHOULD resemble tutorial prose in these ways:

- explain before enumerating
- prefer engineering narrative over abstract taxonomy
- keep paragraphs concrete and grounded in the current repo

Concept pages do NOT need to copy the tutorial table/step format. They only need to share the same voice and readability standard.

## Requirement: API details belong in examples or linked implementation docs

Concept pages MAY include short code examples, but SHALL NOT turn into symbol-by-symbol API references.

When a page needs to mention concrete runtime objects, it SHOULD:

- mention the key type names in prose
- include one compact example when it improves understanding
- link to subsystem docs or source files for deeper detail

This keeps concepts readable while still anchored in the real codebase.

## Requirement: Navigation labels may stay plain even when page titles are richer

Navigation labels in `notes/nav.yml` MAY use simple capability names such as:

- `资产系统`
- `材质系统`
- `渲染管线`

The page title itself SHOULD be allowed to be more expressive and explanatory.

This separation is intentional:

- nav labels optimize for scanning
- page titles optimize for understanding

## Acceptance Criteria

A concept page is compliant with this spec if:

- its primary narrative voice is `我们`, not `你`
- its title helps a reader understand the topic, not just classify it
- its sections read as a guided explanation of the current engine
- incomplete features are clearly marked and linked to requirements
- it points readers to deeper implementation docs rather than trying to become a full reference manual
