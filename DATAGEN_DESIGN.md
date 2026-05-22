# Datagen Design (Working Doc)

Working document for the datagen refactor. Not authoritative. Edit freely.

## Why

Today, adding a new dataset (e.g. sharegpt4video) means touching every datagen
that wants to use it: synthetic single-turn, shared-prefix, conversation
replay, OTel trace replay. That couples "where prompts come from" to "how
prompts are arranged into requests", which makes every new dataset cost N
times what it should.

The goal of the refactor is to split datagen into orthogonal axes so each new
dataset is a one-place change, and so that benchmarking concerns (sharing,
multi-turn structure, multimodality, arrival timing) compose freely.

## Axes

| Axis | Question | Examples |
| --- | --- | --- |
| **Sources** | Where does prompt content come from? Configured as a **list** of one or more sources, optionally with disjoint partitions. Referenced by index. | Synthetic, ShareGPT, OTel trace |
| **Composition** | How are prompts arranged into requests, and what (if anything) is shared across them? | Single-turn, multi-turn, future DAG; sharing slots live here |
| **Delivery** | When and how fast are requests dispatched? | Poisson, constant rate, follow trace timing |

Each axis is configured independently, with some legal-pairing rules
(multi-turn requires chat API; OTel trace replay bundles source + composition
+ delivery in the recording itself).

Sharing is not its own axis. It is exposed as **typed slots on each
composition** (e.g. `single_turn.shared_prefix`, `multi_turn.shared_first_turns`).
This keeps which sharing shapes are valid for which composition expressible
in the schema rather than checked at runtime.

## Core primitives

### Sources

Defined under `inference_perf/config/datagen/source/`. The top-level
`sources:` field accepts either a single source instance or a list of
source instances. The list form is needed only when more than one source
is in play; single-source configs (the common case) use the inline form
and drop the list nesting. Each entry is one item from the source
discriminated union:

- **Synthetic**: `MockSource`, `SyntheticSource`, `RandomSource`.
- **Dataset**: HuggingFace-loaded corpora (`ShareGPTSource`, `CNNDailyMailSource`,
  `InfinityInstructSource`, `BillsumConversationsSource`, future
  `ShareGPT4VideoSource`).
- **Recorded**: `OTelTraceReplaySource`.

Every source carries two optional common fields, declared *inside* the
source-kind value:

- `max_distinct_items: <int>` — caps the size of the source's content space.
  On synthetic/random sources this defines the otherwise-unbounded space.
  On HF/recorded sources this caps below the intrinsic size (lightweight
  subsetting, also useful for smoke tests). If omitted on a bounded source,
  the intrinsic size is used. If omitted on an unbounded source, the space
  is treated as unbounded and fractional partitions become invalid.
- `partitions: <dict>` — declares named, guaranteed-disjoint subsets of the
  source's content space. Partition values are polymorphic by type:
  `float` in `(0, 1]` is a fraction, `int` is a count, `list[int]` is an
  index list. Mixing types in a single `partitions:` block is invalid.
  Fractional partitions must sum to `1.0`.

```yaml
# Single-source: inline form (no list nesting)
sources:
  synthetic:
    input_distribution: { mean: 1024, std: 128 }
    max_distinct_items: 1000

# Multi-source: list form
sources:
  - synthetic:
      input_distribution: { mean: 1024, std: 128 }
      max_distinct_items: 1000
  - sharegpt4video:
      max_distinct_items: 500
      partitions:                        # disjoint slices of those 500
        warm: 0.5                        # float -> fraction
        cold: 0.5
```

Sharing slots reference sources by index via a `from:` field
(`from: 1.warm` for a partition of source 1, `from: 1` for the whole
source). The inline single-source form is normalized to a one-element
list internally, so `from: 0` is valid in either form. If `from:` is
omitted and exactly one source is declared, the slot defaults to that
source.

Validation: a slot's `num_items` must be `<=` the size of its referenced
source (or partition). Config-load fails otherwise.

### Composition

Defined under `inference_perf/config/datagen/composition/`. Discriminated
union of graph-shape strategies:

- `SingleTurnComposition`: one request per prompt, no dependencies. Sharing
  slots: `shared_prefix`, `shared_media`.
- `MultiTurnComposition`: chains of turns with intra-chain dependencies.
  Sharing slots: `shared_system`, `shared_first_turns`, `shared_media`.
- `TraceReplayComposition`: structure comes from the recording. No sharing
  slots today (organic sharing already lives in the recorded content).
- Future: agent / branching / DAG.

`SharedPrefixComposition` from the old design is gone; a single-turn workload
with a shared prefix is now `single_turn` with its `shared_prefix:` slot set.

### Sharing (new primitive)

`Shared<T>` is a small reusable primitive that declares "there is a finite
pool of T-typed content reused across the request population." It is
structurally similar to `Distribution`: small, embedded wherever sharing
makes sense, with explicit fields rather than hidden config layers.

Shape:

```yaml
# Common shape, set as the value of a typed sharing slot on a composition
<slot_name>:
  from:         <int | "int.partition">                  # source index, optionally with partition; defaults to source 0
  num_items:    <int>          # number of distinct shared items
  reuses_per_item: <int>          # how many consumers reuse each item
  content:      <T>            # T-typed pool item shape (Prompt, Conversation, Section)
  name:         <optional string>                       # power-user override for the reporting label;
                                                        # defaults to the slot name
```

The available slots per composition, what `T` is for each, and the
sharing unit the slot implies:

| Slot | Found on | `T` | Shared across | What gets pooled |
| --- | --- | --- | --- | --- |
| `shared_prefix` | `single_turn` | `Prompt` | requests | a prompt-shaped chunk that prepends every request in the group |
| `shared_media` | `single_turn`, `multi_turn` | `Section` | requests | one media (or text) section reused across requests |
| `shared_system` | `multi_turn` | `Prompt` | conversations | the system message body of conversations in a group |
| `shared_first_turns` | `multi_turn` | `Conversation` | conversations | the first K turns of conversations in a group |

A single composition may set multiple slots; "stacking" two sharing pools
on the same composition is just "set both slots." Each set slot can be
named so the run report breaks out cache-hit stats per slot.

### Section (new primitive)

`Sharing` operates on sections. A prompt is a sequence of typed sections.
Modality is key-as-discriminator and is treated uniformly: no modality
(text, image, audio, video) is privileged structurally.

```yaml
# Single section, modality determined by key
text:  { length: <int | Distribution> }
image: { dimensions: [<w>, <h>], resolution: <res> }
audio: { duration: <seconds> }
video: { duration: <seconds>, fps: <int> }

# A multi-section content block is a list. Order in the list = order in
# the prompt, unless a section opts into floating placement.
content:
  - text: { length: 200 }
  - image: { dimensions: [512, 512] }
  - text: { length: 100 }
```

For text-only prompts the primitive is invisible (a string is one text
section). Sections become load-bearing for multimodal and for the `content`
field of `Sharing`.

**Common fields on every section** (orthogonal to modality):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `count` | `int \| Distribution` | `1` | How many of this section. Distributions yield variable count. |
| `position` | absent \| `float \| Distribution` | absent | When absent, the section sits in **list order** (fixed slot). When set, the section is **floating**: inserted into the assembled-from-fixed-sections "spine" at the specified fractional position (`0.0` = start, `1.0` = end, `0.5` = middle). Distributions yield variable position per draw. |
| `source` | `int \| "int.partition"` | source 0 (or the only source) | Which top-level source supplies the content. |

**Assembly rule:**

1. Lay out the fixed sections (those without `position:`) in list order. This is the implicit spine.
2. Thread floating sections (those with `position:`) into the spine at their resolved positions.

This means index 0 of the spine is where the first fixed section starts;
position 0.0 inserts before it. Floating sections are the only ones whose
list slot in YAML doesn't determine their location in the rendered prompt.

### Delivery

Defined under `inference_perf/config/datagen/delivery/`. Owns arrival timing
and dispatch policy. Today: `arrival` schedules (Poisson, constant, trace).
Future: concurrency caps, sharing-aware clustering.

## Type model

The primitives compose as a small generic algebra. The YAML config is the
surface; the type model is the runtime / Pydantic class shape underneath.

```
type Section      = TextSection | ImageSection | AudioSection | VideoSection
type Prompt       = List<Section>
type ChatTurn     = { role: user | assistant | system, content: Prompt }
type Conversation = List<ChatTurn>

# Source list (the top-level `sources:` field)
type Source       = Synthetic | ShareGPT | CnnDailyMail | ... | OTelTraceReplay
type Partition    = float | int | List<int>     # fraction | count | indices
type SourceEntry  = Source & {                   # common fields live inside the source-kind value
  max_distinct_items?: int                       # cap on content-space size
  partitions?:         Dict<string, Partition>
}
type SourcesField = SourceEntry | List<SourceEntry>   # inline for single-source, list for multi-source

# The sharing primitive: a typed pool drawing from a source
type Shared<T> = {
  from?:           int | string          # source index, or "<index>.<partition>"; defaults to source 0 when only one
  num_items:       int                   # number of distinct shared items
  reuses_per_item: int                   # how many consumers reuse each item
  content:         T                     # T-typed pool item shape
  name?:           string                # reporting label; defaults to slot name
}                                        # sharing unit (request vs conversation) is implied by the slot

# Compositions expose typed sharing slots as optional fields
type SingleTurn = {
  ...config
  shared_prefix?: Shared<Prompt>
  shared_media?:  Shared<Section>
}

type MultiTurn = {
  ...config
  num_conversations:      int
  turns_per_conversation: int | Distribution
  shared_system?:   Shared<Prompt>
  shared_first_turns?:  Shared<Conversation>
  shared_media?:    Shared<Section>
}

type TraceReplay = { ...config }         # bundles its own shape from the recording
```

`Shared<T>` is the only generic. The position (prefix vs system message vs
opening turns vs anywhere) is encoded by **which slot** the `Shared<T>` is
assigned to on the composition, not by a separate `placement` field. The
slot name and the `T` it accepts are both fixed by the composition.

Users don't write these types in YAML; the YAML's key-as-discriminator form
is the surface. The types matter for the Pydantic class graph, IDE
autocomplete on the Python API, and keeping invalid combinations
(e.g. `shared_first_turns` on `single_turn`) out at schema-validation time
rather than runtime.

## UX examples

The same six cases we walked through, expressed in the proposed config shape.

### a. single-turn with shared prefix

**Type:** `SingleTurn { shared_prefix: Shared<Prompt> }`

```yaml
data:
  sources:
    synthetic:
      input_distribution: { mean: 1024, std: 128 }
  composition:
    single_turn:
      shared_prefix:                    # Shared<Prompt>
        num_items: 10
        reuses_per_item: 100
        content:
          text: { length: 500 }
        # `from:` omitted -> uses the only source
```

### b. single-turn with shared multimodal prefix

**Type:** `SingleTurn { shared_prefix: Shared<Prompt> }` (Prompt = List<Section>, so a multimodal prefix is just a multi-section Prompt)

```yaml
data:
  sources:
    sharegpt4video: {}
  composition:
    single_turn:
      shared_prefix:                    # Shared<Prompt>
        num_items: 5
        reuses_per_item: 20
        content:
          - text:  { length: 200 }
          - image: { dimensions: [512, 512] }
          - text:  { length: 100 }
```

### c. multi-turn with shared system prompt across conversations

**Type:** `MultiTurn { shared_system: Shared<Prompt> }`

```yaml
data:
  sources:
    synthetic: {}
  composition:
    multi_turn:
      num_conversations: 200
      turns_per_conversation: 8
      shared_system:                    # Shared<Prompt>
        num_items: 10
        reuses_per_item: 20
        content:
          text: { length: 800 }
```

### d. multi-turn with shared opening turns

**Type:** `MultiTurn { shared_first_turns: Shared<Conversation> }`

```yaml
data:
  sources:
    synthetic: {}
  composition:
    multi_turn:
      num_conversations: 200
      turns_per_conversation: 10
      shared_first_turns:                   # Shared<Conversation> (i.e. List<ChatTurn>)
        num_items: 5
        reuses_per_item: 40
        content:
          - user:      { length: 300 }
          - assistant: { length: 400 }
          - user:      { length: 200 }
```

### e. both (c) and (d), with named blocks for per-block cache reporting

**Type:** `MultiTurn { shared_system: Shared<Prompt>, shared_first_turns: Shared<Conversation> }`

```yaml
data:
  sources:
    synthetic: {}
  composition:
    multi_turn:
      num_conversations: 1000
      turns_per_conversation: 10
      shared_system:                    # Shared<Prompt>
        num_items: 4
        reuses_per_item: 50
        content:
          text: { length: 1200 }
      shared_first_turns:                   # Shared<Conversation>
        num_items: 4
        reuses_per_item: 50
        content:
          - user:      { length: 300 }
          - assistant: { length: 400 }
```
Stacking two pools is "set two slots." No nested generics, no list ordering
to reason about.

### f. OTel trace replay (today)

**Type:** `TraceReplay` (no sharing slots; host shape and timing come from the recording)

```yaml
data:
  sources:
    otel_trace_replay:
      trace_directory: /path/to/traces
      model_mapping: { ... }
      include_errors: false
  composition:
    trace_replay: {}                    # structure comes from the source
  delivery:
    follow_trace_timing: true
```

If a team eventually wants to *inject* additional sharing on top of a
recording (e.g. "force every replayed prompt to also share a 500-token
synthetic prefix to stress prefix cache"), `trace_replay` would grow a
`shared_prefix` slot. The trace replay engine would inject the shared
section at fire time. This is a future capability, not current.

### g. team ask: shared image pool, not in prefix

**Type:** `SingleTurn { shared_media: Shared<Section> }`

```yaml
data:
  sources:
    sharegpt4video: {}
  composition:
    single_turn:
      shared_media:                     # Shared<Section>
        name: image_pool
        num_items: 100
        reuses_per_item: 2               # each image used in exactly 2 requests
        content:
          image: {}
```
The image content itself is drawn from the source; the `shared_media` slot
enforces the reuse pattern. Having `shared_media` as a separate slot from
`shared_prefix` is what makes "shared but not at the front" expressible
without overloading the prefix concept.

### h. team ask: mutually exclusive media pools (cache warmup + measure)

**Type:** Two `SingleTurn { shared_media: Shared<Section> }` workloads (across stages or runs) drawing from partitioned subsets of the same source.

```yaml
data:
  sources:
    sharegpt4video:                     # single-source inline form
      partitions:                       # guaranteed disjoint subsets of source 0
        warm: 0.5
        cold: 0.5

  # stage 1: warm the cache against the `warm` partition
  composition:
    single_turn:
      shared_media:
        from: 0.warm                    # source 0, partition `warm`
        num_items: 50
        reuses_per_item: 4
        content:
          image: {}

  # stage 2 (next config or stage): measure against `cold` to verify no overlap
  # composition:
  #   single_turn:
  #     shared_media:
  #       from: 0.cold
  #       num_items: 50
  #       reuses_per_item: 4
  #       content: { image: {} }
```
`fraction` declares disjoint slices of the source's content space. Two
slots referencing `0.warm` and `0.cold` are guaranteed never to share an
item, which the team needed to keep cache-effect measurements clean.
`count: <int>` and `indices: <list>` are other partition shapes when
fractional splits aren't precise enough.

### i. issue #506: images woven into a prefix, count and position can vary

**Type:** `SingleTurn { shared_prefix: Shared<Prompt> }` using floating image sections.

Today's `shared_prefix` config exposes an `insertion_point: <float>` knob
intended to control where images land within the shared content. Issue #506
reports that `insertion_point: 0.5` doesn't actually push the image to the
middle. In the new design the `insertion_point` knob goes away as a special
case: any section can opt into floating placement by setting `position:` on
itself, and the assembled fixed sections form the implicit spine that
floating sections are inserted into.

**Deterministic version of the reporter's case** (2 images positioned at
50% of the assembled text spine):

```yaml
data:
  sources:
    sharegpt4video: {}
  composition:
    single_turn:
      shared_prefix:                    # Shared<Prompt>
        num_items: 20
        reuses_per_item: 1
        content:
          - text:  { length: 200 }      # fixed, list slot 0 of the spine
          - image:                      # floating: ignores list slot
              count:    2
              position: 0.5             # mid-spine
              resolution: 720p
              source: 0
          - text:  { length: 100 }      # fixed, list slot 1 of the spine
```

**Non-deterministic version** (1-3 images, position varies per request):

```yaml
content:
  - text: { length: 200 }
  - text: { length: 100 }
  - image:
      count:    { uniform: [1, 3] }
      position: { uniform: [0.0, 1.0] }
      resolution: 720p
      source: 0
```

Three design wins this surfaces:

- **No `insertion_point` special case.** `position:` lives on every section
  uniformly, replacing the per-modality knob.
- **Multiple sections at distinct positions** are expressible naturally
  (one image at 0.3, another at 0.7) instead of being forced into a single
  per-prefix number.
- **Modality is not privileged.** A video-only or image-only prompt is just
  one section in the list. Text is not the spine by default; the *fixed
  sections* are, whatever modality they happen to be.

## Open questions

- **Slot extensibility.** New sharing shapes (e.g. `shared_tool_results` for
  agentic workloads) require adding a new typed slot to the relevant
  composition. That is a schema change, not a config-only change. Tradeoff
  is accepted in exchange for schema-level validity checking.
- **Source/sharing compatibility.** Some sources (Random) have no semantic
  content. Sharing on a Random source is probably meaningless; should it
  validate-error or silently degrade?
- **Sharing on Recorded sources.** Layering synthetic sharing on top of an
  OTel replay would mean adding sharing slots to `trace_replay`. Defer until
  a concrete team asks.
- **Delivery interaction.** A shared prefix only hits server-side cache if
  the requests using it are temporally clustered. Should the runtime imply
  a clustering policy in delivery when a sharing slot is set, or should
  delivery stay sharing-agnostic and the user size their arrival schedule
  accordingly?
- **Partition shapes per source kind.** Fractional partitions require a
  finite content space, which `max_distinct_items` supplies for
  synthetic/random.
  Recorded traces with implicit dependencies (e.g. OTel chains that span
  across replays) still can't be cleanly partitioned even with a row cap;
  validate-error on `partitions:` for those source kinds until a concrete
  team asks.
- **Cross-composition sharing.** Can two compositions (in a future mixed
  workload) share a pool? Probably not worth supporting; the slot-on-
  composition design implicitly forbids it. Partitions of a single source
  cover most of the same use cases.

## Out of scope (for this refactor)

- **Nested sharing.** Earlier drafts explored `Shared<T>` nested inside
  another `Shared<T>` (e.g. a shared opening pool whose turns each carry
  their own shared prefix). No team has asked for this and real workloads
  push shared instructions to the system message instead. `ChatTurn` does
  not carry its own `shared_prefix` field, and any inner-pool functionality
  is deferred indefinitely. If a team eventually needs sharing patterns
  that the slot-per-composition model cannot express, the **custom
  dataset** escape hatch (below) is the preferred path rather than
  growing more nesting.
- **Custom dataset escape hatch.** Long term, allowing users to load a
  pre-built corpus (their own JSON/HF dataset/recording) is a cleaner
  generality escape than expanding the sharing primitive further. Sharing
  becomes an opt-in modifier on top of whatever content the custom source
  provides. Not in this refactor, but the source axis is already shaped to
  accommodate it (just another entry in the source discriminated union).
- **Runtime unification under a Prompt-DAG.** The mental model of "every
  composition is a DAG of Prompts with dependencies and the delivery layer
  walks the DAG" is useful for reasoning, but the runtime today is four
  separate engines. Unifying them is a much larger rewrite and not blocking
  for the config shape.
- **New compositions.** Agent loops, branching flows, etc., come later. The
  current axes have to be expressive enough to *receive* them without
  another refactor.
- **Section primitive on the runtime side.** Sections can be config-shape
  only at first; runtime can still flatten to strings until multimodal
  forces the issue.

## Status

- Source axis: implemented on the `data-source` branch as a single
  `data.source:` field. Promoting to a named `data.sources:` registry with
  partitions is design only, not yet in code.
- Composition axis: implemented (parallel surface, sibling-list sharing
  shape), not yet wired into `DataConfig`. Slot-based sharing from this
  doc is not yet reflected in code.
- `Shared<T>` primitive (with `from:`, `num_items`, partitions): design only,
  not implemented.
- Section primitive: design only, not implemented.
- Delivery axis: implemented (parallel surface), not yet wired in.

The plan is to land the source extraction first, keep composition / delivery
on the integration branch, and iterate on Sharing / Section in this document
before touching code.
