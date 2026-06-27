# vgather/vscatter Discussion and Compiler Inference Proposal (PTO mi ISA)

Status: discussion draft

## 1. Background and Problem Statement

In PTO mi ISA, gather/scatter has the following current shape:

- Gather has two ISA forms:
  - `vgather2`
  - `vgather2_bc`
- Scatter has one ISA form:
  - `vscatter`

The key practical issue is how index width and element width interact with reachable UB tile range, especially when tile upper bound (UB size/range) is not known early or may exceed narrow-index coverage.

## 2. Current Behavior Summary

### 2.1 `vgather2` and `vscatter`

For `vgather2`/`vscatter`, index dtype size and payload element size are matched.

- `b16` payload -> `b16`-sized index
- `b32` payload -> `b32`-sized index

Implication:

- With `b16` index, addressable element-index space is limited by 16-bit index range.
- Interpreting index as element index (base unit = element size), max covered byte span is:

```
range_bytes = 2^index_bits * element_size_bytes
```

For `b16` payload/index:

```
range_bytes = 2^16 * 2B = 128KB
```

This becomes a constraint when UB tile span (or effective gather/scatter working set span) can be larger than this range.

### 2.2 `vgather2_bc`

`vgather2_bc` has different constraints:

- offset is 32-bit
- dtype support is only `b16`/`b32`

Because it offers 32-bit offset, it is the practical escape hatch when `b16`-matched indexing is too narrow for a large tile/range.

## 3. Why This Matters for Compiler Lowering

For many kernels, tile size/range may be:

- statically known and small enough
- statically unknown at early passes
- data-dependent after scheduling/fusion/layout decisions

If the compiler keeps a fixed gather/scatter form too early, it can either:

- over-constrain legality (false rejection), or
- require late, expensive rewrites, or
- generate suboptimal fallback code.

So this should be solved by an explicit inference policy in lowering.

## 4. Proposed Compiler Inference Policy

### 4.1 Inputs to the Decision

At gather/scatter lowering point, derive:

- payload dtype (`b16` or `b32` in this scope)
- operation kind (`gather` or `scatter`)
- required index domain upper bound in elements: `idx_max_required`
- equivalent byte span:

```
required_bytes = (idx_max_required + 1) * element_size_bytes
```

### 4.2 Instruction Selection Rules

1. Gather case:
- If payload is `b16`:
  - If required range is provably within `b16` matched-index coverage, choose `vgather2` with `b16` index.
  - Else choose `vgather2_bc` with 32-bit offset.
- If payload is `b32`:
  - Treat `vgather2` and `vgather2_bc` as functionally equivalent for range coverage.
  - Select one canonical opcode by backend policy (or keep either by codegen preference), not by range-size legality.

2. Scatter case:
- Only `vscatter` exists.
- Use matched index width (`b16`/`b32`) according to payload and legality/range constraints.
- If range exceeds `b16` matched-index capacity, promote to `b32` index path.

### 4.3 Unknown-Range Conservative Rule

When upper bound is unknown at compile-time:

- Use conservative-safe path:
  - gather (`b16` payload): prefer `vgather2_bc`,
  - gather (`b32` payload): either `vgather2` or `vgather2_bc` (backend canonical choice),
  - scatter: promote to `b32` index for `vscatter`.

This avoids late illegalization for big-tile cases.

### 4.4 Cost-Aware Refinement (Optional)

If profile/cost model exists, add a threshold rule:

- For known-small ranges, prefer narrow-index form (less conversion/shuffle overhead).
- For uncertain/large ranges, prefer wide-index form to avoid guard and rewrite overhead.

## 5. Required Index Adaptation for `b16` Data with Wide Range

For `b16` payload when required address domain exceeds narrow-index coverage:

- select wide (`b32`) index representation in IR/lowering,
- use select/shift/pack sequence as needed to form the proper half/portion semantics expected by backend path,
- lower to `vgather2_bc` for gather when applicable,
- lower to `vscatter` with `b32` index for scatter.

This matches the practical need to access higher/lower halves when narrow indexing cannot directly represent full range.

## 6. Suggested IR-Level Contract

Introduce an explicit logical op contract that separates semantic indexing from concrete ISA opcode:

- logical gather/scatter carries:
  - payload dtype
  - index semantic domain
  - optional proven upper bound metadata
- ISA opcode (`vgather2`, `vgather2_bc`, `vscatter`) is selected in target-lowering pass.

This keeps early/mid passes target-agnostic and enables stable late selection.

## 7. Verification and Diagnostics Recommendations

Add verifier checks and diagnostics:

- Emit clear error if selected narrow-index path cannot cover required range.
- Emit info/warn note when compiler auto-promotes index width (`b16 -> b32`).
- Provide debug trace for opcode selection decision:
  - required range,
  - proven bound source,
  - chosen opcode/index width,
  - fallback reason.

## 8. Example Decision Matrix

| Op | Payload dtype | Range known small | Range unknown/large | Recommended selection |
|---|---|---|---|---|
| gather | b16 | yes | no | `vgather2` + b16 index |
| gather | b16 | no | yes | `vgather2_bc` + 32-bit offset |
| gather | b32 | yes | no | `vgather2` or `vgather2_bc` (backend canonical choice) |
| gather | b32 | no | yes | `vgather2` or `vgather2_bc` (backend canonical choice) |
| scatter | b16 | yes | no | `vscatter` + b16 index |
| scatter | b16 | no | yes | `vscatter` + b32 index |
| scatter | b32 | yes/no | yes/no | `vscatter` + b32 index |

## 9. Recommended Next Step for PTO-Gym Compiler

Implement automatic inference as default behavior:

- Compiler should infer gather/scatter opcode and index width from:
  - payload dtype,
  - proven/estimated index range,
  - legality and target profile.
- Users should not manually pick `vgather2` vs `vgather2_bc` in common cases.
- Manual override can remain as expert option for micro-tuning.

This gives predictable correctness first, with room for profile-guided optimization later.

## 10. Applied Example: A5 TopK (Radix-Select) in PTO-ISA

The attached A5 TopK kernels are a good concrete case for this policy:

- `xson/pto-isa/kernels/manual/a5/topk/draft.cpp`
- `xson/pto-isa/kernels/manual/a5/topk_ub/draft.cpp`

Observed pattern from those docs/implementations:

- Winner selection phases use gather-style reads from histogram/cumulative arrays.
- Final selection phase uses compare-gather to collect GT/EQ indices.
- In `topk_ub`, EQ gather path may internally use vscatter optimization (see `PTO_A5_TGATHER_B16_EQ_USE_VSCATTER` note in README).

### 10.1 Why gather is natural in TopK phase logic

TopK radix-select repeatedly does:

- read `C[winner-1]` from cumulative histogram,
- read keys or index candidates that satisfy compare condition,
- append or compact selected indices.

These are primarily read-by-index behaviors, so gather-first lowering is natural for core selection logic.

### 10.2 Where scatter appears in TopK practice

Scatter is useful in the compaction/write stage when selected indices must be placed into output slots that are not contiguous at generation time.

Typical pattern:

1. Build predicate for selected lanes (GT or EQ).
2. Compute destination offsets (prefix/count based).
3. Scatter lane indices to destination positions.

So for TopK:

- selection is gather-dominant,
- output compaction can be scatter-assisted.

This exactly matches your question: yes, in TopK/radix you often need scatter when writing to different destinations; gather alone is ideal for indexed reads but not always ideal for variable-position writes.

## 11. Applied Example: Full Radix Reorder (Stable Bucket Pass)

For a classical radix-sort pass (not only TopK threshold selection), scatter is usually first-class.

Given keys and bucket ids:

1. Histogram each bucket count.
2. Exclusive-scan to get bucket base offsets.
3. For each element, compute `dst = base[bucket] + local_rank_in_bucket`.
4. Scatter key/index payload to `dst`.

That means:

- read side may use gather (optional),
- write side fundamentally needs scatter to place each element into bucketed output layout.

## 12. Practical Lowering Rules for These Two Workloads

### 12.1 TopK radix-select

- Winner/cumulative lookup: prefer gather.
- GT/EQ candidate extraction: prefer gather-style compare path.
- Segment compaction/output merge: allow scatter path when destination positions are dynamic.

### 12.2 Full radix-sort reorder

- Bucket placement phase: prefer scatter.
- If index/address range may exceed `b16` matched coverage, auto-promote to wide index:
  - gather: `vgather2_bc` where applicable,
  - scatter: `vscatter` with `b32` index.

## 13. PTO-ISA View (What to expose in compiler IR)

To avoid hardcoding one opcode too early, keep a logical op layer:

- `gather_read` (semantic indexed read)
- `scatter_write` (semantic indexed write)
- metadata: payload dtype, index bound info, stability requirement, compaction mode

Then at target lowering:

- choose `vgather2` vs `vgather2_bc` by range+dtype legality,
- choose `vscatter` index width (`b16`/`b32`) by the same policy,
- optionally use gather+concat or scatter-compaction depending on profile.

This lets TopK and full radix-sort share one consistent inference framework.
