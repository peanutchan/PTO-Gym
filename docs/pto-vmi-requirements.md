# PTO Virtual micro Instruction (`pto.vmi`) — Requirements

- v0.1: Doc init. Philosophy, compile principles, rules, op surface, nVL
  register type (1D / ragged / 2D), and the open exploration space for compiler
  inference and instruction mapping.

**Status:** draft / RFC companion. Extends
[RFC-VPTO-Logical-Vector-ISA.md](./RFC-VPTO-Logical-Vector-ISA.md) and builds on
the `Vb32Hist256` design note
(`xson/pto-isa/kernels/manual/a5/topk/docs/logical_vb32_hist256_design.md`).

**Scope:** A5 NPU vector pipe. `pto.vmi` is a logical SIMD layer that lowers to
`pto.mi` (`pto` micro ISA) via a layout-assignment pass (`pto.as`). This doc
specifies *requirements and principles*, not a final lowering algorithm — the
last two sections deliberately keep the inference and instruction-mapping
choices open.

[toc]

---

## Part 0 — Reading guide

This document is layered so each part only depends on the ones above it:

1. **Philosophy** — the invariants every part of `pto.vmi` must preserve.
2. **Compile principles** — the contract `pto.as` (the layout-assignment pass)
   must satisfy.
3. **Rules** — the mechanical lowering rules (R1–R6) derived from layout
   metadata.
4. **Operations** — the `pto.vmi` op surface and its A/B/C classification.
5. **nVL register type** — the logical register: 1D, ragged (dynamic length),
   and the **2D hierarchical** proposal for VLane-level utilization.
6. **Open exploration** — what we deliberately leave to the compiler: inference
   ability and instruction-mapping option spaces.

---

## Part 1 — Philosophy (basic principles)

`pto.vmi` exists to give the kernel author **one logical, in-order vector
value** and to keep every physical-layout decision (interleave, half, parity,
VLane packing, fan-out) as **compile-time metadata the author never writes**.

The following principles are normative — every rule, op, and type below MUST be
consistent with them.

- **P1 — Logical-only surface.** The author sees a contiguous logical array
  `L x T` (or `R x C x T` in 2D). No `EVEN/ODD`, `Bin_N0/N1`, `INTLV_B*`,
  `PART_*`, `BRC_*`, `vcg*` VLane reasoning, or physical register count appears
  in `pto.vmi` source.

- **P2 — Layout is metadata, resolved at compile time.** Every `pto.vmi` value
  carries a `#layout` descriptor. It is populated and propagated by the
  compiler, has **zero runtime representation**, and only ever picks *which*
  `pto.mi` instructions are emitted.

- **P3 — One logical value = K physical regs + a descriptor.** A logical
  `L x T` is backed by `K = ceil(L · bitwidth(T) / 2048)` physical 256B vregs.
  `K` is implementation detail; it surfaces only inside lowering.

- **P4 — Never silently mis-lower.** If an op cannot absorb the current layout,
  the *worst* legal action is to insert an explicit `.contiguous()`
  materialization. The compiler MUST NOT reinterpret interleaved bytes as
  contiguous (or vice versa) without a real data movement.

- **P5 — Lowering is an optimization search, not a fixed recipe.** For most ops
  there are several legal `pto.mi` realizations (e.g. reduce-as-fold vs
  reduce-as-partial-sums). `pto.vmi` semantics fix the *result*, never the
  instruction schedule. Section 6 enumerates the option spaces.

- **P6 — 1:1 degeneracy.** When `K = 1` and the descriptor is contiguous, every
  `pto.vmi` op lowers to exactly one `pto.mi` op with zero overhead. `pto.vmi`
  is never more expensive than hand-written `pto.mi` for the contiguous case.

- **P7 — Decompose until UB, unless layout-unification wins.** Multi-register
  logical values should stay **decomposed across K physical regs in-register**;
  they are re-assembled into contiguous form at UB boundaries (store /
  `.contiguous()`) **or earlier only when required by legality/cost**. `pto.as`
  is allowed to run backward/forward layout inference across a use-def region:
  if one operand is interleaved and another is contiguous, it may retarget the
  contiguous path to the same interleave pattern (lazy unification) instead of
  forcing immediate materialization. If no legal common layout exists, insert an
  explicit `.contiguous()` before the blocking op (P4). The author still gets a
  single logical view throughout (applies to squeeze/unique — see R5).

---

## Part 2 — Compile principles (`pto.as` contract)

`pto.as` is the pass that turns a `pto.vmi` program into legal `pto.mi`. It MUST
provide the following, in this priority order.

- **C1 — Layout inference.** Assign every SSA `pto.vmi` value a
  `LayoutDescriptor`. Category-B producers set the axis they introduce; other
  producers inherit/propagate their input descriptor.

- **C2 — Layout propagation & coalescing.** Choose layouts so adjacent ops agree
  and avoid round-trip interleave/deinterleave. This is a layout
  allocation/coalescing problem analogous to register allocation.

- **C3 — Minimal materialization.** Insert `.contiguous()` **only** when an op
  is Category C *and* its input is not already contiguous. Track an
  `is_contiguous` bit and never materialize redundantly (P4 + cost model).

- **C4 — Cost-driven option selection.** Where a rule admits multiple `pto.mi`
  realizations (Section 6), pick by a cost model (instruction count, pipe
  pressure, latency). The chosen option MUST be observationally equal to the
  logical semantics.

- **C5 — Hoist & fuse.** Hoist loop-invariant `.contiguous()` out of loops;
  fuse consecutive contiguous-required consumers to share one materialization;
  keep per-physical-reg partial accumulators across loop iterations and only
  fold at the end (see R4 partial-sum option).

- **C6 — Lazy layout unification before materialization.** Honor P7 by default,
  but before inserting `.contiguous()`, attempt region-wide layout unification:
  propagate producer/consumer constraints backward and forward, pick a common
  legal descriptor for all Category-A paths, and materialize only at a true
  legality frontier (Category-C boundary or incompatible Category-B mode).

- **C7 — Verifiability.** Every layout transition has a numpy/byte oracle: the
  `.contiguous()` of a value MUST be byte-equal to what a store of that value
  would land in UB.

---

## Part 3 — Rules

Each rule is mechanical and derived purely from the descriptor. R1–R3 are
carried over from the `Vb32Hist256` note; R4–R6 are introduced here.

### R1 — Elementwise fan-out (Category A)

Per-lane, dtype-uniform ops (`vadd`, `vmul`, `vmin`, `vsel`, `vcmps`, `vand`,
`vcvt`-logical, …) emit one `pto.mi` op per physical reg under a full-stride
predicate. **Layout passes through unchanged.** Broadcast operands are allowed
(see R6): an operand whose cardinality along an axis is 1 is *replicate-read*.

### R2 — Mode producer: parity / half / width (Category B)

A producer that naturally writes a strided/halved sub-view introduces an axis:

- `chistv2 Bin_N0/Bin_N1` → `half` axis.
- `vcvt` widen `16→32` → `parity(r=2)` axis, modes `{EVEN, ODD}`.
- `vcvt` widen `8→32` → `parity(r=4)`, lowered as **two stacked radix-2
  stages** (A5 interleave is x2-only; `r=4` = `INTLV` of `INTLV`). Modes
  `{P0,P1,P2,P3}` are realized by stage composition, not a native x4 op.
- `vsunpack/vzunpack/vpack` → `width(r)` axis (narrow↔wide as real semantics;
  half placement as layout).

### R3 — Contiguous UB access ⇒ interleave / deinterleave (Category B)

When a consumer needs stride-1 logical order in UB:

- **Store**: emit `vsts INTLV_B*` per axis instance (per half / per parity
  stage). `r=4` ⇒ two interleave stages.
- **Load**: emit `vlds DINTLV_B*` (symmetric).

This is the only path that *consumes* a `parity`/`half`/`width` axis.

### R4 — Reduce-combine (Category B when lane-aligned; otherwise option search)

A logical reduction over a `pto.vmi.vreg` has **three sub-cases** keyed off how
the reduction axis sits in the physical hierarchy (reg / VLane / lane):

- **R4a — VLane-aligned group reduce → native Category B.** If the reduction
  axis maps exactly to the 32B VLane boundary (inner dim `C` bound at `vlane`
  level, see 2D type in 5.3), lower to `vcgadd/vcgmax/vcgmin` — **one op per
  physical reg, no cross-reg combine, no materialization.** Produces one partial
  per VLane (up to 8 per reg). Best case; this is the MoE small-inner-dim batch
  reduce.

- **R4b — Full reduce across K regs → option search (P5).** Reducing the whole
  logical array is **not** a forced `.contiguous()`. Legal realizations include:
  - **Fold-then-reduce:** `(K-1)× vadd` to fold K regs into 1, then `1× vcadd`.
    Fewest reduce ops; serial `vadd` dependency chain.
  - **Partial-then-combine:** `K× vcadd` (independent, good ILP) + a `K`-way
    scalar/tiny-vector combine tree.
  - **Loop partial-sums (C5):** keep `K` per-reg accumulators across a loop and
    fold/reduce once at the end — the "physical 2/4, logical 1" pattern: the K
    partials are only made contiguous (via `vadd`) at the tail.
  The choice is cost-model driven. The result is a degenerate small nVL
  (`L ∈ [1,8]`).

- **R4c — Arg-reduce index offset.** For `vcmax/vcmin` (value+index), any
  cross-reg combine MUST add `k · lanes_per_reg` to reg-`k` indices before
  combining, so the global argmin/argmax matches logical index order (MoE
  tie→min-index pattern). The offset injection is part of the rule, not the
  user's concern.

- **R4d — Sub-VLane / unaligned reduce → Category C.** If the reduction stride
  does not align to a 32B VLane and no `vcg*` mode applies, materialize
  (`.contiguous()`) then reduce, or rotate/shuffle. Last resort.

> **Fused reduce+broadcast.** A reduce immediately consumed by a broadcast
> (`vcadd` then `vdup`/`vbr`) is a recognized fusion: the result is logically a
> length-1 value broadcast over a VL (or logically nVL via a `broadcast` axis,
> R6). `pto.as` SHOULD emit the reduce and the broadcast back-to-back and keep
> the result as a broadcast-axis value rather than materializing K copies.

> **Order-insensitive class (normalization point).** Some ops are insensitive
> to within-register physical ordering because they collapse an axis: e.g.
> `vcadd/vcmax/vcmin` over the reduced axis, and the immediate broadcast of that
> scalar/tiny-vector result. After this collapse+broadcast pair, `pto.as` may
> normalize the descriptor to a canonical broadcast/contiguous form (1-VL
> backing for the reduced value, replicated logically to nVL) without an
> explicit interleave round-trip, as long as byte semantics at UB remain equal.

### R5 — Squeeze / compaction: scan-with-carry over a ragged value (Category C, streamed)

`vsqz/vusqz` compact predicate-active lanes to the front; the valid count is a
**runtime scalar** (`SPR SQZN`). Requirements:

- The **logical tile size stays static** in the user view (`L` known at compile
  time). Only the *number of valid elements after squeeze* is dynamic.
- Squeeze across K physical regs is a **sequential scan-with-carry**: squeeze
  reg `k`, append its survivors after reg `k-1`'s using the residual/unaligned
  store (`vstus/vstur`, which consume `SQZN`), threading the running count.
- The op **returns the post-squeeze count** as an SSA `i32` so a producer can
  store exactly the valid region and a **downstream stage can re-tile by static
  nVL with a runtime loop trip count** (load `ceil(count / L_tile)` tiles, each
  a compile-time-known nVL).
- Per P7, squeeze stays decomposed in-register and only assembles the compact
  stream when it reaches UB.

### R6 — Broadcast reuse (no materialization)

A `broadcast` axis has **physical backing of 1 reg/value** and a logical
cardinality = fan-out. Producers: `vbr` (scalar→vec), `vlds BRC_B*`, `BRC_BLK`.

- Category-A consumers (R1) of a broadcast operand emit `K` ops that all
  **replicate-read the single physical reg** — they MUST NOT expand the
  broadcast into K stored copies.
- Materialize a broadcast only at a Category-B/C edge that truly needs the
  expanded contiguous form (e.g. storing the broadcasted tile).

---

## Part 4 — `pto.vmi` operations supported

The op surface is the logical subset of `pto.mi` (RFC §8): physical-layout ops
(`vintlv/vdintlv`, `vldsx2/vstsx2` interleaved forms, all `dist`/`part` tokens)
are **removed** from the surface and become lowering artifacts. Every surviving
op declares a **Category** and the **axes it can absorb**.

### 4.1 Category table

| `pto.vmi` op(s) | Category | Absorbs / produces | Notes |
|---|---|---|---|
| `vadd vsub vmul vdiv vmax vmin vand vor vxor vshl vshr` | **A** | any (per-lane) | layout pass-through; broadcast operands via R6 |
| `vadds vmuls vmaxs vmins vshls vshrs` (vec-scalar) | **A** | any | scalar operand is implicit broadcast (R6/M2 fused) |
| `vabs vneg vexp vln vsqrt vrelu vnot` (unary) | **A** | any | |
| `vsel vselr vcmp vcmps` | **A** | any | predicate/select; mask propagates via descriptor |
| `vdup arange` (index/broadcast generators) | **A** | produces `broadcast` axis or chunk axis | Category A replication across K regs (R1) |
| `vbr` (scalar broadcast) | **B/R6** | produces `broadcast` axis | 1-reg backing; replicate-read (R6/M2) |
| `vcvt` (widen/narrow) | **B** | produces/consumes `parity(r)` / `width(r)` | `r=4` = two stages (M3) |
| `vlds vsts` (contiguous) | **A/B** | consumes `parity/half/width` on store | R3; also padded load with mask (R1 + A) |
| `chistv2` | **B** | produces `half` axis | MoE histogram/count; two-half fanout |
| `vreduce_add vreduce_max vreduce_min` (full reduce) | **B (R4a) / search (R4b) / C (R4d)** | consumes logical `reduce` axis | R4; arg-offset R4c for argmin/max |
| `vcgadd vcgmax vcgmin` (VLane group reduce) | **B** | consumes `lane`-level axis in 2D tile | R4a; 8-way per-VLane grouping; no materialize |
| `vcpadd` (pairwise) | **B** | consumes pairwise-stride axis | per-pair reduction |
| `vsqz vusqz` (squeeze/compact) | **C, streamed** | produces `ragged` axis + `i32` count | R5; scan-with-carry over K regs |
| `vgather2 vgatherb vscatter` | **C** | — | needs contiguous index table; Category C |
| `vbitsort vmrgsort4` | **C** | — | tile-level op; no decomposition |

### 4.2 Surface omissions (lowering-only in `pto.mi`)

Not part of the `pto.vmi` surface: `vintlv vdintlv`, the x2 interleaved
`vldsx2/vstsx2`, and all `dist`/`part`/`Bin_N` tokens. They appear only inside
R2/R3/R4 lowering.

---

## Part 5 — nVL register type

### 5.1 1D logical register `!pto.vmi.vreg<L x T, #layout>`

- `L` logical elements of type `T`; backed by `K = ceil(L·bitwidth(T)/2048)`
  physical regs.
- Legality: `L · bitwidth(T)` MUST be a multiple of 2048 bit (256B). Per-dtype
  `L` granularity: f32/i32 → 64, f16/bf16/i16 → 128, i8 → 256.
- `#layout` is normally omitted in source (filled by `pto.as`).

### 5.2 Ragged register `!pto.vmi.vreg<? x T, #layout>` (dynamic length)

- Produced by squeeze/compaction (R5). The **tile capacity** is static; the
  **valid count** is a runtime `i32` carried alongside the value.
- Surface form:

  ```mlir
  %compact, %count = pto.vmi.vsqz %v, %pred
      : !pto.vmi.vreg<256xf32>, !pto.vmi.mask<256xb32>
      -> !pto.vmi.vreg<?xf32>, i32
  ```

- A downstream stage re-tiles `%compact` by a **static nVL** with a **runtime
  loop trip count** `ceil(%count / L_tile)` (R5). This keeps every load a
  compile-time-known shape while the number of iterations is dynamic.

### 5.3 2D hierarchical register `!pto.vmi.vreg<R x C x T, #layout>` (proposal)

**Motivation.** The 1D nVL ignores the internal structure of a physical vreg:

```
vreg (256B) = 8 VLanes × 32B ;  VLane = E_v lanes,  E_v = 32 / sizeof(T)
              (f32 → E_v=8,  f16 → E_v=16,  i8 → E_v=32)
```

`vcgadd/vcgmax/vcgmin` reduce **within each VLane** in one instruction. For a
small inner dim `C` (e.g. an MoE per-group score of width 8 f32 = exactly one
VLane), packing one logical row per VLane lets a single `vcg*` reduce **8 rows
at once** — far better VL utilization than padding each row to a full reg. A 1D
type cannot express "row = VLane".

**Proposal — bind each logical axis to a hardware level.** A 2D logical
`R x C x T` maps onto the three-level physical hierarchy
`reg ⊃ vlane ⊃ lane`:

```
logical [R, C]            physical hierarchy
   C  (inner)   ───────►  lane    level   (within a VLane; require C ≤ E_v, pad otherwise)
   R  (outer)   ───────►  vlane   level   (0..7), then overflow to
                          reg     level   (the nVL fan-out K)
```

Concretely, `R` rows of `C` columns: each row occupies one VLane's first `C`
lanes; 8 rows fill one physical reg's 8 VLanes; `R > 8` fans out across
`K = ceil(R / 8)` physical regs.

**Descriptor extension.** Each `LayoutAxis` gains a `level ∈ {lane, vlane,
reg}` binding plus the existing `mode`:

```
#pto.vmi.vreg.layout<
  logical_shape = [R, C],
  phys_dtype    = T,
  E_v           = 32 / sizeof(T),     // lanes per VLane
  axes = [
    #axis<name="col", level=lane,  cardinality=C, mode=None,   stride=1>,
    #axis<name="row", level=vlane, cardinality=min(R,8), mode=None, stride=C>,
    #axis<name="row_hi", level=reg, cardinality=ceil(R/8),  mode=None, stride=8*C>,
  ]
>
```

**Constraints.**
- `lane`-level cardinality (here `C`) MUST satisfy `C ≤ E_v`; if `C < E_v` the
  remaining lanes are padded (identity for the reduce operator: 0 for add,
  ±INF for max/min).
- `vlane`-level cardinality ≤ 8.
- `reg`-level cardinality = `K` (the 1D nVL fan-out).

**What it buys (R4a as a first-class case).**
- A reduce over the `col` (lane-level) axis lowers to `vcgadd` — **one op per
  physical reg**, output = `R` partials laid out one-per-VLane. No
  materialization, no cross-reg combine for the within-row reduction.
- Elementwise ops (R1) are unchanged: they still fan out over the `reg` axis.
- A 1D type is the degenerate `R = (8·K)`, `C = E_v` case.

**Worked example — per-row softmax sum (the 2D win).** The AscendC
`moe_gating_top_k_softmax` VF computes a per-row reduction over the expert axis
followed by a broadcast-divide. Today the author hand-writes it as a 2D
level-2 reduce plus an explicit broadcast plus an nxVL divide loop
(`moe_gating_top_k_softmax_fullload_generalized_regbase.h`):

```cpp
uint32_t shape[] = {rowCount, expertCountAlign_};
ReduceSum<float, Pattern::Reduce::AR, false>(reduceValue, softmax, tmp, shape, true); // per-row sum
Brcb(tmpTensor, reduceValue, ...);                                                    // broadcast sum
__VEC_SCOPE__ {                                                                         // nxVL divide
  for (i < rowLoops) {
    DataCopy(sumVreg, sumTensorAddr + i*B32_BLOCK_COUNT);
    Duplicate(sumVreg, sumVreg, mask);                  // broadcast per-row sum into a reg
    for (j < expertCountLoops) {                        // fan out over K physical regs
      DataCopy(vreg0, softmaxAddr + rowOff + j*VL);
      Div(vreg0, vreg0, sumVreg, mask);
      DataCopy(softmaxAddr + rowOff + j*VL, vreg0, mask);
    } } }
```

With a 2D nVL tile `R x E`:

```mlir
%t    : !pto.vmi.vreg<R x E x f32>                              // logical 2D tile
%sum  = pto.vmi.vreduce_add %t {axis = col}                      // R4a → ReduceSum AR
      : !pto.vmi.vreg<R x E x f32> -> !pto.vmi.vreg<R x 1 x f32>
%sumb = pto.vmi.vbr %sum {axis = col}                            // R6 fused → Brcb
      : !pto.vmi.vreg<R x 1 x f32> -> !pto.vmi.vreg<R x E x f32>
%norm = pto.vmi.vdiv %t, %sumb : !pto.vmi.vreg<R x E x f32>      // R1 fan-out → the divide loop
```

The inner `col` axis is `lane`-level, so the reduce lowers to the same
`ReduceSum AR`, the broadcast to the same `Brcb`+`Duplicate`, and the divide to
the same `K`-iteration fan-out loop — **identical instructions, identical
performance**, but the source no longer spells `shape`, `Brcb`, `B32_BLOCK_COUNT`
offsets, or the `expertCountLoops` tail-mask loop.

**Open sub-questions** (carried to Section 6): non-power-of-two `C`, `C > E_v`
(row spans multiple VLanes → 1.5D), and whether `R`-axis (cross-VLane) reduce
is exposed or always materialized.

### 5.4 `LayoutDescriptor` (formal sketch)

```python
@dataclass(frozen=True)
class LayoutAxis:
    name: str                 # "chunk","parity","half","width","broadcast",
                              # "reduce","row","col", ...
    level: str                # "lane" | "vlane" | "reg"
    cardinality: int
    mode: str | None          # CCE mode: INTLV_B*, PART_*/P0..P3, Bin_N*,
                              # BRC_*, vcg*, None(=chunk)
    radix: int = 1            # parity/width radix r (2 or 4)
    stride_in_logical: int = 1

@dataclass(frozen=True)
class LayoutDescriptor:
    logical_shape: tuple[int, ...]
    phys_dtype: np.dtype
    e_v: int                  # lanes per 32B VLane
    axes: tuple[LayoutAxis, ...]

    @property
    def is_contiguous(self) -> bool:
        return all(a.mode is None for a in self.axes)
```

---

## Part 6 — Open exploration (deliberately unresolved)

These are the spaces we want the compiler to search, not fix in the spec (P5).
Listed as questions + candidate options.

### 6.1 Compiler inference ability

- **I1 — Layout assignment as allocation.** How should `pto.as` jointly choose
  layouts across a region to minimize interleave round-trips? (graph-coloring /
  ILP / greedy by op-table priority?)
- **I2 — When to keep `broadcast` vs expand.** Cost threshold for replicate-read
  (R6) vs one-time materialize when a broadcast feeds many consumers.
- **I3 — `.contiguous()` placement.** Hoisting/fusion policy (C5): how far to
  hoist out of loops, when to fuse multiple Category-C consumers.
- **I4 — 2D inference.** Auto-detect when a 1D reduce over a small inner dim
  should be re-typed to a 2D VLane-packed layout (R4a) vs left 1D.
- **I5 — Ragged re-tiling.** After a squeeze (R5), inferring the static tile
  shape + runtime trip count for the consumer, and proving the count bound.

### 6.2 Instruction-mapping options

- **M1 — Reduce realization (R4b).** fold-then-reduce vs partial-then-combine
  vs loop-partial-sums. Pick by `K`, loop depth, ILP, and whether an arg-index
  (R4c) is needed.
- **M2 — Fused reduce+broadcast.** `vcadd`+`vdup` fusion, and whether the
  broadcast result is kept as a `broadcast`-axis nVL or a real VL.
- **M3 — Radix-4 widen staging (R2).** ordering of the two `INTLV`/`PART`
  stages for `8↔32`, and whether intermediate `16` lands in UB or stays in
  registers.
- **M4 — VLane packing degeneracy.** when `C < E_v`, choose pad-and-`vcgadd`
  vs pack multiple short rows into one VLane (sub-VLane reduce → R4d).
- **M5 — Squeeze append path.** `vstus` vs `vstur` residual store selection and
  `SQZN` threading granularity (per-reg vs batched).
- **M6 — Unique/dedup decomposition.** keep the bitset `vor/vand/vsel` (pure A,
  batched 2D over rows) decomposed until the compaction store (P7), and whether
  to share the squeeze append path (R5) for the survivors.

---

## Part 7 — Consolidated open questions

1. Radix-4 interleave is realized as **two stacked radix-2 stages** (confirmed):
   spec the stage order and intermediate residency (M3).
2. Reduce result type: degenerate `!pto.vmi.vreg<1..8 x T>` vs scalar; how the
   `vcmax` value+index pair surfaces (R4/M1).
3. Ragged values in registers: may a `<?xT>` feed Category-A ops, or must it
   materialize at the squeeze store? (R5/P7).
4. 2D legality envelope: `C > E_v` (row spans VLanes), non-power-of-two `C`,
   and exposing the cross-VLane (`R`-axis) reduce (5.3).
5. Unique: single fused surface op vs decomposed bit-ops + squeeze (M6/P7).

---

## Appendix A — Worked examples from ops-transformer AscendC VF

These map **real** AscendC `MicroAPI` vector-function (VF) code in
`ops-transformer/moe/**/op_kernel/arch35/` to `pto.vmi`. The point of each is
that the nVL form lowers back to the **same `pto.mi`/CCE instructions** — so the
rewrite is purely a readability/safety win at equal performance (P6).

Source: `moe_gating_top_k_softmax_fullload_generalized_regbase.h`.

### A.1 The universal nxVL fan-out (every VF in the tree has this)

Every loop of the shape "`(len + VL - 1)/VL` chunks, `UpdateMask` tail, body
per chunk" **is** an nVL fan-out (R1). E.g. the softmax normalize divide:

```cpp
uint16_t expertCountLoops = (expertCount_ + repeatCount - 1) / repeatCount;  // = K
for (uint16_t j = 0; j < expertCountLoops; j++) {
    mask  = UpdateMask<int32_t>(precessExpert);          // tail predicate, by hand
    off   = rowOff + j * repeatCount;
    DataCopy(vreg0, addr + off);
    Div(vreg0, vreg0, sumVreg, mask);
    DataCopy(addr + off, vreg0, mask);
}
```

```mlir
%v = pto.vmi.vlds %ub[%row]        : !pto.mi.ptr<f32,ub> -> !pto.vmi.vreg<E x f32>
%n = pto.vmi.vdiv %v, %sum_bcast   : !pto.vmi.vreg<E x f32>
pto.vmi.vsts %n, %ub[%row]         : !pto.vmi.vreg<E x f32>, !pto.mi.ptr<f32,ub>
```

Lowering: exactly `K` × (`UpdateMask` + `vlds` + `vdiv` + `vsts`). **Same.**
The `expertCountLoops` count, the `j*repeatCount` offset math, and the manual
`UpdateMask` all disappear into the descriptor's `chunk`/`reg` axis.

### A.2 Fold-then-reduce-then-broadcast (softmax row max, R4b + R6)

The hand-written max-over-experts is *already* the R4b "fold-then-reduce" option
followed by the R6/M2 fused broadcast:

```cpp
LoadOneTensorForDtypeT(xAddr, reduceMid, mask, off0);            // chunk 0
for (j = 1; j < expertCountLoops; j++) {
    LoadOneTensorForDtypeT(xAddr, vreg0, mask, offj);
    Max<MERGING>(reduceMid, reduceMid, vreg0, mask);             // (K-1) folds
}
ReduceMax(reduceVreg, reduceMid, mask);                          // 1 reduce → lane0
Duplicate(dupVreg, reduceVreg, mask);                           // broadcast back
// ... Sub(x, dup); Exp(x) per chunk
```

```mlir
%m  = pto.vmi.vcmax %x                : !pto.vmi.vreg<E x f32> -> !pto.vmi.vreg<1 x f32>
%mb = pto.vmi.vbr   %m                : !pto.vmi.vreg<1 x f32> -> !pto.vmi.vreg<E x f32>
%e  = pto.vmi.vexp (pto.vmi.vsub %x, %mb) : !pto.vmi.vreg<E x f32>
```

`pto.as` picks R4b fold-then-reduce (cost model: `K-1` `Max` + `1` `ReduceMax`
beats `K` partials + combine here) and fuses the `Duplicate`. **Same
instructions.** The author stops choosing the reduce schedule by hand.

### A.3 Interleaved (value,index) load — DINTLV is layout, not user intent (R3)

The top-k stage stores sorted KV pairs interleaved at stride 2, then reads them
back splitting value/index with an explicit deinterleave token:

```cpp
DataCopy<int32_t, LoadDist::DIST_DINTLV_B32>(valueVreg, indexVreg,
                                             sortedAddr + rowOff + j*VL*2);
```

```mlir
%val, %idx = pto.vmi.vlds %kv[%row]
    : !pto.mi.ptr<i32,ub> -> !pto.vmi.vreg<K x f32>, !pto.vmi.vreg<K x i32>
```

The `DIST_DINTLV_B32` is exactly R3's consume-`parity` on load. The nVL form
names a logical `(value, index)` SoA pair; `pto.as` re-derives `DINTLV_B32`.
**Same load.**

User-specified packed structure form (explicit layout contract):

```mlir
// AoS in UB: [k0,v0,k1,v1,...] packed by user declaration.
%kv = pto.vmi.vlds %buf[%off]
  {layout = #pto.vmi.layout<aos2(k:f16, v:f16), level=lane>}
  : !pto.mi.ptr<f16,ub> -> !pto.vmi.struct<k: !pto.vmi.vreg<E x f16>, v: !pto.vmi.vreg<E x f16>>

%k = pto.vmi.getfield %kv["k"] : !pto.vmi.vreg<E x f16>
%v = pto.vmi.getfield %kv["v"] : !pto.vmi.vreg<E x f16>
```

The user chooses the packed AoS layout once in the load contract, then consumes
`%k` and `%v` as ordinary logical vectors. Accessing K and V separately is a
type-level projection; lowering selects the matching `DINTLV/INTLV` sequence.

### A.4 Widen 1B→4B + residual append (R2 radix + R5 building blocks)

The `hasFinished` path widens an `int8` flag to `int32` (R2 width), and the
store path uses the unaligned residual append that R5's squeeze scan reuses:

```cpp
Cast<int32_t, int8_t, castTrait>(finishedB32, finishedB8, finishedMask); // 1B→4B widen
...
DataCopyUnAlign(yOutAddr, valueVreg, u0, count);    // residual append (UnalignReg carry)
DataCopyUnAlignPost(yOutAddr, u0, 0);               // flush tail
```

In nVL the widen is one `pto.vmi.vcvt` (no `part`/stage tokens), and the
unaligned append is the lowering of an R5 squeeze/compaction store — the
`UnalignReg u0` carry **is** the scan-with-carry state. The user writes a
logical `vsqz` + `vsts`; `pto.as` emits the `DataCopyUnAlign`/`Post` pair.

Concrete shape-preserving example (`4 x 64` of `i8` widened to `i32`):

```mlir
// Logical shape is preserved: 4x64 stays 4x64 after widen.
%x8  : !pto.vmi.vreg<4 x 64 x i8>
%x32 = pto.vmi.vcvt %x8 : !pto.vmi.vreg<4 x 64 x i8> -> !pto.vmi.vreg<4 x 64 x i32>
```

Interpretation:

- Logical domain: still 256 elements arranged as 4 rows x 64 cols.
- Physical storage: element width grows 4x, so register fan-out `K` grows 4x.
- Descriptor: gains `width/parity` axes (radix-4 realized as two radix-2
  stages), but source shape does not change.

Equivalent explicit layout view (optional, compiler usually infers it):

```mlir
// Same logical value with explicit virtual layout annotation after widen.
%x32_l = pto.vmi.layout_cast %x32
    : !pto.vmi.vreg<4 x 64 x i32, #layout<row=4,col=64,width(r=4),reg=K4>>
```

So yes: you can read this as "`4x64` logical vector, now backed by a `4xVL`
physical decomposition". The readability gain is that shape semantics stay on
the surface while interleave staging stays in lowering.

### A.5 Summary — what collapses, and why perf is unchanged

| VF pattern (AscendC `MicroAPI`) | nVL surface | Lowers to | Rule |
|---|---|---|---|
| `for j<K { UpdateMask; load; op; store }` | one op on `vreg<L×T>` | same `K`-loop | R1 |
| `(K-1)×Max + ReduceMax + Duplicate` | `vreduce_max` + `vbr` | same | R4b+R6/M2 |
| `ReduceSum AR + Brcb + divide-loop` | 2D `vreduce_add` + `vbr` + `vdiv` | same | R4a+R6 (2D) |
| `DataCopy<DIST_DINTLV_B32>(val,idx)` | `vlds → (val,idx)` SoA | same DINTLV | R3 |
| `Cast<i32,i8>` widen | `vcvt` | same (2-stage if 1B→4B) | R2 |
| `DataCopyUnAlign + Post` | `vsqz`+`vsts` lowering | same | R5 |
| `Arange(offset) + nested loop DataCopy` | `vdup(arange)` + replicate-read | Arange ×1 + K replicate | R1+R6 |
| `K-loop: Load + Add + Muls(scalar) + Add` | nVL for-loop + vmuls scalar | same K iterations | R1+R6/M2 |
| `UpdateMask + fold-Max + Sub-Exp + ReduceSum + Brcb` | mask+vreduce_max+vreduce_add+vdiv | same reduce+bcast pipeline | R1+R4b+R4a+R6 |

In every row the nVL form carries **no** `repeatCount`, `expertCountLoops`,
`UpdateMask`, `B32_BLOCK_COUNT`, `DIST_DINTLV_B32`, `castTrait`, or `UnalignReg`
in the source, yet `pto.as` is required (P6) to emit the same instruction
stream. The win is readability and the impossibility of an interleave/tail-mask
bug, not a different schedule.

### A.6 Index generation broadcast (Arange + replicate-read, R1 + R6)

The `InitIndices()` VF initializes expert and row indices by Arange per chunk,
then replicates each chunk across multiple row lanes:

```cpp
uint32_t repeatCount = B32_VF_COUNT;                    // = 64
uint16_t expertCountLoops = (expertCount_ + 64 - 1)/64; // = K
for (i = 0; i < expertCountLoops; i++) {
    mask = UpdateMask<int32_t>(expertCount_);           // tail mask
    uint16_t offset = i * repeatCount;
    Arange(vreg0, offset);                              // [offset, offset+63]
    for (j = 0; j < rowLoops; j++) {                    // replicate across rows
        DataCopy(expertIdxAddr + (j * expertCountAlign_) + offset, vreg0, mask);
    }
}
```

The second VL_scope replicates a base row index and adds offsets:

```cpp
for (i = 0; i < kLoops; i++) {
    DataCopy(vreg0, rowIdxAddr + offset);              // read base
    for (j = 1; j < rowLoops; j++) {
        Adds(vreg1, vreg0, j, mask);                   // offset by row j
        DataCopy(rowIdxAddr + (j*expertCountAlign_) + offset, vreg1, mask);
    }
}
```

```mlir
%e_idx = pto.vmi.arange              : !pto.vmi.vreg<E x i32>
%e_tiled = pto.vmi.vdup %e_idx      : !pto.vmi.vreg<E x i32> -> !pto.vmi.vreg<R x E x i32>
%r_base = pto.vmi.vlds %row_base    : !pto.mi.ptr<i32,ub> -> !pto.vmi.vreg<K x i32>
%r_offs = pto.vmi.vadds %r_base, [0,1,2,...]           : !pto.vmi.vreg<K x i32> (broadcast scalar offset, per row)
```

Lowering: `Arange` is Category A (R1), emitted once per chunk; `vdup` broadcasts
the result across rows (R6 + M2) with replicate-reads, costing zero extra
instructions. The `vadds` scalar add fans out normally. **Same as hand-written.**

### A.7 Weighted accumulate (softmax-like): fold-then-reduce + fused broadcast-scale (R4b + R6 + M2)

The `moe_finalize_routing` VF accumulates K expert representations scaled by
routing weights — the classic `Σ scale_k · (row_k + bias_k)`:

```cpp
// Setup phase: broadcast sum per chunk
LocalTensor<T> outLocal;                                // accumulator nVL
for (int k = 0; k < K; k++) {
    T scalesVal = scalesLocal[k];
    int32_t expertIdx = expertForSourceRow[k];
    // Load row k
    DataCopyPad(rowTmp, gmExpandedPermutedRows[idx[k]], ...);
    DataCopyPad(biasTmp, gmBias[expertIdx], ...);
    // Compute
    Add(rowTmp, rowTmp, biasTmp, dataLen);              // row + bias, Category A (R1)
    Muls(rowTmp, rowTmp, scalesVal, dataLen);           // row *= scale (scalar broadcast add, R6)
    Add(outLocal[0], outLocal[0], rowTmp, dataLen);     // accumulate, Category A (R1)
}
```

```mlir
%acc = pto.vmi.vbr 0.0 : !pto.vmi.vreg<H x f32>
for %k = 0 to K {
    %s_k = scf.extract scales[%k]                          // scalar
    %row = pto.vmi.vlds(perm_ub[idx[k]])                   // load row k → nVL
    %bias = pto.vmi.vlds(bias_ub[expert[k]])               // load bias → nVL
    %r_b = pto.vmi.vadd %row, %bias                        // Category A: fan-out ×K
    %r_bs = pto.vmi.vmuls %r_b, %s_k                       // scalar mul: fused broadcast (R6/M2)
    %acc = pto.vmi.vadd %acc, %r_bs                        // accumulate: Category A
}
pto.vmi.vsts %acc, out_ub[token]
```

Lowering: the K-iteration loop runs as-is (scalar loop). Inside, `vadd` + `vadd`
(rows + bias, then accumulate) are both Category A fan-out; `vmuls` is a
broadcast-scalar multiply recognized as R6/M2 fused, costing one `Muls` per
physical reg + no gather overhead. **Identical to the hand-written loop.**

### A.8 Padded load + mask + per-group reduce (R1 + mask propagation + R4a for VLane groups)

The `ComputeSoftmax` VF loads (with padding), applies pointwise softmax, then
computes per-row sums and broadcasts back for normalization:

```cpp
__VEC_SCOPE__ {
    for (i = 0; i < rowLoops; i++) {
        // Load chunk j with mask for tail elements
        for (j = 0; j < expertCountLoops; j++) {
            mask = UpdateMask<float>(expertCount_);
            LoadOneTensorForDtypeT<T>(xAddr, vreg, mask, offset);
            Max<MERGING>(reduceMid, reduceMid, vreg, mask);  // (K-1) folds for per-row max
        }
        ReduceMax(reduceVreg, reduceMid, mask);              // 1 reduce (R4b)
        Duplicate(dupVreg, reduceVreg, mask);               // broadcast max (R6)
        // Subtract and exp per chunk
        for (j = 0; j < expertCountLoops; j++) {
            LoadOneTensorForDtypeT(xAddr, vreg0, mask, offset);
            Sub(vreg0, vreg0, dupVreg, mask);                // Category A: fan-out
            Exp(vreg0, vreg0, mask);                         // Category A: fan-out
        }
    }
}
// Compute row sums and normalize
ReduceSum<float, Pattern::Reduce::AR>(reduceValue, softmax, tmp, shape, true);
Brcb(tmp, reduceValue, ...);                                // broadcast per-row sum
__VEC_SCOPE__ {
    for (i = 0; i < rowLoops; i++) {
        DataCopy(sumVreg, sumTensorAddr + i*BLOCK_COUNT);
        Duplicate(sumVreg, sumVreg, mask);                  // broadcast sum
        for (j = 0; j < expertCountLoops; j++) {
            DataCopy(vreg0, softmaxAddr + offset);
            Div(vreg0, vreg0, sumVreg, mask);               // broadcast divide (R6)
        }
    }
}
```

```mlir
%x = pto.vmi.vlds %x_ub[row]  : !pto.vmi.vreg<E x T>
%x_pad = pto.vmi.vsel %mask, %x, neginf  // mask + pad, Category A + predicate
%amax = pto.vmi.vreduce_max %x_pad  // R4b: fold-then-reduce
%amax_b = pto.vmi.vbr %amax         // R6 + M2: broadcast
%exp = pto.vmi.vexp (pto.vmi.vsub %x_pad, %amax_b)  // Category A fan-out

%sum = pto.vmi.vreduce_add %exp (2D version)  // R4a if 2D, else R4b
%sum_b = pto.vmi.vbr %sum                     // broadcast sum
%norm = pto.vmi.vdiv %exp, %sum_b             // broadcast divide (R6)
pto.vmi.vsts %norm, %y_ub[row]
```

Lowering: load → pad/mask (Category A); fold-then-reduce + broadcast (R4b+R6)
for max-subtract-exp; per-row reduce+broadcast (R4a or R4b depending on layout);
final normalize divide with broadcast (R6). Compared to the hand-written code,
the `UpdateMask` logic, `repeatCount`, `BLOCK_COUNT` arithmetic, and the
explicit `Duplicate` calls all collapse into the descriptor. The `ReduceSum AR`
+ `Brcb` step is inferred as R4a+R6 when the 2D type is recognized. **Same
instructions, cleaner intent.**

### A.9 Summary table (extended)

| VF pattern (AscendC `MicroAPI`) | nVL surface | Lowers to | Rule | Kernel source |
|---|---|---|---|---|
| `for j<K { UpdateMask; load; op; store }` | one op on `vreg<L×T>` | same `K`-loop | R1 | everywhere |
| `(K-1)×Max + ReduceMax + Duplicate` | `vreduce_max` + `vbr` | same | R4b+R6/M2 | softmax max |
| `ReduceSum AR + Brcb + divide-loop` | 2D `vreduce_add` + `vbr` + `vdiv` | same | R4a+R6 (2D) | softmax norm |
| `DataCopy<DIST_DINTLV_B32>(val,idx)` | `vlds → (val,idx)` SoA | same DINTLV | R3 | sorted KV read |
| `Cast<i32,i8>` widen + `DataCopyUnAlign/Post` | `vcvt`, `vsqz`+`vsts` | same | R2, R5 | hasFinished+squeeze |
| `Arange per chunk + replicate via nested loop` | `vdup(vmi.arange)` + replicate broadcast | Arange ×1 + replicate-reads | R1+R6 | InitIndices |
| `K-iteration: Load + Add + Muls (scalar) + Add` | `for-loop{ vlds + vadd + vmuls + vadd }` | K iterations with R6 scalar mul | R1+R6/M2 | finalize_routing |
| `UpdateMask + fold-reduce-max + subtract-exp + ReduceSum + Brcb + normalize-divide` | mask+pad + vreduce_max + vexp + vreduce_add + normalize | same reduce+bcast+divide steps | R1+R4b+R4a+R6 | ComputeSoftmax |

In every row the nVL form carries **no** `repeatCount`, `expertCountLoops`, `UpdateMask`, `B32_BLOCK_COUNT`, `DIST_DINTLV_B32`, `castTrait`, or `UnalignReg` in the source, yet `pto.as` is required (P6) to emit the same instruction stream. The win is readability, elimination of interleave/mask arithmetic bugs, and explicit layout intent at the call site.

### A.10 `pto.vmi` vs normal `pto.mi`/real-vreg mapping — benefit summary

| Topic | Normal `pto.mi` / real vreg coding | `pto.vmi` with virtual layout | Net benefit |
|---|---|---|---|
| Tail/chunk handling | Author writes `repeatCount`, chunk loops, `UpdateMask` at every site | One logical op on nVL; chunking inferred | Less boilerplate, fewer off-by-one/tail bugs |
| Layout tokens | Author manages `DIST_*`, `INTLV_*`, `PART_*`, `Bin_*` explicitly | Layout in descriptor; op surface stays logical | Lower cognitive load, safer refactor |
| Mixed-layout operands | Manual choice: convert one side now vs keep mixed state | C6 lazy unification searches path first, materializes only at frontier | Fewer unnecessary interleave round-trips |
| Reduce + broadcast | Hand-select fold order + duplicate/broadcast steps | R4+R6 classify and fuse; result can normalize to 1-VL backing | Same perf, clearer intent |
| Packed AoS access (`k/v`) | Manual deinterleave and pointer arithmetic | One packed-load contract + `getfield(k/v)` | Structure-aware readability, same instructions |
| 1B→4B widen | Manual staging details and part/interleave reasoning | One `vcvt` at source, shape-preserving (`4x64 -> 4x64`) | Type-level clarity, staging hidden |
| Squeeze append | Manual `UnalignReg` carry threading | `vsqz` + `vsts` logical op, carry in lowering | Easier correctness for ragged paths |
| Performance parity | Achieved by hand tuning | Required by P6 (1:1 degeneracy and equal lowering choices) | Maintainability without perf loss |

Practical reading: `pto.mi` tells you **how bytes move now**; `pto.vmi` tells
you **what vector meaning is**, and delegates byte movement choices to `pto.as`
under explicit rules (P1–P7, C1–C7, R1–R6).

---
