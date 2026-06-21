# `pto.vmi` — Design Exploration: 2D mapping, register budget, fusion placement, predicates

**Status:** design exploration / discussion driver. Companion to
[RFC-VPTO-Logical-Vector-ISA.md](./RFC-VPTO-Logical-Vector-ISA.md) and
[pto-vmi-requirements.md](./pto-vmi-requirements.md). This doc deliberately stays
at the *design-options* level and ends each part with concrete open issues, so
it can be used to continue the discussion rather than freeze decisions.

**Why this doc exists.** The requirements doc establishes the philosophy (P1–P8)
and the lowering rules (R1–R6), but four areas are still under-specified and were
flagged as hard to follow from the existing examples:

1. **2D — where does it live?** Should the *register* stay 1D and 2D appear only
   as a *grouping concept* on non-elementwise ops (reduce/broadcast by group), or
   should we introduce a true 2D register type? And how does a 2D tile **in UB**
   map onto an nxVL logical register so fused ops stay correct?
2. **Register budget.** Auto VF fusion needs a model for the **32 vector
   registers** *and* the **8 predicate registers**, otherwise fan-out × widening ×
   multi-step chains overflow silently.
3. **Where does fusion run?** Pros/cons of doing VF loop fusion on the logical
   nxVL vs on real VL vs in the C++ header library — measured against the actual
   MoE/quant fusion space.
4. **Predicates.** The A5 predicate file is **8 registers, each 256-bit, 1 bit per
   byte**. For a 4B element only the LSB of each 4-bit group is meaningful, so
   parity/interleave data layouts force matching **predicate pack / unpack /
   interleave / bit-extend** operations. This was never worked through.

Hardware constants used throughout (A5 vector pipe):

```
vector register file : 32 architectural vregs, 256 B (2048 bit) each
predicate file       : 8  architectural pregs, 256 bit each, 1 bit controls 1 byte
VLane                : 32 B sub-lane; 8 VLanes per vreg
E_v = 32 / sizeof(T) : lanes per VLane   (f32 -> 8, f16 -> 16, i8 -> 32)
VL_B32 = 64, VL_B16 = 128, VL_B8 = 256   : logical lanes per vreg by element width
K   = ceil(L * bitwidth(T) / 2048)       : physical vregs backing one logical L x T
```

---

## Part 1 — The 2D question: register vs view vs UB-tile mapping

### 1.1 The tension

The requirements doc (5.3) proposes a 2D register `vreg<R x C x T>` bound to the
`reg ⊃ vlane ⊃ lane` hierarchy. That is the right *physical* story (it is what
`vcgadd/vcgmax` need), but it raises an interface question: **do we want the
programmer to declare 2D registers, or keep the surface 1D and let 2D appear only
where it is actually needed (group reduce / group broadcast)?**

PTO-gym's instinct — *keep the register-level interface 1D* — is sound: almost
every elementwise op (R1) does not care about the inner structure, and a 1D
`vreg<L x T>` is the simplest mental model. The 2D structure only matters for the
**non-elementwise** ops (`vreduce_*`, `vbr`, `vcg*`, pairwise). So we have a
genuine 3-way choice.

### 1.2 Three interface levels

| Level | Surface | 2D appears as | Example |
|---|---|---|---|
| **L0 — pure 1D** | `vreg<L x T>` only | nowhere; cross-row reduce is Category C (materialize) | today's stable core |
| **L1 — 1D reg + group op-attribute** | `vreg<L x T>`; ops carry `{group=C}` | a *per-op* grouping hint on reduce/bcast only | `vreduce_add %v {group=32}` |
| **L2 — true 2D reg type** | `vreg<R x C x T>` | a first-class type with axis→level binding | `vreg<4 x 32 x bf16>` |

**L1 — group as an op attribute (no new type).** The register stays
`vreg<L x T>`. A reduce or broadcast takes a `group` size `C` (with `C | L`), and
the lowering maps the group axis onto VLane (`vcgadd`) when `C ≤ E_v` and aligned,
otherwise falls back to Category C. Nothing in the *type* changes; the 2D-ness is
local to the one op that needs it.

```mlir
%v   = pto.vmi.vlds %x_ub          : !pto.mi.ptr<f32,ub> -> !pto.vmi.vreg<256xf32>
// "reduce every 32 lanes to 1" — group is an attribute, register stays 1D:
%sum = pto.vmi.vreduce_add %v {group = 32}
     : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<8xf32>   // 8 partials (256/32)
%sb  = pto.vmi.vbr %sum {group = 32}
     : !pto.vmi.vreg<8xf32> -> !pto.vmi.vreg<256xf32>   // replicate each partial across its 32
%n   = pto.vmi.vdiv %v, %sb        : !pto.vmi.vreg<256xf32>   // R1, pure 1D
```

- **Pros:** minimal surface; elementwise code never sees 2D; matches PTO-gym's
  "register interface is 1D"; trivially degenerates to L0 when no `group` is used.
- **Cons:** the group→VLane mapping is *re-inferred at each op*; a chain that wants
  to keep a stable 2D layout (e.g. several per-row ops) cannot carry it in the
  type, so the inference must re-derive "this 8-partial value is one-per-VLane"
  every time. Ambiguous when `C` does not divide `E_v` or spans VLanes.

**L2 — true 2D type.** The layout is stable across a chain and the load/store
mapping is explicit, at the cost of a heavier surface and the programmer (or the
API) committing to `R x C`.

### 1.3 Recommendation — **2D as a typed *view*, storage stays 1D**

Keep the **stored / SSA value 1D** (`vreg<L x T>`, the programmer default), and
introduce 2D only as a **reinterpreting view** consumed by group-aware ops:

```mlir
%v2 = pto.vmi.group_view %v {rows = R, cols = C}
    : !pto.vmi.vreg<L x T> -> !pto.vmi.vreg<R x C x T>     // L == R*C, zero-cost
```

- `group_view` is a **layout-only cast** (no data movement): it only asserts how
  the existing K physical regs are read as `(row, col)`. It is the bridge between
  L1's simplicity and L2's stability — the type *can* carry 2D when a chain needs
  it, but the default storage and all R1 ops stay 1D.
- Group-aware ops (`vreduce_*`, `vbr`, `vcg*`, pairwise) accept either a 1D value
  + `{group=C}` (the L1 sugar, which is just `group_view` + op) **or** a 2D view.
- `is_contiguous` and the descriptor are unchanged; `group_view` just adds
  `row`/`col` axes with `level ∈ {lane, vlane, reg}` bindings (req-doc 5.3).

This means **the programmer writes 1D**; the API/library inserts `group_view` at
the exact ops that need VLane structure. It is the smallest change that still lets
`pto.as` keep a stable 2D layout across a fused per-row chain.

### 1.4 The UB 2D-tile → nxVL mapping (the part that must be correct for fusion)

A 2D tile lives in UB as `[rows][cols]` with some row stride `S` (often padded to
32B). Loading it into an nxVL register must define a **mapping**
`(row r, col c) → (phys reg k, VLane v, lane l)` and record it in the descriptor,
because every fused reduce/broadcast afterwards depends on it. Two canonical
mappings cover the MoE/quant cases:

**M-A — row-into-VLane (the `vcg*` / block-reduce mapping).** Used when the inner
dim is small (`C ≤ E_v`): each row occupies the first `C` lanes of one VLane; 8
rows fill one vreg; `R > 8` fans out across `K = ceil(R/8)` regs.

```
UB tile  [R rows][C cols], C <= E_v          nxVL register (one vreg = 8 VLanes)
 row0: c0 c1 ... c(C-1) [pad..]   ─────►   VLane0: row0 cols | pad(identity)
 row1: c0 c1 ... c(C-1) [pad..]   ─────►   VLane1: row1 cols | pad(identity)
 ...                                        ...
 row7: ...                        ─────►   VLane7: row7 cols | pad(identity)
```

- Load: `vlds` with row stride `S`; if `C < E_v`, the tail lanes of each VLane are
  **padded with the reduce identity** (0 for add, ±INF for max/min) *and* masked
  off (see Part 4 — the load mask and the pad value must agree).
- A `vreduce_add {group=C}` then lowers to **one `vcgadd` per vreg**, output = `R`
  partials, one per VLane. No cross-reg combine, no `.contiguous()`. This is the
  MoE per-row softmax sum and the MX 32-element block reduce.

**M-B — interleaved cols (the parity / DINTLV mapping).** Used when adjacent cols
arrive interleaved in UB (e.g. `vldsx2 DINTLV_B16`, sorted KV pairs, MX windows):
the load *produces* a `parity` axis, so `col` is split EVEN/ODD across two regs.
The descriptor records `parity` so a later contiguous store re-interleaves
(`vstsx2 INTLV`) and the governing predicate is split with `pdintlv` (Part 4).

```
UB: c0 c1 c2 c3 c4 c5 ...  (interleaved)   nxVL: regEVEN = c0 c2 c4 ...
                                                 regODD  = c1 c3 c5 ...
```

**Why this must be explicit.** If a fused reduce runs *before* the mapping is
known, it cannot tell M-A (reduce within VLane → `vcgadd`) from M-B (reduce across
the parity split → fold the two regs first). So the **load op is where the
mapping/axis is committed**, and fusion downstream only reads it. This is the
concrete reason fusion must run with the descriptor present but the *mode* still
abstract (Part 3).

### 1.5 Open issues (2D)

- **O-2D.1** `C` not dividing `E_v`, or `C > E_v` (row spans VLanes → "1.5D"):
  pack multiple short rows per VLane (M4) vs pad-and-waste vs materialize?
- **O-2D.2** Does `group_view` need to survive layout assignment as a real axis,
  or is it always lowered immediately at the consuming op?
- **O-2D.3** Cross-VLane (R-axis) reduce: expose as a second `vcg`-style step, or
  always Category C?
- **O-2D.4** Should `vlds` of a 2D tile take the mapping (`{map = vlane_rows}`) as
  an explicit token, or infer it from the consuming reduce's `group`? (Explicit is
  safer for fusion correctness; inferred is terser.)

---

## Part 2 — Register-budget model for auto VF fusion

Auto fusion that fans a logical value over `K` regs and chains several steps will
overflow the file unless budget is a first-class constraint. There are **two**
budgets, and the predicate one is easy to forget.

```
VREG budget : 32 total. Target working set <= ~24 (headroom for addr/align/temps).
PREG budget : 8  total. Target working set <= ~6  (headroom for tail + cmp results).
```

### 2.1 Live-set accounting

At any program point inside a fusion region, the live cost is:

```
live_vregs = Σ_{v live}  K(v)                       // each logical value costs its K
           + temps (cast intermediates, reduce scratch)
live_pregs = #distinct governing predicates live    // see Part 4: axis variants count
           + carries (vaddc/vsubc), align/unalign carriers
```

The two multipliers that blow this up:

- **Widening** multiplies `K`: `i8 -> i32` is `K×4`; quant `f32 -> int16 -> half ->
  int8` keeps 2–3 widths live at once.
- **Op-by-op fan-out across all K**: lowering a chain by doing op1 on all K regs,
  then op2 on all K regs, keeps `chain_depth` values × `K` regs live
  simultaneously.

### 2.2 Pressure-reduction levers (in preference order)

1. **Per-physical-reg fusion of Category-A chains (primary lever).** For a maximal
   elementwise subgraph between layout-changing ops, loop `k = 0..K-1` (and over
   VLanes) on the *outside* and run the whole chain on reg `k`. Live data drops to
   `chain_depth` (≈ 2–4), **independent of K**. This is the same transform as VF
   loop fusion, applied on the logical nxVL.
2. **Serialize the fan-out (Tier-2).** When even one full-width op cannot fit
   (large `K`, streaming), emit a bounded loop over `K` rather than unrolled
   straight-line code. Programmer-owned tiling stays outside.
3. **Spill to UB via `.contiguous()`.** Last resort (P4); the only legal way to
   drop a layout. Costs `K` stores + `K` loads.

A fusion region is **legal under budget** iff its peak `live_vregs ≤ 24` and
`live_pregs ≤ 6` after applying levers 1–3. The cost model picks the cheapest
lever sequence that satisfies both.

### 2.3 Worked pressure counts

**MX `ComputeMaxExp` (quant, 2-VL DINTLV window).** Naive op-by-op:

```
vdExp0, vdExp1            (2)   ← DINTLV load (parity, K=2)
vdExpExtract0, vdExpExtract1 (2)
vdMaxExp                  (1)
+ exp mask, tail mask     (2 pregs)
peak ≈ 5 vregs, 2 pregs   → fits, but only because the window is 2 VLs.
```

Now scale to a 4-VL window or add the scale-broadcast step and the FP4 pack chain:
without per-reg fusion the extract/fold/reduce live-set tracks `K`, and the
`half -> int8` cast chain adds 2 more widths. Per-reg fusion (lever 1) keeps it at
`chain_depth` regardless of window size.

**Softmax row (`x, xpad, amax, amaxb, exp, sum, sumb, norm`).** Eight named values;
at `K=4` op-by-op that is up to `8×4 = 32` vregs → overflow. With per-reg fusion the
elementwise spans (`xpad`, `sub`, `exp`, `div`) collapse to one loop body
(`chain_depth ≈ 3`), and only the reduce boundaries (`amax`, `sum`) need all `K`
live → peak ≈ `K + 3`, comfortably under 24.

### 2.4 Open issues (budget)

- **O-B.1** Budget thresholds (24/6) — fixed, or per-kernel tunable via an attribute?
- **O-B.2** Interaction of lever-1 (per-reg fusion) with reduce boundaries: a reduce
  forces all `K` live, breaking the loop. How to schedule fuse-region splitting?
- **O-B.3** Should the predicate budget be modeled jointly with vregs (one ILP) or
  as a separate, looser constraint (pregs rarely bind except at tails/cmp)?
- **O-B.4** Spill granularity: spill a whole nxVL value, or only the `K`-tail that
  overflows?

---

## Part 3 — Where does VF fusion run? (pros/cons over the real fusion space)

The goal you stated: **fuse on a virtual nxVL first, then lower to real-VL base
instructions, so the fusion pass never has to handle interleave/part/half layout.**
There are three places fusion could live.

| Option | Fuses on | Sees layout modes? | Mechanism |
|---|---|---|---|
| **A** | logical `pto.vmi` (descriptor present, modes abstract) | **no** | new `VPTOFuse` MLIR pass above `InferLayout` |
| **B** | real VL `pto.mi` / VPTO | **yes** | extend existing `PTOInferVPTOVecScope` |
| **C** | C++ header library | no (templates) | expression templates over `NVL<L,T>` |

### 3.1 The fusion space (what actually needs fusing in MoE/quant)

| Kernel / pattern | Fusion needed | Layout involved | A | B | C |
|---|---|---|---|---|---|
| `mask_indices_by_tp` (pure elementwise stream) | chain 5 vops + load/store, pick batch N | `chunk` only | ✅ | ✅ | ✅ |
| `reduce_fused` / `expand_to_fused` (bandwidth) | load→fma→store, K-loop | `chunk` | ✅ | ✅ | ✅ |
| `ascend_quant` (f32→{fp8,int8,int4}) | scale/offset + **cast chain** + pack store | `width`+`parity` (pack) | ✅ | ⚠️ must know pack | ✅ (if constexpr) |
| MX `ComputeMaxExp` | DINTLV load + extract + **block reduce** + bcast | `parity`+VLane | ✅ | ⚠️ DINTLV explicit | ✅ |
| softmax row | fold-max + reduce + bcast + sub/exp + sum + div | `chunk`(+group) | ✅ | ✅ | ✅ |
| histogram (`group_count`/`aux_fi`) | chistv2 + widen + accumulate + INTLV store | `half`+`parity` | ✅ | ❌ fusion entangled with Bin_N/PART | ⚠️ |
| `top2_sum_gate` | per-group reduce + topk + softmax | `chunk`+`group` | ✅ | ⚠️ | ✅ |

**Reading the table.** The kernels where Option B struggles are exactly the
layout-heavy ones (histogram half/parity, quant pack, MX DINTLV): on real VL the
fusion pass would have to match `Bin_N0/Bin_N1`, `PART_EVEN/ODD`, `INTLV_B32` *and*
decide fusion at the same time — the entanglement you want to avoid. Option A fuses
on the logical chain (`chistv2` is one op, `vcvt` is one op) and only *afterwards*
does `InferLayout` expand them into Bin_N/PART/INTLV. Option C gets the same
layout-freedom but only inside one header-only kernel (no cross-call fusion, no
MLIR cost model).

### 3.2 Pros / cons

**Option A — logical `VPTOFuse` above `InferLayout` (recommended for the compiler path).**
- ➕ Fusion reasons only about iteration domains (`K`, VLane) and op categories;
  never about EVEN/ODD/INTLV. Re-layout cannot invalidate a fusion decision.
- ➕ Handles the layout-heavy kernels (histogram/quant/MX) cleanly.
- ➕ Shares the static op-table with the budget model (Part 2) and the predicate
  model (Part 4).
- ➖ New pass + a logical dialect level to maintain; needs the descriptor present
  (but mode-abstract) at fusion time.

**Option B — extend `PTOInferVPTOVecScope` (real VL).**
- ➕ Reuses an existing pass; no new dialect level.
- ➖ Fusion sees fully-lowered layout → the entanglement above; every new CCE mode
  risks a silent mis-fuse. Loses the stated design goal.
- ➖ Budget/predicate reasoning happens after layout, so widening blow-ups are
  already committed.

**Option C — C++ expression templates in the header library.**
- ➕ Pragmatic *now*: the A5 `T*.hpp` library is header-only; `if constexpr` on
  static shape already selects branches (see `TCvt` 1D/2D, `TQuant` layouts).
- ➕ Zero new compiler infra; perfect for hand-written kernels.
- ➖ No cross-op MLIR cost model; fusion limited to what templates can see in one
  TU; no automatic `.contiguous()`/budget spill.

### 3.3 Recommendation — A for the compiler, C for the library, one shared op-table

```
TileLang / DSL ─► pto.vmi (logical)
                     │  VPTOFuse        (Option A: layout-agnostic nxVL fusion)
                     ▼
                  pto.vmi (fused regions, modes still abstract)
                     │  InferLayout     (assign parity/half/width/chunk + predicate axes)
                     ▼
                  VPTOLowerToVL  ─► pto.mi / VPTO real-VL  ─► PTOInferVPTOVecScope ─► LLVM
```

Hand-written kernels use the **same op-table** through Option C templates
(`NVL<L,T>` + branch dispatch), lowering to the existing `T*.hpp` so P6 (1:1) and
instruction parity hold. The op-table (Category A/B/C + absorbable axes +
predicate companion, Part 4) is the single source of truth both paths consult.

### 3.4 Open issues (fusion)

- **O-F.1** Does `VPTOFuse` need the descriptor *fully* inferred, or only the
  axis *kinds* (parity/half present yes/no) to make fusion decisions?
- **O-F.2** Fusion region boundaries: are they exactly the Category-B/C ops, or do
  some Category-B ops (e.g. broadcast) fuse *through*?
- **O-F.3** How do Option A and Option C stay in sync (one op-table generator for
  both MLIR td and C++ headers)?

---

## Part 4 — Predicates on nxVL (the missing model)

`pto.vmi` hides data layout, but a layout change **also changes the predicate**.
This was never worked through, and it is where the A5 predicate file's quirks bite.

### 4.1 A5 predicate facts

- **8 architectural predicate registers**, each **256-bit**, **1 bit controls 1
  byte** of the governed vector register.
- Element-width views: `mask<b8>` (256 lanes, 1 bit/elem), `mask<b16>` (128 lanes,
  2 bits/elem), `mask<b32>` (64 lanes, 4 bits/elem).
- For a 4-byte element the predicate spends **4 bits per element, and only the LSB
  of each 4-bit group is the meaningful "lane active" bit** (the other 3 cover the
  upper bytes). This matters the moment a value is *re-laid-out*: splitting fp32
  data into EVEN/ODD halves is a **stride-2 reshuffle of 4-bit groups**, not of
  single bits — so the predicate must be reshuffled with the matching op.
- Native predication is **ZEROING** (inactive lane → 0).

### 4.2 Principle — the predicate is a logical companion that carries the same axes

A logical `mask<L x G>` is backed by `K' ` predicate regs and **carries the same
`LayoutDescriptor` axes as the data value it governs**. The programmer writes one
logical predicate over `L`; `pto.as` fans it out and applies, to the predicate,
the *companion* of whatever transform it applied to the data.

```
data axis / transform      data op                 predicate companion op
─────────────────────      ───────────────         ──────────────────────────
parity (16→32 widen)       vcvt PART_EVEN/ODD       ppack PART=LOWER/HIGHER  (keep parity bit)
parity → contiguous (load) vlds DINTLV_B*           pdintlv_b*               (split governing pred)
parity → contiguous (store)vsts INTLV_B*            pintlv_b*                (merge governing pred)
width (16↔32, radix-2)     vcvt PART_EVEN/ODD       ppack / punpack          (one radix-2 step)
width (8↔32, radix-4)      see 4.6 — NO radix-4 predicate op needed (load/store dist carries the spread)
half (Bin_N0/N1)           chistv2 Bin_N*           per-half predicate (PAT_ALL each, usually)
tail                       plt/pge                  per-reg tail predicate; full regs use PAT_ALL
```

The two predicate reshape ops in play:

- **`ppack PART=LOWER/HIGHER`** keeps one bit out of each adjacent 2-bit group and
  packs them into the selected half — the predicate analog of `vcvt PART_EVEN/ODD`.
- **`punpack PART=LOWER/HIGHER`** zero-extends each 1-bit element into a 2-bit
  group — the **bit-extend** the question referred to, used when a narrow-element
  predicate (`b16`) must govern a widened (`b32`) value.
- **`pintlv_b*` / `pdintlv_b*`** are the predicate analog of `vstsx2 INTLV` /
  `vldsx2 DINTLV`.

### 4.3 Why the LSB-meaningful detail forces this

Consider a tail mask over 100 active `i16` lanes that you then widen to `i32`.

- The data `vcvt PART_EVEN` takes lanes `0,2,4,…` into `regEVEN`, `PART_ODD` takes
  `1,3,5,…` into `regODD`.
- The **predicate must follow the same parity split**, but at the b16→b32 *group*
  granularity. `ppack PART=LOWER` on the b16 tail mask yields the b32 predicate for
  the EVEN reg; `ppack PART=HIGHER` yields the ODD reg's predicate. Each output is a
  proper b32 mask whose meaningful bits are the LSBs of its 4-bit groups.
- If instead you reused the b16 mask bit-for-bit on a b32 register, every other
  4-bit group would be misaligned → silent wrong-lane predication. That is exactly
  the class of bug P4 forbids, and it is invisible without the companion op.

### 4.4 Predicate budget (only 8 regs)

A naive lowering that materializes a distinct predicate per `(axis-instance × K)`
overflows 8 fast. Mitigations, mostly cheap:

- **Full masks are axis-invariant.** `ppack`/`pdintlv` of an all-active `PAT_ALL`
  is still all-active. So when a region has no tail and no data-dependent mask, **all
  K regs share one `PAT_ALL`** — predicate pressure ≈ 1.
- **Only tails and `vcmp` results cost predicate regs.** The expensive cases are
  data-dependent masks: softmax `-inf` pad, top-k winner filter, MX exponent
  extract. Each such mask, if it crosses a parity/width axis, needs its companion
  variant live → that is where the 8-reg budget actually binds.
- **Recompute vs keep.** A tail predicate is cheap to recompute (`plt` is one op);
  prefer recompute over keeping many parity-variant copies live.

### 4.5 Worked example — `i16 → i32` widen + bias-add + contiguous store, with tail

Logical surface (one predicate over 100 active lanes; no part/pack/INTLV):

```mlir
%p   = pto.vmi.plt %rem            : i32 -> !pto.vmi.mask<128 x b16>   // 100 active of 128
%a   = pto.vmi.vlds %x_ub, %p      : !pto.mi.ptr<i16,ub> -> !pto.vmi.vreg<128xi16>
%w   = pto.vmi.vcvt %a             : !pto.vmi.vreg<128xi16> -> !pto.vmi.vreg<128xi32>
%s   = pto.vmi.vadd %w, %b, %p     : !pto.vmi.vreg<128xi32>
pto.vmi.vsts %s, %y_ub, %p         : !pto.vmi.vreg<128xi32>, !pto.mi.ptr<i32,ub>
```

`pto.as` data + **predicate** co-lowering (K grows 1→2 across the widen):

```mlir
// --- predicate companions, derived from %p ---
%pE = pto.ppack %p {part="LOWER"}  : !pto.mask<b16> -> !pto.mask<b32>   // EVEN reg pred
%pO = pto.ppack %p {part="HIGHER"} : !pto.mask<b16> -> !pto.mask<b32>   // ODD  reg pred
// --- data path ---
%e  = pto.vcvt %a, %p  {part="EVEN"} : !pto.vreg<128xi16>, !pto.mask<b16> -> !pto.vreg<64xi32>
%o  = pto.vcvt %a, %p  {part="ODD"}  : !pto.vreg<128xi16>, !pto.mask<b16> -> !pto.vreg<64xi32>
%se = pto.vadd %e, %bE, %pE          : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32>
%so = pto.vadd %o, %bO, %pO          : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32>
// --- contiguous store re-interleaves data AND predicate ---
%pI, %_ = pto.pintlv_b32 %pE, %pO    : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>
pto.vstsx2 %se, %so, %y_ub[%c0], "INTLV_B32", %pI
```

The author wrote one `plt` and four ops; the EVEN/ODD predicate split (`ppack`),
the per-half predication, and the interleave-on-store (`pintlv`) are all derived.
Budget: 2 vregs (e/o) + 1 (a) live; pregs `%pE,%pO` (`%p` dies after the cvt) → 2
pregs. Per-reg fusion of the `vcvt+vadd` chain would keep it at 1 each.

### 4.6 The b8 ↔ b32 (radix-4) case — no radix-4 predicate op, no UB roundtrip

A natural worry: A5 has only **radix-2** predicate reshape ops (`ppack`/`punpack`,
`pintlv`/`pdintlv`), so a 1↔4 width change (b8↔b32, i.e. `i8/u8 ↔ i32/f32`) seems
to need either a stacked predicate chain or — worse — a store+load roundtrip
through UB (extra latency + scratch). **It does not.** The `TCvt.hpp` reference
mapping does both directions register-resident, because the 1↔4 *lane spread* is
carried by the **data load/store distribution** (`UNPK_B*` / `PK4_B32`) or a
**`vselr` byte-gather**, not by predicate ops. The predicate then only ever
crosses a single radix-2 step.

A5 does expose native radix-4 *data* part modes (`vcvt PART_P0..P3`, the `Part_T`
token family) and the `PK4_B32` packed store ("extract lower 8 bits"), `UNPK_B8`
unpack load, plus `vintlv`/`vselr` register shuffles — these absorb the 4:1 / 1:4.

**b8 → b32 widen** (`cast8to32_1D_NoPostUpdate`): the spread comes from the load
distribution + `vintlv`; the predicate is **one `punpack`**.

```mlir
// data: UNPK_B8 load + interleave-with-zero + radix-4 part convert
%x   = pto.vlds %src[%i], "UNPK_B8"          : !pto.ptr<u8,ub> -> !pto.vreg<256xu8>
%a,%b = pto.vintlv %x, %zero                  : ...            // spread bytes
%o0  = pto.vcvt %a, %p8 {part="P0"}          : -> !pto.vreg<64xi32>
%o1  = pto.vcvt %b, %p8 {part="P0"}          : -> !pto.vreg<64xi32>
// predicate companion: ONE radix-2 bit-extend, b16 -> b32 (NOT stacked, NOT a roundtrip)
%p32 = pto.punpack %p16 {part="LOWER"}        : !pto.mask<b16> -> !pto.mask<b32>
pto.vsts %o0, %dst[..], "NORM_B32", %p32
```

**b32 → b8 narrow** — two register-resident options, pick by destination:

```mlir
// Option 1 (straight to UB): single packed store, predicate stays b32. Cheapest.
pto.vsts %y, %dst[%i], "PK4_B32", %p32        : !pto.vreg<64xi32>, !pto.ptr<i8,ub>, !pto.mask<b32>

// Option 2 (cast32to8_1D, register path): low-byte extract + index byte-gather.
%lo  = pto.vcvt %x, %p32 {part="P0"}          : !pto.vreg<64xi32> -> !pto.vreg<256xi8>
%idx = ...                                     // vci + vmuls 4  -> [0,4,8,...]
%pk  = pto.vselr %lo, %idx                     : gather low bytes contiguous
pto.vsts %pk, %dst[%i], "NORM_B8", %p8         // plain b8 predicate
```

**Cost model entries (not a roundtrip, but not free):**

- one setup vreg each way: a zero-vec for `vintlv` (widen) / an index-vec via
  `vci`+`vmuls` for `vselr` (narrow);
- `vselr` is a gather (heavier than `PK4_B32`) — prefer the `PK4_B32` packed store
  when the destination is contiguous UB;
- a `mem_bar(VST_VST)` on the `vselr` narrow path;
- **no extra UB scratch** in the 1D contiguous fast path.

**When a roundtrip *does* appear:** only if a b8↔b32 value needs a register-resident
*contiguous logical view* and **no** load/store distribution applies (a true
Category-C consumer). Then `.contiguous()` is the store+load, exactly as P4
intends — but the ordinary "load-then-widen" and "narrow-then-store" flows never
hit it. So the radix-4 predicate gap is a **non-problem** for the common mapping;
it only constrains the rare register-resident-contiguous case.

### 4.7 Open issues (predicates)

- **O-P.1** Is `mask<L x G>` a first-class logical type with its own descriptor, or
  is the predicate always *derived* from the data value it governs at lowering?
- **O-P.2** When a data-dependent mask (`vcmp` result) crosses a parity axis, do we
  recompute the compare per parity reg, or `ppack` the b32 result? (Recompute may be
  cheaper than holding variants under the 8-reg budget.)
- **O-P.3** Radix-4 (b8↔b32) is **resolved** (4.6): no radix-4 predicate op and no
  UB roundtrip — the data distribution (`UNPK_B*`/`PK4_B32`) or `vselr` carries the
  1↔4 spread and the predicate crosses a single `punpack`/`ppack`. Remaining check:
  confirm `punpack PART=LOWER`'s LSB-of-2-bit-group semantics is exactly what the
  widened b32 value needs (it is in `cast8to32_1D`, under ZEROING). Document the
  `PK4_B32`-store vs `vselr`-gather choice in the cost model.
- **O-P.4** Predicate pressure should enter the same cost model as vregs (Part 2);
  is a joint 32-vreg/8-preg allocator warranted, or a two-phase one?
- **O-P.5** Half axis (`chistv2`) with *distinct* N0/N1 predicates (vs the current
  shared-predicate assumption) — needed by any caller that filters bins per half.

---

## Part 5 — Grouped 2D operation mapping, the broadcast asymmetry, and the nxVL cost model

Part 1 established the 2D *type/view*. This part asks the operational question:
**does a grouped 2D op force the same extra load/store the radix-4 case avoided?**
The answer is asymmetric and is the reason a cost model is needed.

### 5.1 Reduce is cheap; grouped broadcast is not

- **Grouped reduce** (`vcgadd/vcgmax/vcgmin`) collapses each VLane in **one op per
  reg**, no load/store, no scratch (the M-A / R4a win). Output = one partial per
  VLane at lanes `0,8,16,…`.
- The **reverse — grouped broadcast** (each VLane's partial fanned back across its
  own lanes) has **no native single instruction**. This is the asymmetry you
  flagged. Three realizations, each with a different cost shape:

| Realization | Mechanism | Extra ld/st | Scratch | Throughput | Used by |
|---|---|---|---|---|---|
| **UB roundtrip** | `vsts` partials + `vlds BRC_BLK`/strided reload | +1 st +1 ld (~18c) | UB block | store-bound | MX example + the SPEC softmax example |
| **`vselr` gather** | index reg (`vci`+`vmuls`) + `vselr` (`dst[i]=src[idx[i]]`) | none | 1 index vreg | **~4× lower than INTLV** (gather/permute class) | register fusion |
| **masked recompute** | per-group full reduce / arithmetic under per-group masks | none | mask pregs | more ops, scales with #groups | small group count |

- **Ungrouped** broadcast of a single reduced scalar is *not* in this trap: `vdup`
  (lane→all, register, cheap) avoids any roundtrip. Note, though, that the SPEC's
  own `pto.mi` softmax sample still uses `vcadd` → `vsts` + `vlds BRC_B32` (UB
  roundtrip) rather than `vdup` — so even the ungrouped reduce+bcast is often a
  roundtrip in hand-written code unless `vdup` is chosen deliberately.

### 5.2 So: does grouped 2D have the extra ld/st issue?

- **Reduce direction:** **no** — `vcg*` is one op/reg.
- **Grouped broadcast direction:** **only if you pick the UB-roundtrip
  realization.** `vselr` keeps it register-resident (pay ~4× throughput);
  masked-recompute keeps it register-resident (pay op count/complexity). There is
  **no universally best choice** — it depends on group count, `K`, surrounding
  ops, and whether UB bandwidth or vector-issue is the bottleneck. Hence it must be
  a **cost-model decision with a programmer escape hatch**, not a fixed rule.

### 5.3 Consequence for `reduce + bcast + eltwise` fusion

- **Without register fusion** (today's MX / requirements pattern): store the reduce
  result to UB with a dist mode, reload broadcast, do the eltwise. Simple and
  correct, but **2 extra ld/st per group-step + UB scratch**, and it breaks the
  fused region (the store/load is a Category-C frontier).
- **With register fusion**: the grouped broadcast must be materialized in-register
  → `vselr` (throughput hit) or masked recompute (complexity). The budget model
  (Part 2) interacts: the `vselr` index reg and the per-group masks consume the
  vreg/preg budget, which can itself force a spill.

This is exactly why "register fusion of reduce+bcast+eltwise is not simple": the
bcast leg has no cheap grouped form, so the fuser must *choose*, and the choice
needs numbers.

### 5.4 The nxVL cost model (the vmi cost table)

**Principle.** A vmi op's cost is derived from the ISA cost of the `pto.mi` ops it
lowers to, times fan-out, plus any materialization and setup:

```
cost(vmi_op) ≈ fanout × Σ cost(lowered pto.mi ops)
             + materialization (extra ld/st when a Category-C frontier is crossed)
             + setup (index / zero / mask register construction)
fanout = K (1D, reg-level) ;  grouped ops add a per-VLane factor handled by vcg*
```

**Base ISA latencies** (A5 CA simulator, `Ascend910_9599`; *simulator, not
silicon* — values from SPEC §"Latency and throughput"):

| Class | Representative op | Latency (cyc) | Throughput note |
|---|---|---|---|
| contiguous / unpack / deint / brc / pk load-store | `NORM`,`UNPK_B*`,`DINTLV_B*`,`BRC_B*`,`PK4_B32` | **9** | `vlds` dual-issue, or `vlds`+`vsts` 1+1 |
| interleave store | `INTLV_B*` (`vstsx2`) | **12** | store-bound |
| binary arith | `vadd` f32 | **7** | PIPE_V |
| reduce | `vcadd`/`vcgadd`/`vcmax` | reduction-class¹ | one op per reg |
| broadcast (reg) | `vdup` (lane→all) | reg-op¹ | cheap, no ld/st |
| select/permute | `vselr` | permute/gather-class | **~4× lower thruput than INTLV** (reported) |
| gather / scatter | `vgather2`/`vgatherb`/`vscatter` | **27–28 / ~21 / ~17** | `vgather2` ~0.1 op/cyc |

¹ This SPEC rev does not publish a separate CA cycle for the reduce/`vdup` ops;
treat them as single-issue PIPE_V (~order of the arith latency), one per reg.

**Derived vmi cost table** (per logical value, `K` = reg fan-out):

| vmi op / pattern | lowering | instr / value | extra ld/st | scratch | dominant cost |
|---|---|---|---|---|---|
| `vadd/vmul/vsel/...` (R1) | `K ×` arith | `K` | 0 | 0 | `K×7` |
| `vcvt` 16↔32 (R2 parity) | `2K ×` `vcvt EVEN/ODD` + `ppack` pred | `2K` | 0 | 0 | parity widen |
| `vcvt` 8↔32 (radix-4) | `UNPK/PK4` + `vcvt P0` (+`vselr` narrow) | ~2–3 | 0 (reg path) | index/zero reg | see §4.6 |
| `vlds/vsts` contiguous | `K × 9` | `K` | — | 0 | bandwidth |
| `vlds DINTLV` / `vsts INTLV` | `K ×` (9 / 12) | `K` | — | 0 | INTLV 12c |
| `vreduce` grouped (R4a) | `K ×` `vcgadd` | `K` | 0 | 0 | **cheap** |
| `vreduce` full (R4b) | `(K-1)` `vadd` + `vcadd` | `K` | 0 | 0 | fold chain |
| **grouped `vbr` (the hard one)** | UB roundtrip **or** `vselr` **or** masked recompute | varies | UB option: +2 | index/mask | **decision point** |
| ungrouped `vbr` (scalar) | `vdup` (reg) or `vlds BRC` | 1 | 0 (`vdup`) | 0 | cheap |
| `vgather/vscatter` (C) | gather | — | — | index | 21–28c, ~0.1/cyc |

### 5.5 Surfacing the cost to the programmer

- **Decision log.** When `pto.as` lowers a `reduce+bcast+eltwise` group, it should
  emit a one-line note of which broadcast realization it picked
  (`ub-roundtrip` / `vselr` / `masked`) and why (the dominating cost term), so the
  choice is auditable.
- **Escape hatch.** Keep the RFC §6.1 `prefer_layout` hint as the override
  (`hint = "contiguous" | "interleaved"`, plus a `bcast = "vselr" | "ub" | "mask"`
  variant) — semantically inert, so a programmer who knows the bottleneck can pin
  the realization without touching correctness.
- **Per-op annotations.** Expose the §5.4 derived cost at the vmi level (an
  attribute or query) so the programmer can reason *before* lowering, mirroring how
  they reason about `__VEC_SCOPE__` instruction counts today.

### 5.6 Open issues (grouped 2D / cost model)

- **O-G.1** Grouped-broadcast realization thresholds: when does `vselr`'s ~4×
  throughput hit beat the UB roundtrip's +2 ld/st + scratch + sync? Calibrate vs
  group count, `K`, and whether the loop is UB- or issue-bound.
- **O-G.2** Add a first-class `vmi.group_bcast` op (so the realization is one
  cost-model switch), or always decompose at lowering?
- **O-G.3** Cost-model calibration: the numbers are simulator, not silicon, and per
  SOC. How is the vmi table kept in sync with ISA-rev latency updates (generated
  from the SPEC table)?
- **O-G.4** Does the decision log / `prefer_layout` `bcast` hint belong on the op,
  the region, or a kernel-level policy attribute?

---

## Part 6 — Consolidated open issues (new, for discussion)

Carried from the parts above, plus the cross-cutting ones:

1. **2D as view vs type** (Part 1.3) and the UB-tile mapping token (O-2D.4) — the
   highest-leverage decision; it shapes load, reduce, and broadcast lowering.
2. **Budget thresholds and joint vreg/preg modeling** (O-B.1, O-B.3, O-P.4).
3. **Fusion descriptor granularity** — does `VPTOFuse` need full layout or just
   axis-kinds (O-F.1)?
4. **Predicate as type vs derived** (O-P.1) — determines whether the surface ever
   spells a logical mask shape.
5. **Op-table single-source** for MLIR + C++ headers (O-F.3) — to keep Option A and
   Option C from drifting (the MoE analysis §6 "axis enumeration drift" risk).
6. **Grouped-broadcast realization** (O-G.1/O-G.2) — the only grouped-2D op with no
   cheap native form; the cost model must pick UB-roundtrip vs `vselr` vs masked
   recompute, and this is the gating difficulty for register-fusing
   `reduce+bcast+eltwise`.
7. **nxVL cost-model sourcing** (O-G.3) — generate the §5.4 vmi cost table from the
   SPEC latency table so it tracks ISA revisions and SOC, and decide how/whether to
   surface per-op cost + a decision log to the programmer (O-G.4).

These are intentionally left open; the next discussion pass should pick a position
on (1) and (4) first, since the load mapping and predicate-type choices gate the
worked examples in Parts 1.4 and 4.5.

