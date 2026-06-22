# PTO Virtual micro Instruction (`pto.vmi`) — Operation List, Layout & Cost Mapping

- v0.1: Doc init. Grouped op surface with nxVL 1D/2D layout support, group-level
  `pto.vmi -> pto.mi` mapping, and a simple instruction-count / VL-utilization /
  cycle-budget cost model per op.

**Status:** draft / RFC companion. This doc is the *operation reference* for
`pto.vmi`. It enumerates the op surface, each op's datatype + 1D/2D layout
support, a **group-level** (not exhaustive) lowering to `pto.mi`, and a simple
cost model. It deliberately does **not** re-derive the philosophy, compile
principles, or lowering rules — those live in:

- [pto-vmi-requirements.md](./pto-vmi-requirements.md) — philosophy (P1–P8),
  `pto.as` contract (C1–C7), rules (R1–R6), Category A/B/C, the nVL register
  type (1D / ragged / 2D), and the open option spaces.
- [pto-vmi-design-exploration.md](./pto-vmi-design-exploration.md) — the 2D
  view vs type question, register/predicate budgets, fusion placement, the
  predicate companion model, and the nxVL cost model (Part 5.4).
- [PTO-micro-Instruction-SPEC.md](./PTO-micro-Instruction-SPEC.md) — the
  `pto.mi` target ISA this surface lowers to (per-`pto.mi`-op semantics,
  pseudocode, and the A5 simulator latency tables).

Whenever this doc says "Category A/B/C", "R1–R6", "P1–P8", or "axis", it means
the definitions in `pto-vmi-requirements.md`. Whenever it cites a cycle latency,
it means the A5 `Ascend910_9599` CA simulator numbers from the micro-SPEC
(*simulator, not silicon*).

[toc]

---

## Part 0 — How to read this doc

The surface is split into **10 op groups** (the grouping you reason about when
writing kernels): load/store, dup/index-gen, eltwise, broadcast, full reduce,
VLane group reduce, grouped reduce+broadcast (2D), convert, SFU, and predicate.

Each group is documented with **two tables**:

1. a single **Group Index** table (Part 3) gives a one-row overview of every
   group — its member ops, Category, layout role, and 1D/2D relevance;
2. a **per-group Detail table** (Parts 4–13) gives one row per op with the
   functional description, datatypes, layout support, the group-level
   `pto.vmi -> pto.mi` mapping, and the cost model.

Layout/cost-interesting groups (reduce, group reduce, grouped broadcast, cvt,
SFU histogram, predicate companion) additionally carry 1–2 worked mapping
examples. Part 16 gives a per-group **`pto.vmi` ↔ `pto.mi` coverage** table
(how many `dist`/`part` modes each group hides and whether the surface fully
covers `pto.mi`), and the Appendix (Part 17) has a consolidated cost cheat-sheet.

---

## Part 1 — Hardware constants (A5 vector pipe)

These set every fan-out and VL-utilization number in the cost model.

```
vector register file : 32 architectural vregs, 256 B (2048 bit) each
predicate file       : 8  architectural pregs, 256 bit each, 1 bit controls 1 byte
VLane                : 32 B sub-lane; 8 VLanes per vreg
E_v = 32 / sizeof(T) : lanes per VLane     (f32 -> 8, f16/bf16 -> 16, i8 -> 32)
VL_B32 = 64, VL_B16 = 128, VL_B8 = 256     : logical lanes per vreg by element width
K   = ceil(L * bitwidth(T) / 2048)         : physical vregs backing one logical L x T
```

Budget targets (design-exploration Part 2): working set `<= ~24` vregs and
`<= ~6` pregs inside a fusion region. Core profile (requirements P3):
`K <= 4`, compile-time-known, fully-unrolled fan-out.

Representative A5 simulator latencies (micro-SPEC §"Latency and throughput"):

| Class | Op(s) | Latency (cyc) |
|---|---|---|
| contiguous / unpack / deint / brc / pk load-store | `NORM`,`UNPK_B*`,`DINTLV_B*`,`BRC_B*`,`PK4_B32` | **9** |
| interleave store | `INTLV_B*` (`vstsx2`) | **12** |
| binary arith | `vadd` f32 | **7** |
| reduce / group-reduce / dup | `vcadd`/`vcgadd`/`vdup` | ~7 (PIPE_V, 1 op/reg) |
| select / permute | `vselr` | permute-class, **~4×** lower thruput than INTLV |
| gather / scatter | `vgather2`/`vgatherb`/`vscatter` | **27–28 / ~21 / ~17** |

Issue note: `vlds` is dual-issue (two loads/cycle) **or** pairs `1+1` with one
`vsts`. So at a UB boundary a store+reload pair often costs ~1 issue slot, not 2.

---

## Part 2 — nxVL layout primer (1D vs 2D)

A `pto.vmi` value is logical; its `#layout` descriptor (requirements §5.4) maps
it onto the `reg ⊃ vlane ⊃ lane` hierarchy. Two surface shapes:

**1D — `!pto.vmi.vreg<L x T>` (default).** `L` logical elements, backed by `K`
physical vregs. Every elementwise op (R1) is layout-agnostic here. This is the
common case and the one with P6 1:1 degeneracy at `K = 1`.

**2D — `!pto.vmi.vreg<R x C x T>` (view, storage stays 1D).** A reinterpreting
view (design-exploration §1.3) consumed by group-aware ops. The axes bind to
levels:

```
logical [R, C]                physical hierarchy
  C (inner / col)  ───────►   lane   level   (within a VLane; require C * sizeof(T) % 32 == 0)
  R (outer / row)  ───────►   vlane  level   (0..7), overflowing to
                              reg    level   (the K fan-out)
```

The view is produced either by the L1 sugar `{group = C}` on a reduce/bcast, by
a zero-cost `group_view {rows=R, cols=C}` cast, or by a `#pto.vmi.tile`
descriptor on the load. The closed tile presets (design-exploration §1.4.1):

| Preset | `(col_order, col_vl, row_level)` | Use |
|---|---|---|
| `BLOCK_ROWS` | `normal, 1, vlane` | M-A: row-into-VLane (softmax row, MX block) |
| `DINTLV_COLS` | `intlv, 2, vlane` | M-B: parity-split cols (KV pairs, MX window) |
| `BLOCK_2x2` | `normal, 2, vlane` | 2-col × 2-row tiles |

Rules that constrain 2D (design-exploration §1.3.1): group size is a 32B
multiple (whole VLanes, never mid-VLane); dynamic/data-dependent masking stays
**1D-only**; only static descriptor-derived pad masks are allowed inside a 2D op.

In the Detail tables, the **Layout** column states each op's relationship to
these shapes:

- **1D pass-through** — layout flows through unchanged (R1 elementwise).
- **2D-native** — the op only makes sense on the 2D view (group reduce).
- **2D-aware** — works 1D, with a 2D `{group=C}` form (reduce, broadcast).
- **produces axis / consumes axis** — Category-B layout transitions (cvt
  parity/width, store interleave, histogram half).

---

## Part 3 — Cost-model conventions & mapping legend

**Cost model (per logical value).** Following design-exploration §5.4:

```
cost(vmi_op) ≈ fanout × Σ cost(lowered pto.mi ops)
             + materialization (extra ld/st when a Category-C frontier is crossed)
             + setup (index / zero / mask register construction)
fanout = K  (1D, reg-level);  grouped ops add a per-VLane factor handled by vcg*
```

Each Detail row's **Cost** cell carries three light fields:

- **`#mi`** — number of `pto.mi` ops emitted per logical value, as a function of
  `K` (e.g. `K`, `2K`, `K-1+1`, `1/reg`). This is the "how many instructions
  mapped to pto.mi" you asked for.
- **VL-util** — fraction of the 256B vreg width doing useful work after layout.
  `100%` = contiguous full reg; `C/E_v` = a `C`-wide row padded to a VLane;
  `1/E_v` = a single reduced partial sitting alone in a VLane.
- **note** — dominant cycle term and/or budget pressure (extra vreg/preg live,
  or a Category-C materialization).

**Mapping shorthand legend** (used in the `pto.vmi -> pto.mi` column):

| Shorthand | Meaning |
|---|---|
| `K × op` | `op` emitted once per physical reg (R1 fan-out) |
| `1/reg op` | one `op` per reg, but it is the whole lowering (e.g. `vcgadd`) |
| `(K-1)× vadd + vcadd` | fold chain then one reduce (R4b fold-then-reduce) |
| `+ ppack` / `+ pdintlv` | predicate companion op also emitted (Part 4 design) |
| `EVEN/ODD`, `P0..P3` | parity (radix-2) / radix-4 part modes on `vcvt` |
| `[C]` | Category-C frontier → `.contiguous()` (store+reload) may be inserted |

`#mi` counts are the **straight-line, fully-unrolled** count under the P3 core
profile (`K <= 4`); a programmer who serializes the fan-out trades these for a
bounded loop (design-exploration Part 2, lever 2).

### Mask / predicate input convention

Each Detail table has a **Mask in** column stating whether the op takes a
predicate operand **at the `pto.mi` target level** (not whether `pto.vmi` lets
you *write* one). Values:

| Mask in | Meaning |
|---|---|
| `no` | The `pto.mi` op has **no** predicate operand at all. |
| `Pg` | Optional governing predicate; honors a **predication mode** (ZEROING or MERGE, see below); tail/data-dependent mask. |
| `Pg req` | A predicate is a **required semantic operand** (e.g. `vsel` selector, `pand` governing, `vcmp` `%seed`). |
| `gen` | The op **produces** a predicate and takes no input mask (`pset`/`pge`/`plt`). |
| `pred` | The operand itself **is** a predicate being reshaped; there is no separate governing `Pg` (`ppack`/`punpack`/`pintlv`/`pdintlv`). |

**Predication mode: ZEROING vs MERGE.** A `Pg`-carrying op resolves its inactive
lanes in one of two modes; `pto.vmi` exposes the mode as an attribute on the op
(`{pmode = "zeroing" | "merge"}`, default `zeroing`):

| Mode | Inactive lane `i` (where `Pg[i] = 0`) gets | Use |
|---|---|---|
| **ZEROING** (native) | `dst[i] = 0` | the default; tail lanes / masked-out lanes cleared |
| **MERGE** | `dst[i] = dst_old[i]` (destination preserved) | accumulate-in-place, partial-update, conditional write-back |

- **A5 — MERGE is emulated, not native.** The A5 vector pipe only predicates in
  **ZEROING** mode. `pto.as` realizes a MERGE-mode op as: compute the op under
  `Pg` (zeroing) to get `new_z` (active lanes set, inactive = 0), select the old
  destination on the **complement** of `Pg`, then OR the two disjoint halves:

  ```mlir
  // MERGE emulation on A5:  dst = Pg ? op(...) : dst_old
  %npg   = pto.vmi.pnot %pg                         // complement predicate
  %new_z = pto.vmi.<op> %a, %b, %pg                 // ZEROING: inactive -> 0
  %old_z = pto.vmi.vand %dst_old, %npg_mask         // keep old on inactive lanes
  %dst   = pto.vmi.vor  %new_z, %old_z              // disjoint OR -> merged
  ```

  Practically this is the **`pnot` + `vor`** idiom (with a `vand`/`vsel` to gate
  the old value): one extra predicate op, one extra logical op, and the old
  destination must stay live. Some chains fold the gate into a single `vsel
  %pg, %new, %dst_old` where a select is cheaper than `vand`+`vor`.

- **A6 — some ops support MERGE natively.** On A6 a subset of `Pg`-carrying ops
  take the merge mode directly (inactive lanes read-modify-preserve the
  destination in one instruction), so the `pnot`+`vor` emulation collapses to the
  single predicated op. `pto.as` selects the native form when the target is A6
  and the op is in the merge-capable set; otherwise it falls back to the A5
  emulation. (The exact A6 merge-capable op set is target-table data, kept with
  the per-SOC ISA tables, not duplicated here.)

**Cost of MERGE (A5).** Add to the op's base cost: **+1 `pnot`** (predicate, once
per distinct `Pg`), **+1 `vor`** (or `+1 vsel`) **per reg** (`+K` ops), and
**+1 vreg** live for `dst_old` (and the complement preg if not recomputed). The
Detail tables cost the **ZEROING** (default) form; a MERGE-mode use adds this
delta on A5 and is ~free on A6 where native. Prefer ZEROING wherever the inactive
result is genuinely don't-care (tails that get re-masked or never stored).

**The load subtlety (the fix that motivated this column).** On A5, **every load
is unpredicated** — `vlds`, `vldsx2`, `vldas`, `vldus` have **no** `%mask`
operand. Therefore a logical tail/predicate that a `pto.vmi` author associates
with a load is **never** lowered to a "masked load". `pto.as` realizes it one of
three ways instead:

1. apply the predicate at the **consuming compute op** (which is predicated),
   e.g. `vlds` then `vsel %mask, %x, pad` / a predicated `vadd` (this is the
   `ComputeSoftmax` pattern, requirements A.7);
2. carry it to the **store** (`vsts`/`vstsx2` *are* predicated) so only valid
   lanes are written;
3. shorten the **load length** / use `plt` tail bookkeeping so the out-of-range
   lanes are never loaded.

Stores are the opposite: `vsts`/`vstsx2` and `vscatter` **do** take `%mask`, so
a tail/predicate naturally lands there. This asymmetry (unpredicated load,
predicated store) is why the `pto.vmi -> pto.mi` mapping for a "predicated load"
shows the mask migrating to a consumer or the store, never onto the load.

---

## Part 4 — Group Index (Level-1)

One row per group. "Category" is the dominant requirements Part-4 category;
"Layout role" summarizes what the group does to the descriptor; "1D/2D" states
where the 2D view matters. Per-op predicate support is in the **Mask in** column
of each Detail table (legend in Part 3); at a glance: **loads, `vbr`, `vci`,
`vselr`, sort, and `pset`/`pge`/`plt` take no input mask**, while stores,
gather/scatter, and all compute/reduce/cvt ops carry a governing `Pg`.

| # | Group | Member `pto.vmi` ops | Category | Layout role | 1D/2D |
|---|---|---|---|---|---|
| 1 | **Load / Store** | `vlds`, `vsts` | A (+B on interleave) | contiguous; consumes parity/half/width on store; carries `#pto.vmi.tile` on 2D load | 1D + 2D-native (tile) |
| 2 | **Dup / index-gen** | `vdup`, `arange`/`vci` | A / R6 | produces `broadcast` or index value; replicate-read | 1D + 2D-aware (row replicate) |
| 3 | **Eltwise compute** | `vadd vsub vmul vdiv vmax vmin vand vor vxor vshl vshr`, `vadds vmuls …`, `vabs vneg vexp vln vsqrt vrelu vnot`, `vcmp vcmps vsel vselr` | A | pass-through; broadcast operands via R6 | 1D pass-through |
| 4 | **Broadcast** | `vbr` | B / R6 | produces `broadcast` axis (1-reg backing) | 1D + 2D-aware (`{group=C}`) |
| 5 | **Full reduce** | `vreduce_add vreduce_max vreduce_min`, `vcpadd` | B (R4a) / search (R4b) / C (R4d) | consumes `reduce` axis; arg-offset R4c | 1D + 2D-aware |
| 6 | **VLane group reduce** | `vcgadd vcgmax vcgmin` | B (R4a) | consumes `lane`-level axis, one op/reg | **2D-native** |
| 7 | **Grouped reduce + bcast (2D)** | `vreduce_* {group}` + `vbr {group}` | B + (B/C bcast) | reduce cheap; grouped bcast = cost-model decision | **2D-native** |
| 8 | **Convert** | `vcvt` (widen/narrow), `vtrc` | B | produces/consumes `parity(r)` / `width(r)` | 1D + axis-producing |
| 9 | **SFU** | `chistv2`, `vbitsort vmrgsort4`, `vgather2 vgatherb vscatter`, `vexpdif vaxpy vlrelu vmull vmula` | B (`chistv2`) / C (sort/gather) / A (fused) | `chistv2` produces `half`; gather/sort are Category-C | mixed |
| 10 | **Predicate ops** | `pset pge plt`, `ppack punpack`, `pintlv pdintlv`, `pand por pnot` | companion | derive the predicate companion of a data-layout change; 8-preg budget | follows governed value |

---

## Part 5 — Group 1: Load / Store

The `pto.vmi` surface keeps load/store **logical**: the author writes
`vlds`/`vsts` of a 1D value or a 2D tile, and `pto.as` selects the `dist` token
(`NORM`, `DINTLV_B*`, `INTLV_B*`, `BRC_*`, `UNPK_B*`, `PK4_B32`). The
interleaved `vldsx2`/`vstsx2` forms and all `dist`/`part` tokens are
lowering-only (requirements §4.2). On a 2D load, the `#pto.vmi.tile` descriptor
(Part 2) commits the `(row,col)->(reg,VLane,lane)` mapping so downstream fused
reduce/broadcast can read it.

| `pto.vmi` op | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `vlds` (1D contig) | UB→vreg contiguous load | i8–i64, f16/bf16, f32 | 1D pass-through | `no` | `K × vlds NORM` | `#mi=K` / 100% / 9c each; dual-issue |
| `vlds` (2D tile) | UB→vreg load with tile mapping | i8–i32, f16/bf16, f32 | 2D-native (`#pto.vmi.tile`) | `no` | `K × vlds` with `NORM` (`BLOCK_ROWS`) or `DINTLV_B*` (`DINTLV_COLS`); pad tail lanes with reduce identity | `#mi=K` / `C/E_v` (row<VLane) / 9c; DINTLV K=2 |
| `vlds` (broadcast) | scalar/block replicate load | i8–i32, f16/bf16, f32 | produces `broadcast` (R6) | `no` | `1 × vlds BRC_B*` / `BRC_BLK` | `#mi=1` / replicate-read / 9c; no fan-out |
| `vlds` (tail/partial) | logically-predicated load (tail) | as above | 1D + tail predicate | `no` (load) | `K × vlds` (unpredicated); the logical mask migrates to the **consumer** / **store**, or shortens load length (Part 3) | `#mi=K` / `len/L` / 9c; predicate not on the load |
| `vsts` (1D contig) | vreg→UB contiguous store | i8–i64, f16/bf16, f32 | 1D pass-through | `Pg` | `K × vsts NORM_B*` | `#mi=K` / 100% / 9c; pairs 1+1 with `vlds` |
| `vsts` (interleave) | store consuming a parity/half axis | i8–i32, f16/bf16, f32 | consumes `parity`/`half`/`width` (R3) | `Pg` | `K × vstsx2 INTLV_B*` (+ `pintlv` companion) | `#mi=K` / 100% / **12c** (INTLV); `r=4`→2 stages |
| `vsts` (packed quant) | narrow-on-store distribution | fp8, int8, int4 | consumes `width` (R3) | `Pg` | `vsts PK4_B32` (int4: `i*VL/2` addr + packed mask) | `#mi=K` / 100% / 9c; no register pack |
| `vldas`/`vstas`/`vstar` | alignment-state load/store helpers | i8–i32, f16/bf16, f32 | 1D (align carrier) | `no` | alignment-state pipe ops | `#mi=K` / — / 9c; unpredicated |

**Notes.** `vlds`/`vsts` of a contiguous `K=1` value is the P6 1:1 case — exactly
one `pto.mi` op, zero overhead. The interleave/deinterleave is never user-spelled;
it is the R3 lowering of a store/load that crosses a `parity`/`half`/`width` axis,
and it drags the matching predicate companion (`pintlv`/`pdintlv`, Group 10).
**Loads are unpredicated** (Part 3): the "tail/partial" row carries no mask on the
`pto.mi` load; the predicate is realized at a consumer, at the store, or by load
length. The predicated stores are `vsts`/`vstsx2`/`vscatter`; the alignment-state
stores `vstas`/`vstar` are unpredicated.

---

## Part 6 — Group 2: Dup / index-gen

Replication and index materialization. These produce a `broadcast` axis (1-reg
backing, replicate-read under R6) or an index vector, and never expand into `K`
stored copies until a Category-B/C edge needs the expanded form.

| `pto.vmi` op | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `vdup` | replicate a lane/value across lanes (or rows in 2D) | i8–i32, f16/bf16, f32 | produces `broadcast`; 2D-aware (row replicate) | `Pg` | `1 × vdup` (reg); 2D row-replicate = replicate-read across VLanes | `#mi=1` (+K replicate-reads) / 100% / cheap, no ld/st |
| `arange` / `vci` | generate `[base, base±i]` lane indices | i8–i32, f16, f32 | produces index value (Category A) | `no` | `1 × vci {ASC/DESC}` per chunk | `#mi=1`/chunk / 100% / cheap; reused via R6 |

**Notes.** `vdup` is the cheap ungrouped broadcast (lane→all, register-resident,
no UB roundtrip) — prefer it over a UB `BRC` reload when broadcasting a single
reduced scalar. The 2D "replicate one value per row across its VLane" still uses
`vdup`/replicate-reads; the **grouped** broadcast (each VLane's own partial fanned
back over its lanes) is the hard case and lives in Group 7, not here.

---

## Part 7 — Group 3: Eltwise compute

Pure Category-A, per-lane ops. Layout passes through unchanged; an operand whose
cardinality along an axis is 1 is a broadcast (R6) and is replicate-read. Under
the P3 core profile these fan out as fully-unrolled straight-line code.

| `pto.vmi` op(s) | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `vadd vsub vmul vdiv vmax vmin` | binary arithmetic | i8–i32, f16/bf16, f32 (`vdiv` f16/f32) | 1D pass-through | `Pg` | `K × <op>` | `#mi=K` / 100% / ~7c each |
| `vand vor vxor vnot` | bitwise | i8–i32 (bit-typed) | 1D pass-through | `Pg` | `K × <op>` | `#mi=K` / 100% / ~7c |
| `vshl vshr` | element shift (vector count) | i8–i32 | 1D pass-through | `Pg` | `K × <op>` | `#mi=K` / 100% / ~7c |
| `vadds vmuls vmaxs vmins vshls vshrs` | vec-scalar (scalar implicit broadcast) | i8–i32, f16/bf16, f32 | 1D pass-through; scalar = R6/M2 | `Pg` | `K × <op>s` | `#mi=K` / 100% / ~7c; no extra reg for scalar |
| `vabs vneg vrelu` | unary arithmetic / activation | i8–i32, f16/bf16, f32 | 1D pass-through | `Pg` | `K × <op>` | `#mi=K` / 100% / ~7c |
| `vexp vln vsqrt` | unary transcendental | f16, f32 | 1D pass-through | `Pg` | `K × <op>` | `#mi=K` / 100% / transcendental-class |
| `vcmp vcmps` | compare → predicate mask | i8–i32, f16/bf16, f32 | 1D; produces a `mask` (Group 10) | `Pg req` (`%seed`) | `K × vcmp(s)` → `mask` | `#mi=K` / 100% / ~7c; +1 preg per live mask |
| `vsel` | predicate select between two vecs | i8–i32, f16/bf16, f32 | 1D pass-through | `Pg req` (selector) | `K × vsel` | `#mi=K` / 100% / ~7c |
| `vselr` | register gather/permute select | i8–i32, f16/bf16, f32 | 1D; permute (used by grouped bcast) | `no` (uses `%idx`) | `K × vselr` (+ index reg setup) | `#mi=K` / 100% / **~4×** lower thruput; +1 index vreg |

**Notes.** Everything here is `K`-fan-out with VL-util `100%` and no
materialization — the register-fuse sweet spot (design-exploration §6.1: fuse
elementwise chains per-reg). `vselr` is the exception: it is the permute/gather
class and is the register-resident realization of a grouped broadcast (Group 7),
so its cost is called out even though it is surface-eltwise.

---

## Part 8 — Group 4: Broadcast

`vbr` is the logical scalar→vector / reduced→fanned broadcast (R6). The
**ungrouped** form (a single reduced scalar fanned over the whole value) is
cheap; the **grouped** form (per-VLane partial fanned back over its own lanes) is
the asymmetric hard case and is fully treated in Group 7.

| `pto.vmi` op | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `vbr` (ungrouped) | broadcast a length-1 / lane-0 value over `L` | i8–i32, f16/bf16, f32 | produces `broadcast`; 1-reg backing | `no` | `1 × vdup` (reg) **or** `vsts`+`vlds BRC_*` (UB) | `#mi=1` (`vdup`) / replicate-read / cheap, no ld/st if `vdup` |
| `vbr` (grouped, `{group=C}`) | broadcast each VLane partial across its `C` lanes | i8–i32, f16/bf16, f32 | **2D-native**; consumes per-VLane partial | `no`* | UB-roundtrip `BRC_BLK` / `vselr` / masked-recompute → see Group 7 | varies / `C/E_v` / **decision point** |

**Notes.** Fused reduce→broadcast (`vcadd`+`vdup`) is the recognized R4/M2
fusion: `pto.as` emits them back-to-back and keeps the result as a `broadcast`
axis rather than materializing `K` copies. Prefer `vdup` over a UB `BRC` reload
for a single scalar (design-exploration §6.1, RoT for full-reduce→scalar-bcast).
`vbr` itself takes **no** predicate; `*` the *masked-recompute* realization of a
grouped broadcast (Group 7) introduces per-group `Pg` masks at lowering, but the
`vbr` op surface has no mask operand.

---

## Part 9 — Group 5: Full reduce

A reduction over the whole 1D value, or over a 2D row (`{group}` → Group 6/7).
R4 gives three realizations keyed off how the reduce axis sits in the hierarchy;
`pto.as` picks by `K`, ILP, and whether an arg-index is needed (M1).

| `pto.vmi` op | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `vreduce_add` | sum-reduce to a scalar / partial | i8–i32 (widening), f16, f32 | 1D consume `reduce`; 2D-aware | `Pg` | R4b fold: `(K-1)× vadd + 1× vcadd`; or R4b partial: `K× vcadd + combine` | `#mi≈K` / `1/L` result / fold = serial dep; partial = better ILP |
| `vreduce_max` / `vreduce_min` | max/min reduce | i16–i32, f16, f32 | 1D consume `reduce`; 2D-aware | `Pg` | `(K-1)× vmax + 1× vcmax` (value); arg form adds R4c index offset `+k·lanes` | `#mi≈K` / `1/L` (`2/L` w/ index) / arg needs index combine |
| `vreduce_*` (R4a, `{group=E_v}`) | VLane-aligned group reduce | as above | **2D-native** | `Pg` | `1/reg vcgadd/vcgmax/vcgmin` | `#mi=K` / `R/E_v` result / **cheapest**; no combine |
| `vreduce_*` (R4d, unaligned) | sub-VLane / unaligned reduce | as above | 1D, Category C | `Pg` | `[C]` materialize then reduce / rotate-shuffle | `#mi=K+ld/st` / low / **last resort** |
| `vcpadd` | inclusive prefix sum (scan) | f16, f32 | 1D pass-through (per-reg) | `Pg` | `K × vcpadd` (+ carry combine across regs) | `#mi≈K` / 100% / cross-reg carry serial |

**Notes.** R4a (VLane-aligned, the `{group=E_v}` case) is the bridge to Group 6:
when the reduce axis is exactly the 32B VLane it is one `vcg*` per reg, no
cross-reg combine, no materialization. R4b is the 1D whole-array reduce and is a
*local* fold-vs-partial instruction-selection choice over the `K` regs (no loop
synthesis). R4c arg-reduce must inject `k · lanes_per_reg` into reg-`k` indices
before combining so the global argmin/argmax matches logical order.

---

## Part 10 — Group 6: VLane group reduce

The native per-32B-VLane reduce (R4a). This is the 2D-native primitive: each row
of the 2D view occupies one VLane, and one instruction collapses all 8 VLanes of
a reg at once — the high-VL-utilization win over padding each row to a full reg.

| `pto.vmi` op | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `vcgadd` | sum within each VLane | i16–i32, f16, f32 | **2D-native** (`col`=lane, `row`=vlane) | `Pg` | `1/reg vcgadd` | `#mi=K` / inputs 100%, output `R/E_v` / one op/reg, no ld/st |
| `vcgmax` | max within each VLane | i16–i32, f16, f32 | **2D-native** | `Pg` | `1/reg vcgmax` | `#mi=K` / same / no materialize |
| `vcgmin` | min within each VLane | i16–i32, f16, f32 | **2D-native** | `Pg` | `1/reg vcgmin` | `#mi=K` / same / no materialize |

**Notes.** Output is one partial per VLane (at lanes `0,8,16,…` for f32),
i.e. `R = 8K` partials. `C < E_v` rows must be padded with the reduce identity
(0 for add, ±INF for max/min) and the pad is a static descriptor-derived mask
(allowed in 2D; design-exploration §1.3.1). Cross-VLane (R-axis) reduce is *not*
a `vcg*` — it is a second fold step or Category C (open issue O-2D.3).

---

## Part 11 — Group 7: Grouped reduce + broadcast (2D)

The most cost-sensitive group. Grouped **reduce** is cheap (Group 6, one `vcg*`
per reg). Grouped **broadcast** — fanning each VLane's own partial back across
its `C` lanes — has **no native single instruction** (design-exploration §5.1).
The realization is a cost-model decision, not a fixed rule.

| Pattern | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| grouped reduce (`vreduce_* {group=C}`) | per-VLane block reduce | i16–i32, f16, f32 | 2D-native | `Pg` | `1/reg vcg*` | `#mi=K` / `R/E_v` out / cheap |
| grouped bcast — **UB roundtrip** | store partials, reload block-broadcast | i8–i32, f16/bf16, f32 | 2D-native | `Pg` (store) | `vsts` partials + `vlds BRC_BLK` | `#mi=2K` / `C/E_v` / **+2 ld/st** (9c each, pair 1+1) + UB scratch |
| grouped bcast — **`vselr` gather** | index reg + register permute | i8–i32, f16/bf16, f32 | 2D-native, register-resident | `no` | `vci`+`vmuls` (index) + `K× vselr` | `#mi=K+setup` / `C/E_v` / **~4×** lower thruput; +1 index vreg; no ld/st |
| grouped bcast — **masked recompute** | per-group arithmetic under masks | i8–i32, f16/bf16, f32 | 2D-native, register-resident | `Pg req` | per-group full op under `PAT_VLn` masks | `#mi∝#groups` / `C/E_v` / +mask pregs; scales with group count |
| fused `reduce+bcast+eltwise` | per-row normalize (softmax/MX) | f16/bf16, f32 | 2D-native | `Pg` | `vcg*` + (one bcast realization) + `K× eltwise` | see chosen bcast row + `K×7c` eltwise |

**Decision guide (RoT-1, design-exploration §6.2).** Default to the **UB
roundtrip** for grouped `reduce+bcast+eltwise`: the `vsts`+`vlds BRC_BLK` are two
9-cycle ops that pair on the `vlds`+`vsts` (1+1) issue slot and free registers
for the eltwise. Use **`vselr`** only when group count and `K` are both tiny
*and* the loop is vector-issue-bound (not UB-bandwidth-bound). Use **masked
recompute** only for very small group counts. `pto.as` should emit a one-line
decision-log note of which realization it picked and why, and honor a
`prefer_layout {bcast = "ub"|"vselr"|"mask"}` override (semantically inert).

**Why no cheap native form.** Reduce collapses lanes into one slot (`vcg*` does
this directly); the reverse spread has no single op, so it must round-trip
through UB, gather (`vselr`), or recompute. This asymmetry is the gating
difficulty for register-fusing `reduce+bcast+eltwise` and is exactly why this
group needs the cost model rather than a fixed lowering.

---

## Part 12 — Group 8: Convert (cvt)

One logical `vcvt` whose *target dtype is the layout*. `pto.as` expands it into
the dtype-specific cast chain + part/width staging + matching store
distribution, and drags the predicate companion (Group 10). The author never
spells `EVEN/ODD`, `P0..P3`, `Pack`, `int4x2_t`, or `VL/2` addresses.

| `pto.vmi` op (form) | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `vcvt` 16↔32 (radix-2) | widen/narrow f16/i16 ↔ f32/i32 | f16/bf16↔f32, i16↔i32 | produces/consumes `parity(r=2)` | `Pg` | `2K × vcvt EVEN/ODD` + `ppack`/`punpack` companion | `#mi=2K` / 100% / one radix-2 step; K grows 1→2 on widen |
| `vcvt` 8↔32 (radix-4) | widen/narrow i8/u8 ↔ i32/f32 | i8/u8↔i32/f32 | produces/consumes `width(r=4)` | `Pg` | widen: `UNPK_B8`+`vintlv`+`vcvt P0` + one `punpack`; narrow: `PK4_B32` store (or `vselr` gather) + one `ppack` | `#mi≈2–3` / 100% / no UB roundtrip in fast path; +zero/index setup reg |
| `vcvt` quant f32→{fp8,int8,int4} | quantized narrow with packed store | f32 → fp8_e4m3, int8, int4 | consumes `width`; store-dist derived | `Pg` | fp8: `1 cast`+`PK4_B32`; int8: 3-stage cast+`PK4_B32`; int4: cast+`Pack`+`int4x2_t`+`PK4_B32` (`i*VL/2`, packed mask) | `#mi=K..3K` / 100% / chain depth by dtype; int4 half-addr derived |
| `vtrc` | truncate/round-convert (mode token) | f16/bf16, f32 → int | 1D pass-through (per-reg) | `Pg` | `K × vtrc` | `#mi=K` / 100% / ~7c |

**Notes.** Radix-4 (b8↔b32) is **not** a stacked predicate chain and **not** a
UB roundtrip: the 1↔4 lane spread rides the data load/store distribution
(`UNPK_B*`/`PK4_B32`) or a `vselr` byte-gather, and the predicate only ever
crosses a single radix-2 step (`punpack`/`ppack`) — design-exploration §4.6. A
true UB roundtrip appears only for a register-resident *contiguous* b8↔b32 view
with no applicable distribution (rare Category-C consumer).

---

## Part 13 — Group 9: SFU ops

Special-function / domain-accelerator ops. Mixed categories: `chistv2` produces
a `half` axis (Category B); sort and gather/scatter are Category-C tile/permute
ops; the fused activation/arith ops are Category-A `vreg→vreg`.

| `pto.vmi` op | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `chistv2` | histogram / per-bin count | i8–i32 (bin index) | **produces `half` axis** (`Bin_N0/N1`) | `Pg` (per-half) | `chistv2 Bin_N0` + `Bin_N1` (two-half fanout) + widen/accumulate | `#mi≈2K` / `half` / per-half predicate; INTLV store to merge bins |
| `vbitsort` | sort 32 (score,index) proposals | f16/f32 score + i32 idx | Category C (UB helper) | `no` | `vbitsort` (UB→UB, `VBS32`) | UB op / n/a / no decomposition; 8B records |
| `vmrgsort4` | merge 4 pre-sorted lists | i16–i32, f16, f32 | Category C (UB helper) | `no` | `vmrgsort4` (+ `get_vms4_sr`) | UB op / n/a / inputs must be pre-sorted |
| `vgather2` / `vgatherb` | indexed gather (B32 / byte) | i8–i32, f16/bf16, f32 | Category C; needs contiguous index table | `Pg` | `vgather2` / `vgatherb` | `#mi=K` / data-dep / **27–28 / ~21c**, ~0.1 op/cyc |
| `vscatter` | indexed scatter | i8–i32, f16/bf16, f32 | Category C | `Pg` | `vscatter` | `#mi=K` / data-dep / **~17c** |
| `vexpdif` | fused `exp(x − max)` (softmax) | in f16/f32 → out f32 | 1D pass-through (EVEN/ODD contract) | `Pg` | `K × vexpdif` | `#mi=K` / 100% / fuses sub+exp |
| `vaxpy` | fused `α·x + y` | f16, f32 | 1D pass-through | `Pg` | `K × vaxpy` | `#mi=K` / 100% / ~arith |
| `vlrelu` / `vprelu` | leaky / parametric ReLU | f16, f32 | 1D pass-through | `Pg` | `K × vlrelu`/`vprelu` | `#mi=K` / 100% / ~arith |
| `vmull` | widening 32×32→64 multiply (hi/lo) | i32/u32 | 1D; produces hi+lo pair | `Pg` | `K × vmull` | `#mi=K` / 100% / two result regs |
| `vmula` | fused multiply-accumulate | i8–i32, f16/bf16, f32 | 1D pass-through | `Pg` | `K × vmula` | `#mi=K` / 100% / not always = vmul+vadd |

**Notes.** `chistv2` is the MoE histogram/count primitive: it naturally writes
two halves (`Bin_N0`/`Bin_N1`), so `pto.as` keeps the logical op single and
expands the half fanout + per-half predicate + INTLV merge on store
(design-exploration §3.1, histogram row). Sort and gather/scatter are genuine
Category-C ops — no axis absorbs them, so they sit at fused-region boundaries.

---

## Part 14 — Group 10: Predicate ops

Predicates are a **logical companion** that carries the same `LayoutDescriptor`
axes as the data value it governs (design-exploration Part 4). The author writes
one logical mask over `L`; `pto.as` fans it out and applies, to the predicate,
the *companion* of whatever transform it applied to the data. Budget: **8 pregs**
total, target working set `<= 6`.

| `pto.vmi` op | Functional description | Datatypes | Layout | Mask in | `pto.vmi -> pto.mi` mapping | Cost (`#mi` / VL-util / note) |
|---|---|---|---|---|---|---|
| `pset` | materialize mask from named pattern | b8/b16/b32 | static pattern | `gen` | `1 × pset_b* "PAT_*"` | `#mi=1` / n/a / axis-invariant `PAT_ALL` shared across K |
| `pge` | lane-count / tail pattern mask | b8/b16/b32 | static tail | `gen` | `1 × pge_b* "PAT_VLn"` | `#mi=1` / n/a / cheap to recompute |
| `plt` | data-dependent tail mask (post-update) | b8/b16/b32 | dynamic tail (1D-only) | `gen` (from scalar) | `1 × plt_b*` (+ carry scalar) | `#mi=1` / n/a / +1 preg; recompute > keep |
| `ppack` | radix-2 narrowing pack (parity companion) | b16→b32 view | companion of `vcvt EVEN/ODD` widen | `pred` | `ppack PART=LOWER/HIGHER` | `#mi=1–2` / n/a / keeps LSB of each 2-bit group |
| `punpack` | radix-2 widening unpack (bit-extend) | b16→b32 view | companion of widened value | `pred` | `punpack PART=LOWER/HIGHER` | `#mi=1` / n/a / zero-extend 1-bit→2-bit group |
| `pintlv` / `pdintlv` | interleave / deinterleave predicate | b8/b16/b32 | companion of `vsts INTLV` / `vlds DINTLV` | `pred` | `pintlv_b*` / `pdintlv_b*` | `#mi=1` / n/a / merge/split governing pred |
| `pand` `por` `pnot` | predicate logical ops (gated) | b8/b16/b32 | follows governed value | `Pg req` | `K' × p<op>` | `#mi=K'` / n/a / gated by governing pred |

**Notes.** Full (all-active) masks are axis-invariant: `ppack`/`pdintlv` of a
`PAT_ALL` is still `PAT_ALL`, so a region with no tail and no data-dependent mask
shares **one** `PAT_ALL` across all K regs (preg pressure ≈ 1). Only **tails**
and **`vcmp` results** cost preg budget; the expensive cases are data-dependent
masks (softmax `-inf` pad, top-k filter, MX exponent extract) that must hold a
parity/width companion variant live. A tail `plt` is cheap to recompute — prefer
recompute over keeping many parity-variant copies. There is **no radix-4
predicate op**: a b8↔b32 width change crosses only a single `punpack`/`ppack`
(§4.6). Intermediate mask spills default to the raw `NORM` image (no repack;
design-exploration §4.7).

`pnot` (+ `vor`/`vsel`) is also the building block of **MERGE-mode emulation**
on A5 (Part 3): a merge-mode compute op lowers to the zeroing op plus
`pnot %pg` and a disjoint `vor` (or a `vsel %pg, %new, %dst_old`) to preserve
inactive-lane destinations. On A6, merge-capable ops skip this and take the mode
natively.

---

## Part 15 — Worked mapping examples

These show the group-level `pto.vmi -> pto.mi` mapping for the layout/cost-
interesting groups. The `pto.mi` side is the same instruction stream a careful
hand-author would write (P6).

### 15.1 Full reduce → scalar broadcast (Group 5 + 4, R4b + R6)

```mlir
%m  = pto.vmi.vreduce_max %x            : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<1xf32>
%mb = pto.vmi.vbr %m                    : !pto.vmi.vreg<1xf32>   -> !pto.vmi.vreg<256xf32>
%e  = pto.vmi.vexp (pto.vmi.vsub %x, %mb)
```

Lowering (`K = 4`): `(K-1)=3 × vmax` fold + `1 × vcmax` reduce + `1 × vdup`
broadcast + `K × (vsub, vexp)`. `#mi ≈ (K-1)+1+1 + 2K = 13`. VL-util: 100% on the
eltwise, `1/L` at the reduced scalar. No UB roundtrip (the `vdup` keeps the
broadcast register-resident).

### 15.2 Per-row softmax sum/normalize (Group 6 + 7, R4a + grouped bcast)

```mlir
%t    : !pto.vmi.vreg<R x E x f32>                       // 2D tile, row = VLane
%sum  = pto.vmi.vreduce_add %t {group = E}               // R4a → vcgadd, one op/reg
%sumb = pto.vmi.vbr %sum {group = E}                     // grouped bcast → decision
%norm = pto.vmi.vdiv %t, %sumb                           // R1 fan-out
```

Lowering: `K × vcgadd` (cheap, one op/reg, output `R/E_v` per reg) → grouped
broadcast: **UB roundtrip** default `vsts` partials + `vlds BRC_BLK` (`2K` ops,
9c, pair 1+1) → `K × vdiv`. `#mi ≈ K + 2K + K = 4K`. The grouped-bcast leg is the
cost-model decision (RoT-1: UB roundtrip unless tiny + issue-bound → `vselr`).

### 15.3 Parity widen + bias-add + contiguous store with tail (Group 8 + 1 + 10)

```mlir
%p = pto.vmi.plt %rem                   : i32 -> !pto.vmi.mask<128xb16>
%a = pto.vmi.vlds %x_ub                 : !pto.mi.ptr<i16,ub> -> !pto.vmi.vreg<128xi16>   // unpredicated load
%w = pto.vmi.vcvt %a, %p                : !pto.vmi.vreg<128xi16> -> !pto.vmi.vreg<128xi32>
%s = pto.vmi.vadd %w, %b, %p            : !pto.vmi.vreg<128xi32>
pto.vmi.vsts %s, %y_ub, %p              : !pto.vmi.vreg<128xi32>, !pto.mi.ptr<i32,ub>
```

Lowering: the `vlds` is **unpredicated** (Part 3) — the tail `%p` is not applied
on the load; it first governs the predicated `vcvt`/`vadd` and the store. `vcvt`
→ `vcvt EVEN` + `vcvt ODD` (`2K`) + predicate `ppack LOWER`/`HIGHER` (2 pregs);
`vadd` per parity reg; contiguous store → `vstsx2 INTLV_B32` (12c) + `pintlv`
predicate merge. The author wrote one `plt` and four ops; the EVEN/ODD split,
per-half predication, and interleave-on-store are all derived. Budget: 2 vregs +
2 pregs live (1 each with per-reg fusion).

### 15.4 Multi-dtype quant (Group 8, one `vcvt` vs cast ladder)

```mlir
%xf = pto.vmi.vcvt %x                   : vreg<L x f16> -> vreg<L x f32>
%s  = pto.vmi.vmuls %xf, %scale
%q  = pto.vmi.vadds %s,  %offset
%y  = pto.vmi.vcvt %q                   : vreg<L x f32> -> vreg<L x U>   // U = fp8|int8|int4
pto.vmi.vsts %y, %y_ub
```

Lowering selects the dtype-specific chain + store token: fp8 = `1 cast` +
`PK4_B32`; int8 = `f32→int16→half→int8` + `PK4_B32`; int4 = + `Pack` +
`int4x2_t` cast + `i*VL/2` address + `mask4Int4`. The `if constexpr` dtype ladder
disappears; the half-width address and packed mask are descriptor-derived.

---

## Part 16 — `pto.vmi` ↔ `pto.mi` coverage & dist-mode reduction

This part answers three questions per group: (1) **what `pto.mi` surface the
group's `pto.vmi` ops map onto**, (2) **how many `dist`/`part`/token modes the
`pto.vmi` surface hides** (the readability/safety win — these become
lowering-only, requirements §4.2), and (3) **whether the `pto.vmi` surface is
expressive enough to cover the `pto.mi` group**, with a remark wherever it is
**not** (so a kernel author knows when to drop to `pto.mi`).

"Coverage" legend: **Full** = every A5 `pto.mi` op in the group is reachable
from the `pto.vmi` surface (1:1 or as a fixed lowering); **Partial** = some
`pto.mi` ops/modes have no `pto.vmi` form and must be written in `pto.mi`.

### 16.1 Per-group coverage table

| # | Group | `pto.mi` ops it maps onto | `dist`/`part`/token modes hidden | Coverage | Remark (what is *not* covered) |
|---|---|---|---|---|---|
| 1 | Load / Store | `vlds`,`vldsx2`,`vsts`,`vstsx2`,`vldas/vldus/vstas/vstar/vstus/vstur` | **~32**: 19 load `dist` (`NORM`,`UNPK_B*`,`DINTLV_B*`,`BRC_B*`,`BRC_BLK`,`BDINTLV`,`US_B*`,`DS_B*`,`SPLT4CHN`,`SPLT2CHN_B*`) + 13 store `dist` (`NORM_B*`,`PK_B*`/`PK4_B32`,`INTLV_B*`,`MRG*CHN_B*`) → `vlds`/`vsts` (+`#pto.vmi.tile`) | **Partial** | channel split/merge (`SPLT*CHN`/`MRG*CHN`) and up/down-sample (`US_B*`/`DS_B*`) have **no** `vmi` logical op yet; strided/unaligned residual stores (`vstus`/`vstur`) only via **experimental** R5 squeeze; gather/scatter exposed separately (Group 9, Category C) |
| 2 | Dup / index-gen | `vdup`,`vci`,`vbr` | broadcast `position`(LOWEST/HIGHEST), `vci` `order`(ASC/DESC) | **Full** | — |
| 3 | Eltwise compute | `vadd vsub vmul vdiv vmax vmin vand vor vxor vshl vshr vadds… vabs vneg vexp vln vsqrt vrelu vnot vcmp vcmps vsel vselr` | none (pure per-lane; layout pass-through) | **Full** | carry ops `vaddc`/`vsubc`/`vaddcs`/`vsubcs` are surfaced but their `%carry` predicate is hand-threaded (no logical sugar); `vselrv2` is not A5 |
| 4 | Broadcast | `vbr`,`vdup`, `vlds BRC_*` | `BRC_B*`,`BRC_BLK` → `vbr` | **Full** | grouped broadcast has **no single `pto.mi` op** — realized as UB-roundtrip / `vselr` / masked (Group 7), a cost choice not an expressiveness gap |
| 5 | Full reduce | `vcadd`,`vcmax`,`vcmin`,`vcpadd` | reduce realization (fold vs partial) + R4c index offset | **Full** | `vcmax`/`vcmin` value+index pair surfaces as a 2-element result; arg-offset is auto |
| 6 | VLane group reduce | `vcgadd`,`vcgmax`,`vcgmin` | VLane grouping (the 8×32B structure) → 2D view | **Full** | cross-VLane (R-axis) reduce is **not** a `vcg*` — needs a fold step or Category C (O-2D.3) |
| 7 | Grouped reduce + bcast | `vcg*` + (`vsts`+`vlds BRC_BLK` \| `vselr` \| masked) | the grouped-bcast realization choice | **Full** (composed) | no native grouped-bcast op; `pto.as` must pick a realization (cost model, §11) |
| 8 | Convert | `vcvt`,`vtrc`,`vbitcast`,`pbitcast` | **~6** `vcvt` `part` (`EVEN/ODD`,`P0..P3`) + `rnd`/`sat` + the whole quant cast-chain/store-token | **Partial** | `vbitcast`/`pbitcast` (bit reinterpret) stay **explicit** — they are not a `vcvt` and carry no layout to infer |
| 9 | SFU | `chistv2`,`vbitsort`,`vmrgsort4`,`vgather2`,`vgatherb`,`vscatter`,`vexpdif`,`vaxpy`,`vlrelu`,`vprelu`,`vmull`,`vmula` | `chistv2` `Bin_N0/N1` (half axis) | **Partial** | sort + gather/scatter are **Category C**, passed through unchanged (no dist reduction); they sit at fused-region boundaries, not hidden |
| 10 | Predicate ops | `pset*`,`pge*`,`plt*`,`ppack`,`punpack`,`pintlv`,`pdintlv`,`pand`,`por`,`pnot` | `ppack`/`punpack`/`pintlv`/`pdintlv` `PART`(LOWER/HIGHER) companion modes | **Full** (as auto-companions) | companions are auto-inserted from the governed value; a hand author can still write them directly (escape hatch) |

### 16.2 The big reduction: the entire Data Rearrangement group disappears

The largest expressiveness *simplification* is that `pto.mi` **Group 12 (Data
Rearrangement)** — `vintlv`/`vdintlv` (and the non-A5 `*v2` forms), `vpack`/
`vsunpack`/`vzunpack` — has **no `pto.vmi` surface at all**. These are pure
layout artifacts: they only ever appear as the R2/R3 lowering of a `vcvt`
(parity/width) or a `vlds`/`vsts` that crosses an interleave axis. So a kernel
author never writes a single interleave/pack op; the `~32` load/store `dist`
modes plus the `~6` `vcvt` `part` modes plus the rearrangement ops all collapse
into the **logical descriptor** on `vlds`/`vsts`/`vcvt`.

Net: of the A5 vector surface, the `pto.vmi` layer **removes from authoring**
roughly **32 load/store `dist` tokens + 6 `vcvt` part modes + the 4-ish
rearrangement ops + the predicate companion modes**, replacing them with the
`#layout` descriptor and ~10 logical op groups.

### 16.3 Expressiveness gaps (where `pto.vmi` cannot yet cover `pto.mi`)

Summary of the **Partial** remarks above — these are the cases where an author
must still reach for `pto.mi`:

1. **Channel split/merge and up/down-sample loads/stores** (`SPLT4CHN`,
   `SPLT2CHN_B*`, `MRG4CHN_B*`, `MRG2CHN_B*`, `US_B*`, `DS_B*`): no logical
   `vmi` op models these multi-channel / resampling distributions yet. (Also note
   the `MRG*CHN` family is surface-retained but unsupported by current A5 HW.)
2. **Bit reinterpret** (`vbitcast`/`pbitcast`): intentionally kept explicit —
   there is no dtype/layout change to infer, so they pass through as-is.
3. **Carry-chain arithmetic** (`vaddc`/`vsubc` + `%carry`): the op is reachable
   but the multi-precision carry predicate is hand-threaded; no logical sugar.
4. **Category-C ops** (gather/scatter, `vbitsort`, `vmrgsort4`, unaligned/
   squeeze stores): exposed 1:1, **not** hidden — they are fused-region
   boundaries by design (P4/P7), so "coverage" here means pass-through, not
   `dist` reduction.
5. **Cross-VLane (R-axis) reduce**: only the in-VLane (`vcg*`) and full-array
   reduces have a logical form; an R-axis reduce needs an explicit fold or
   materialization (open issue O-2D.3).

Everything else (eltwise, vec-scalar, unary/transcendental, compare/select,
full + group reduce, scalar/grouped broadcast, `vcvt` widen/narrow incl. quant,
`chistv2`, the predicate companions) is **fully covered**: the `pto.vmi` surface
can express it and `pto.as` lowers it to the same `pto.mi` stream (P6).

---

## Part 17 — Appendix: consolidated cost cheat-sheet

Per logical value, `K` = reg fan-out. Mirrors design-exploration §5.4; cycle
terms are A5 `Ascend910_9599` CA simulator (*not silicon*).

| vmi op / pattern | lowering | `#mi` / value | extra ld/st | scratch | VL-util | dominant cost |
|---|---|---|---|---|---|---|
| `vadd/vmul/vsel/...` (R1) | `K ×` arith | `K` | 0 | 0 | 100% | `K×7c` |
| `vadds/...` (vec-scalar) | `K ×` arith | `K` | 0 | 0 | 100% | `K×7c` |
| `vdup` / `vbr` (ungrouped) | `vdup` (reg) | `1` | 0 | 0 | replicate | cheap |
| `arange`/`vci` | `vci` | `1`/chunk | 0 | 0 | 100% | cheap |
| `vlds`/`vsts` contiguous | `K × 9c` | `K` | — | 0 | 100% | bandwidth; 1+1 pairing |
| `vlds DINTLV` / `vsts INTLV` | `K ×` (9 / 12c) | `K` | — | 0 | 100% | INTLV 12c |
| `vcvt` 16↔32 (parity) | `2K × vcvt EVEN/ODD` + `ppack` | `2K` | 0 | 0 | 100% | parity widen |
| `vcvt` 8↔32 (radix-4) | `UNPK/PK4` + `vcvt P0` (+`vselr` narrow) | ~2–3 | 0 (reg path) | index/zero reg | 100% | see §12 |
| `vcvt` quant f32→U | per-dtype cast chain + `PK4_B32` | `K..3K` | 0 | 0 | 100% | chain depth by dtype |
| `vreduce_*` grouped (R4a) | `K × vcg*` | `K` | 0 | 0 | `R/E_v` out | **cheap** |
| `vreduce_*` full (R4b) | `(K-1) vadd + vcadd` | `K` | 0 | 0 | `1/L` out | fold chain |
| `vreduce_*` unaligned (R4d) | `[C]` materialize + reduce | `K`+ld/st | +2 | UB | low | last resort |
| grouped `vbr` — UB roundtrip | `vsts` + `vlds BRC_BLK` | `2K` | +2 | UB block | `C/E_v` | **decision** |
| grouped `vbr` — `vselr` | `vci`+`vmuls` + `K× vselr` | `K`+setup | 0 | index vreg | `C/E_v` | ~4× thruput |
| grouped `vbr` — masked | per-group op under masks | ∝#groups | 0 | mask pregs | `C/E_v` | scales w/ #groups |
| `chistv2` | `Bin_N0`+`Bin_N1` + accumulate | `~2K` | INTLV merge | — | `half` | per-half pred |
| `vgather/vscatter` (C) | gather/scatter | `K` | — | index | data-dep | 17–28c, ~0.1/cyc |
| predicate `pset/pge` | `pset`/`pge` | `1` | — | — | n/a | `PAT_ALL` shared |
| predicate companion (`ppack`/`pintlv`/...) | radix-2 reshape | `1–2` | — | — | n/a | follows data axis |
| MERGE-mode delta (A5) | zeroing op + `pnot` + `K× vor`/`vsel` | `+1+K` | 0 | +1 (`dst_old`) | n/a | ~free on A6 (native merge) |

**Reading the table.** Cheap, register-fuse-friendly: eltwise (R1), ungrouped
broadcast (`vdup`), VLane group reduce (R4a). Cost-model decision points: grouped
broadcast (UB vs `vselr` vs masked) and the reduce realization (fold vs partial).
Category-C frontiers (gather/scatter, sort, unaligned reduce, 1B register-resident
contiguous) are where a fused region splits and a `.contiguous()` may appear.

### Open-issue cross-references

The unresolved questions behind these mappings live in the companion docs:
2D as view vs type and the tile descriptor (design-exploration §1.5, O-2D.*);
register/predicate budget thresholds (§2.4, O-B.*); grouped-broadcast realization
thresholds (§5.6, O-G.*); predicate-as-type vs derived (§4.8, O-P.*); and the
instruction-mapping option spaces M1–M6 (requirements §6.2).

---
