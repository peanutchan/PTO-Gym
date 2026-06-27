# PTO-VMI Unaligned Load/Store/Squeeze Use Cases

> **Source**: Extracted from A5 pto-isa kernel implementations (`TExtract`, `TInsert`, `TQuant`, `TGather`) and the xson radix topk `GatherConcat` pattern.
> **Context**: This document catalogs how the physical A5 ISA uses `vldas/vldus`, `vstus/vstas`, `vsqz/vstur/vstar` to handle misaligned UB addresses and sparse compact writes — patterns that pto.vmi must abstract away at the surface level while pto.as expands them during lowering.

---

## Table of Contents

1. [Background: A5 Alignment Constraint](#1-background-a5-alignment-constraint)
2. [Use Case 1: TExtract — Unaligned Sub-Tile Load](#2-use-case-1-textract--unaligned-sub-tile-load)
3. [Use Case 2: TInsert — Unaligned Sub-Tile Store](#3-use-case-2-tinsert--unaligned-sub-tile-store)
4. [Use Case 3: TQuant — Compact Reduce-Max Partial Write](#4-use-case-3-tquant--compact-reduce-max-partial-write)
5. [Use Case 4: Sorted Unique Dedup — Adjacent Shift vcmp + vsqz](#5-use-case-4-sorted-unique-dedup--adjacent-shift-vcmp--vsqz)
6. [pto.vmi Surface Mapping](#6-pto-vmi-surface-mapping)
7. [pto.as Lowering Decision Summary](#7-pto-as-lowering-decision-summary)

---

## 1. Background: A5 Alignment Constraint

A5 requires **all `vlds`/`vsts` addresses to be 32-byte-aligned** (`PTO_UBUF_ALIGN_BYTES = 32`). Any misaligned UB pointer causes a simulator crash and real-hardware corruption. The unaligned ISA pairs solve this:

| Pair | Direction | Mechanism |
|------|-----------|-----------|
| `vldas` + `vldus` | **Load** | `vldas` primes an `UnalignReg` from a misaligned address; `vldus` consumes it to fetch data into a `vreg` |
| `vstus` + `vstas` | **Store** | `vstus` streams `count` elements from a `vreg` through an `UnalignReg` to UB (advancing pointer); `vstas` flushes residual buffered bytes |
| `vsqz` + `vstur` + `vstar` | **Squeeze-Store** | `vsqz` compacts predicate-active lanes to the front of a vreg (returns count as SSA i32); `vstur` writes the compacted vreg to UB via UnalignReg; `vstar` finalizes the stream |

Key invariant: **pto.vmi never distinguishes aligned vs unaligned** — the compiler (`pto.as`) decides which physical path to use based on stride/offset analysis.

---

## 2. Use Case 1: TExtract — Unaligned Sub-Tile Load

**Semantic**: Extract a sub-tile region `[indexRow:indexRow+validRow, indexCol:indexCol+validCol]` from a source tile into a destination tile. The destination is always aligned (row-stride of dst tile), but the source row-pointer may be misaligned when `indexCol` is not a multiple of 32B/sizeof(T).

### Aligned Path (indexCol is 32B-aligned)

```cpp
for (uint16_t i = 0; i < validRow; ++i) {
    uint32_t srcRowOff = (indexRow + i) * srcRowStride + indexCol;  // aligned offset
    for (uint16_t j = 0; j < lastRepeat; ++j) {
        vlds(vreg, srcAddr, srcRowOff + j * elementsPerRepeat, NORM);
        vsts(vreg, dstAddr, dstRowOff + j * elementsPerRepeat, distValue, pregFull);
    }
    vlds(vreg, srcAddr, srcRowOff + lastRepeat * elementsPerRepeat, NORM);
    vsts(vreg, dstAddr, dstRowOff + lastRepeat * elementsPerRepeat, distValue, pregTail);
}
```

### Unaligned Path (indexCol NOT 32B-aligned)

```cpp
UnalignReg ureg;
MaskReg pregTail = CreatePredicate<RegT>(kTail);
for (uint16_t i = 0; i < validRow; ++i) {
    __ubuf__ RegT *psrc = srcAddr + (indexRow + i) * srcRowStride + indexCol;  // raw pointer, may be misaligned
    uint32_t dstRowOff = i * dstRowStride;
    vldas(ureg, psrc);       // Prime UnalignReg from misaligned source
    vldus(vreg, ureg, psrc); // Consume UnalignReg — fetch data into vreg
    vsts(vreg, dstAddr, dstRowOff, distValue, pregTail);  // Store to aligned dst
}
```

**Multi-chunk variant** (validCol > one VL):

```cpp
for (uint16_t i = 0; i < validRow; ++i) {
    __ubuf__ RegT *psrc = srcAddr + (indexRow + i) * srcRowStride + indexCol;
    for (uint16_t j = 0; j < lastRepeat; ++j) {
        vldas(ureg, psrc + j * elementsPerRepeat);   // Each chunk: prime
        vldus(vreg, ureg, psrc + j * elementsPerRepeat);  // Then consume
        vsts(vreg, dstAddr, dstRowOff + j * elementsPerRepeat, distValue, pregFull);
    }
    vldas(ureg, psrc + lastRepeat * elementsPerRepeat);
    vldus(vreg, ureg, psrc + lastRepeat * elementsPerRepeat);
    vsts(vreg, dstAddr, dstRowOff + lastRepeat * elementsPerRepeat, distValue, pregTail);
}
```

### Key Insight

- **Source pointer** (`psrc`) is a raw UB pointer that can start at any byte offset — `vldas+vldus` handles the misalignment.
- **Destination offset** (`dstRowOff`) is a stride-based scalar offset that must be 32B-aligned (guaranteed by dst tile layout).
- The **single-chunk** case is the simplest pattern: one `vldas` + one `vldus` per row.
- The **multi-chunk** case chains `vldas+vldus` for each VL-sized portion — each chunk's address advances by `elementsPerRepeat` (which is always 32B-aligned since it equals VL size), so after the first chunk the rest are naturally aligned.

---

## 3. Use Case 2: TInsert — Unaligned Sub-Tile Store

**Semantic**: Insert a source sub-tile into a destination tile at position `[indexRow, indexCol]`. The source is read with aligned row-stride offsets, but the destination pointer may be misaligned when `indexCol` is not 32B-aligned.

### Aligned Path (strides + indexCol 32B-aligned)

```cpp
for (uint16_t i = 0; i < validRow; ++i) {
    uint32_t dstRowOff = (indexRow + i) * dstRowStride + indexCol;
    for (uint16_t j = 0; j < lastRepeat; ++j) {
        vlds(vreg, srcAddr, srcRowOff + j * elementsPerRepeat, NORM);
        vsts(vreg, dstAddr, dstRowOff + j * elementsPerRepeat, distValue, pregFull);
    }
    // tail chunk with predicate
    vlds(vreg, srcAddr, srcRowOff + lastRepeat * elementsPerRepeat, NORM);
    vsts(vreg, dstAddr, dstRowOff + lastRepeat * elementsPerRepeat, distValue, pregTail);
}
```

### Unaligned Path (strides or indexCol NOT 32B-aligned)

```cpp
RegTensor<T> vreg;
UnalignReg ureg;

for (uint16_t i = 0; i < validRow; ++i) {
    uint32_t srcRowOff = i * srcRowStride;
    __ubuf__ T *pdst = dstAddr + (indexRow + i) * dstRowStride + indexCol;  // misaligned raw pointer
    for (uint16_t j = 0; j < kFullRepeats; ++j) {
        vlds(vreg, srcAddr, srcRowOff + j * elementsPerRepeat, NORM);  // Aligned load from src
        vstus(ureg, elementsPerRepeat, vreg, pdst, POST_UPDATE);       // Stream to misaligned dst
    }
    if constexpr (kRemainder > 0) {
        vlds(vreg, srcAddr, srcRowOff + kFullRepeats * elementsPerRepeat, NORM);
        vstus(ureg, kRemainder, vreg, pdst, POST_UPDATE);  // Tail elements
    }
    vstas(ureg, pdst, 0, POST_UPDATE);  // Flush residual buffered bytes
}
```

### Key Insight

- **vstus streaming**: Each `vstus(ureg, count, vreg, pdst, POST_UPDATE)` writes `count` elements from `vreg` through the UnalignReg stream and **auto-advances** `pdst` by `count * sizeof(T)`.
- **POST_UPDATE**: The `POST_UPDATE` flag is critical — it means `pdst` is updated in-place after each chunk, so the next `vstus` writes to the next position without recalculating offsets.
- **vstas flush**: The final `vstas(ureg, pdst, 0, POST_UPDATE)` flushes any residual bytes buffered in `ureg`. Without this, the last few bytes would be lost.
- **Symmetry with TExtract**: TExtract uses `vldas+vldus` for unaligned **reads**; TInsert uses `vstus+vstas` for unaligned **writes**. The pattern is load→aligned-dest vs aligned-src→store.

---

## 4. Use Case 3: TQuant — Compact Reduce-Max Partial Write

**Semantic**: Compute per-group abs-max across rows of a source tile, then **write the reduce-max results tightly** (packed layout `[validRows, groupsPerRow]`) using unaligned streaming stores. This is the most important pattern for tilelang_ops kernels — reduce outputs are often compact and don't fill a full tile row.

### Single-LoopRow Variant (elemsPerRow fits in one deinterleave window)

```cpp
__ubuf__ T *writePtr = maxPtr;    // Start of packed output region
UnalignReg ureg_max;

for (uint16_t row = 0; row < validRows; ++row) {
    uint32_t srcRowOff = row * srcCols;
    AbsReduceMax_b16_DintlvWindow(srcPtr, srcRowOff, elemsPerRow, vb16_max);  // Reduce per row
    
    uint32_t outCount = groupsPerRow;   // e.g., 4 groups for validCols=128, grp_size=32
    vstus(ureg_max, outCount, vb16_max, writePtr, POST_UPDATE);  // Stream compact max per row
}

// Pad to 32B alignment at the end
uint32_t groupsWritten = validRows * groupsPerRow;
uint32_t paddedGroups = CeilDivision(groupsWritten, alignGroups) * alignGroups;
uint32_t padCount = paddedGroups - groupsWritten;
if (padCount > 0)
    vstus(ureg_max, padCount, vb16_max, writePtr, POST_UPDATE);  // Pad with last max value
vstas(ureg_max, writePtr, 0, POST_UPDATE);  // Flush stream
```

### Multi-LoopRow Variant (elemsPerRow spans multiple deinterleave windows)

```cpp
__ubuf__ T *writePtr = maxPtr;

for (uint16_t row = 0; row < validRows; ++row) {
    uint32_t srcRowOff = row * srcCols;
    for (uint16_t i = 0; i < loopNumPerRow; ++i) {
        // Reduce partial window
        AbsReduceMax_b16_DintlvWindow(srcPtr, srcRowOff + colOff, remaining, vb16_max);
        
        // Stream partial max output (only groupsThisIter, not full groupsPerRow)
        uint32_t grpsThisIter = min(grpsRemaining, grps_per_dintlv);
        vstus(ureg_max, grpsThisIter, vb16_max, writePtr, POST_UPDATE);
    }
}
vstas(ureg_max, writePtr, 0, POST_UPDATE);  // Single flush after all rows
```

### Key Insight

- **Compact streaming**: The reduce-max output is much smaller than a full tile row (e.g., 4 BF16 values per row for 128-element groups). `vstus` with `POST_UPDATE` streams each row's output sequentially, advancing `writePtr` automatically.
- **Padding**: After all rows are written, the total byte count may not be 32B-aligned. Extra `vstus` pads to alignment, then `vstas` flushes. The pad values are the last `vb16_max` (arbitrary — just filling alignment bytes).
- **No predicate needed**: `vstus` takes an explicit `outCount` parameter — it writes exactly that many elements. This is more natural than `vsts` with predicate masks for compact outputs.
- **One flush per region**: In the multi-row case, **all** `vstus` calls share one `ureg_max` and one final `vstas`. This is because the entire packed region is one contiguous stream — each `vstus` appends to it, and only one `vstas` is needed at the end.
- **Critical for tilelang_ops**: Norm, softmax, and FP8 kernels all produce compact reduce-max or scaling outputs. This pattern is the canonical way to write them on A5.

---

## 5. Use Case 4: Sorted Unique Dedup — Adjacent Shift vcmp + vsqz

**Semantic**: Given a **sorted** 1D vector in UB, compute the unique (deduplicated) elements by loading the same vector twice — once at the original offset and once shifted by 1 element — then comparing adjacent pairs with `vcmp_ne`. The comparison produces a predicate mask where true lanes mark "first occurrence" elements (elements that differ from their predecessor). `vsqz` compacts these predicate-true lanes to the front, and `vstur/vstar` streams the compacted unique values to the output buffer.

This is the canonical `vsqz+vstur/vstar` pattern — it demonstrates how **dynamic-count compact output** works when the number of surviving elements is unknown at compile time.

### Problem Setup

```
Input:  sorted_vec = [1, 1, 1, 3, 3, 5, 5, 5, 7, 9, 9, ...]  (sorted, may have duplicates)
Output: unique_vec = [1, 3, 5, 7, 9, ...]  (compact, no duplicates)
```

The key observation: in a sorted list, an element is a "first occurrence" (unique representative) if it differs from its immediate predecessor. Element at index `i` is unique iff `sorted_vec[i] != sorted_vec[i-1]`. Index 0 is always unique (no predecessor).

### Physical Implementation: Shift-Load + vcmp_ne + vsqz + vstur/vstar

```cpp
// --- Sorted Unique Dedup on A5 ---
// Input:  __ubuf__ T *sortedPtr   — sorted 1D vector, length N
// Output: __ubuf__ T *uniquePtr   — compact unique output (stream-written)

using RegT = RegTensor<T>;        // e.g. bfloat16 or half
constexpr uint32_t VL_ELEMS = CCE_VL / sizeof(T);  // Elements per vector load (128 for BF16)
constexpr uint32_t REPEAT_BYTE = 32;               // 32B alignment unit

__ubuf__ RegT *sortedAddr = (__ubuf__ RegT *)sortedPtr;
__ubuf__ RegT *uniqueAddr = (__ubuf__ RegT *)uniquePtr;
uint32_t totalElems = N;
uint32_t nChunks = CeilDivision(totalElems, VL_ELEMS);

__VEC_SCOPE__
{
    RegT vreg_curr, vreg_prev, vreg_unique;
    MaskReg preg_all, preg_ne, preg_first;
    UnalignReg ureg;

    // Clear stream address register for compact output
    constexpr uint8_t SPR_AR_VALUE = 74;
    sprclr(SPR_AR_VALUE);

    // First element is always unique — special-case the first chunk
    preg_first = CreatePredicate<T>(1);  // Only lane 0 is true for "no predecessor"

    for (uint32_t chunk = 0; chunk < nChunks; ++chunk) {
        uint32_t chunkOff = chunk * VL_ELEMS;
        uint32_t elemCount = min(VL_ELEMS, totalElems - chunkOff);
        preg_all = CreatePredicate<T>(elemCount);

        // Load current chunk
        vlds(vreg_curr, sortedAddr, chunkOff, NORM);

        // Build "not-equal-to-predecessor" mask
        if (chunk == 0) {
            // Chunk 0: element[0] has no predecessor → always unique
            //   For i>0 within chunk 0: compare sorted[i] vs sorted[i-1]
            //   Use vldas+vldus for shifted load (offset 1 byte = not 32B-aligned for BF16)
            UnalignReg ureg_shift;
            __ubuf__ RegT *shiftedPtr = sortedAddr + 1;  // Shift by 1 element (not 32B-aligned!)
            vldas(ureg_shift, shiftedPtr);                // Prime unaligned load for predecessor
            vldus(vreg_prev, ureg_shift, shiftedPtr);    // Load sorted[0..VL_ELEMS-2] as predecessor of sorted[1..VL_ELEMS-1]
            vcmp_ne(preg_ne, vreg_curr, vreg_prev, preg_all);  // ne mask for i>=1
            // Combine: lane 0 = always unique (no pred), lanes 1+ = ne result
            por(preg_ne, preg_first, preg_ne);  // Union: first-element flag + ne mask
        } else {
            // Chunk k>0: all elements have a predecessor
            //   predecessor of sorted[chunk*VL] is sorted[chunk*VL - 1]
            //   Load predecessor from previous chunk's last element + current shifted view
            //   Simplest: load the shifted window from (chunkOff - 1)
            uint32_t prevOff = chunkOff - 1;  // Predecessor starts 1 element before current chunk
            // NOTE: prevOff may not be 32B-aligned! Need vldas+vldus
            UnalignReg ureg_shift;
            __ubuf__ RegT *shiftedPtr = sortedAddr + prevOff;
            vldas(ureg_shift, shiftedPtr);
            vldus(vreg_prev, ureg_shift, shiftedPtr);
            vcmp_ne(preg_ne, vreg_curr, vreg_prev, preg_all);  // All lanes: ne with predecessor
        }

        // Squeeze: compact unique elements (where preg_ne is true) to front of vreg
        vsqz(vreg_unique, vreg_curr, preg_ne, MODE_STORED);

        // Stream-write compacted unique values to output
        vstur(ureg, vreg_unique, uniqueAddr, POST_UPDATE);
    }

    // Finalize the compact output stream
    vstar(ureg, uniqueAddr);
}
```

### Visual Walkthrough

```
sorted: [1, 1, 1, 3, 3, 5, 5, 5, 7, 9, 9, ...]
         ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
prev:   [_, 1, 1, 1, 3, 3, 5, 5, 5, 7, 9, ...]  (shifted by 1)
vcmp_ne:[T, F, F, T, F, T, F, F, T, T, F, ...]  (T = differs from predecessor)
vsqz:   [1, 3, 5, 7, 9, _, _, _, _, _, _, ...]  (compact T-lanes to front)
vstur:  writes [1, 3, 5, 7, 9] to output stream  (only first count elements are meaningful)
```

### Key Insight

- **Adjacent shift requires unaligned load**: Loading the "predecessor" vector (shifted by 1 element from the base) produces a UB address that is **not 32B-aligned** (for BF16, 1 element = 2 bytes, so offset 2 bytes from 32B boundary). This is exactly the `vldas+vldus` pattern from Use Case 1 (TExtract), applied here for a **1D shift-load** rather than a 2D sub-tile extraction.
- **vsqz produces dynamic count**: The number of unique elements in each chunk is unknown at compile time — it depends on the data distribution. `vsqz` returns the post-squeeze count as an SSA i32, which `vstur` uses implicitly (it writes only `count` meaningful elements from the squeezed vreg).
- **vstur/vstar for compact streaming**: The output is a contiguous stream of unique values — each chunk's `vsqz` result is appended via `vstur(POST_UPDATE)`, and `vstar` finalizes the total stream. This contrasts with `vstus+vstas` (Use Case 3, TQuant) where the count is known at compile time.
- **Comparison across use cases**:
  - **TQuant** (Use Case 3): `vstus(ureg, knownCount, vreg, ptr, POST_UPDATE)` — **static count**, explicit parameter
  - **Unique dedup** (Use Case 4): `vsqz + vstur(ureg, vreg, ptr, POST_UPDATE)` — **dynamic count**, vsqz decides how many elements survive
  - **TExtract** (Use Case 1): `vldas+vldus` for **unaligned load of source** — same ISA pair, but for reading misaligned data
  - **Unique dedup** also uses `vldas+vldus` for the shifted predecessor load — combining **unaligned load + squeeze-stream store** in one kernel

---

## 6. pto.vmi Surface Mapping

At the pto.vmi level, these four use cases map to surface ops that **never expose** the aligned/unaligned distinction:

| Use Case | pto.vmi Surface Op | Surface Signature | Hidden Complexity |
|----------|--------------------|-------------------|-------------------|
| TExtract | `TEXTRACT` | `textract(dst_tile, src_tile, row, col, shape)` | vldas+vldus vs vlds decision based on col alignment |
| TInsert | `TINSERT` | `tinsert(dst_tile, src_tile, row, col, shape)` | vstus+vstas vs vsts decision based on col/stride alignment |
| TQuant | `TQUANT` (reduce-max part) | `tquant_reduce_max(src_tile) → packed_max_tile` | vstus compact streaming + padding + vstas flush |
| Unique Dedup | `TUNIQUE` | `tunique(dst_tile, sorted_src_tile)` | vldas+vldus for shifted predecessor load + vcmp_ne + vsqz + vstur/vstar compact stream |

**Principle**: The pto.vmi programmer writes `TEXTRACT(dst, src, row, col, shape)` and the compiler decides:
- If `col * sizeof(T) % 32 == 0` → use `vlds+vsts` (aligned path)
- If `col * sizeof(T) % 32 != 0` → use `vldas+vldus` + `vsts` (unaligned load path)

Similarly for `TINSERT`, `TQUANT`, `TUNIQUE` — the surface op is one thing, the physical implementation varies.

---

## 7. pto.as Lowering Decision Summary

When `pto.as` lowers pto.vmi surface ops to pto.mi physical instructions, it makes these decisions:

### Decision 1: Aligned vs Unaligned Load

```
TEXTRACT(dst, src, row, col, shape)
  ├─ col * sizeof(T) % 32 == 0  → vlds(vreg, srcAddr, srcRowOff, NORM)  [aligned]
  └─ col * sizeof(T) % 32 != 0  → vldas(ureg, psrc) + vldus(vreg, ureg, psrc)  [unaligned]
```

### Decision 2: Aligned vs Unaligned Store

```
TINSERT(dst, src, row, col, shape)
  ├─ stride * sizeof(T) % 32 == 0 AND col * sizeof(T) % 32 == 0
  │   → vlds + vsts  [aligned]
  └─ otherwise
      → vlds + vstus(ureg, count, vreg, pdst, POST_UPDATE) per chunk
        + vstas(ureg, pdst, 0, POST_UPDATE) flush  [unaligned]
```

### Decision 3: Compact vs Predicated Write

```
Reduce output (TQUANT AbsReduceMax):
  ├─ Output size known, compact layout
  │   → vstus(ureg, outCount, vreg, writePtr, POST_UPDATE) per row
  │     + padding vstus + vstas flush  [compact streaming]
  └─ Output fills a full tile row
      → vsts(vreg, dstAddr, dstRowOff, distValue, preg)  [predicated aligned]
```

### Decision 4: Static-Count vs Dynamic-Count Compact Write

```
Compact output write:
  ├─ Count known at compile time (e.g. reduce-max groupsPerRow)
  │   → vstus(ureg, knownCount, vreg, writePtr, POST_UPDATE) per row
  │     + padding vstus + vstas flush  [TQuant pattern — Use Case 3]
  └─ Count unknown at compile time (e.g. unique dedup, predicate-based gather)
      │   → vsqz(dstReg, srcReg, mask, MODE_STORED) to compact predicate-active lanes
      │   → vstur(ureg, dstReg, dstPtr, POST_UPDATE) per chunk to stream compacted output
      │   → vstar(ureg, dstPtr) to finalize  [Unique dedup / TGather pattern — Use Case 4]
      └─ For the shifted predecessor load in unique dedup:
          → vldas(ureg_shift, shiftedPtr) + vldus(vreg_prev, ureg_shift, shiftedPtr)
            (1-element shift produces misaligned address — Use Case 1 pattern)
```

---

## Appendix: Instruction Reference Quick Table

| Instruction | Category | Operands | Semantics |
|-------------|----------|----------|-----------|
| `vldas` | Unaligned Load Prime | `(ureg, psrc)` | Prime UnalignReg from misaligned UB address |
| `vldus` | Unaligned Load Consume | `(vreg, ureg, psrc)` | Fetch data from misaligned address using primed ureg |
| `vstus` | Unaligned Store Stream | `(ureg, count, vreg, pdst, POST_UPDATE)` | Stream `count` elements from vreg to misaligned UB, advance pointer |
| `vstas` | Unaligned Store Flush | `(ureg, pdst, padCount, POST_UPDATE)` | Flush residual buffered bytes in ureg to UB stream |
| `vsqz` | Squeeze Compact | `(dstReg, srcReg, mask, MODE_STORED)` | Pack predicate-active lanes to front of dstReg |
| `vstur` | Squeeze-Store Stream | `(ureg, vreg, pdst, POST_UPDATE)` | Write squeezed vreg to UB via UnalignReg, advance |
| `vstar` | Squeeze-Store Finalize | `(ureg, pdst)` | Finalize the squeeze-stream output |
| `vcmp_ne` | Vector Compare NE | `(dstMask, src0, src1, preg)` | Element-wise not-equal comparison, result as predicate mask |
| `vcmp_eq` | Vector Compare EQ | `(dstMask, src0, src1, preg)` | Element-wise equal comparison, result as predicate mask |
| `vcmp_gt` | Vector Compare GT | `(dstMask, src0, src1, preg)` | Element-wise greater-than comparison, result as predicate mask |
| `por` | Predicate OR | `(dst, src0, src1)` | Union of two predicate masks |
| `sprclr` | Address Register Clear | `(sprValue)` | Clear scalar address register for streaming position reset |
