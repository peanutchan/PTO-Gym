# PTO-VMI Unaligned Load/Store/Squeeze Use Cases

> **Source**: Extracted from A5 pto-isa kernel implementations (`TExtract`, `TInsert`, `TQuant`, `TGather`) and the xson radix topk `GatherConcat` pattern.
> **Context**: This document catalogs how the physical A5 ISA uses `vldas/vldus`, `vstus/vstas`, `vsqz/vstur/vstar` to handle misaligned UB addresses and sparse compact writes — patterns that pto.vmi must abstract away at the surface level while pto.as expands them during lowering.

---

## Table of Contents

1. [Background: A5 Alignment Constraint](#1-background-a5-alignment-constraint)
2. [Use Case 1: TExtract — Unaligned Sub-Tile Load](#2-use-case-1-textract--unaligned-sub-tile-load)
3. [Use Case 2: TInsert — Unaligned Sub-Tile Store](#3-use-case-2-tinsert--unaligned-sub-tile-store)
4. [Use Case 3: TQuant — Compact Reduce-Max Partial Write](#4-use-case-3-tquant--compact-reduce-max-partial-write)
5. [Use Case 4: TGather — Sparse Squeeze-Stream Output](#5-use-case-4-tgather--sparse-squeeze-stream-output)
6. [Use Case 5: Radix TopK GatherConcat — Accumulator Scan & Select](#6-use-case-5-radix-topk-gatherconcat--accumulator-scan--select)
7. [pto.vmi Surface Mapping](#7-pto-vmi-surface-mapping)
8. [pto.as Lowering Decision Summary](#8-pto-as-lowering-decision-summary)

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

## 5. Use Case 4: TGather — Sparse Squeeze-Stream Output

**Semantic**: Gather elements from source rows where a mask pattern selects active columns, then **compact the active elements to a contiguous stream** using `vsqz` (squeeze) and `vstur/vstar` (unaligned compact store).

### Unique Example: Column-Gather with Squeeze-Stream

```cpp
constexpr uint8_t SPR_AR_VALUE = 74;
constexpr auto sprValue = std::integral_constant<::Spr, static_cast<::Spr>(SPR_AR_VALUE)>();
sprclr(sprValue);              // Clear the address register for streaming

MaskReg dstPg0 = GetMaskVal<T, maskPattern>();
RegTensor<T> dstReg;
MaskReg executeMask;
UnalignReg ureg;

for (uint16_t i = 0; i < validRow; ++i) {
    for (uint16_t j = 0; j < repeatTimes; ++j) {
        loadMask = CreatePredicate<T>(maskValue);
        vlds(srcReg, srcPtr + i * srcStride, j * nElemPerVL, NORM);  // Load full row chunk
        pand(executeMask, dstPg0, loadMask, loadMask);                // Intersect mask + predicate
        vsqz(dstReg, srcReg, executeMask, MODE_STORED);              // Squeeze: compact active lanes to front
        vstur(ureg, dstReg, dstPtr, POST_UPDATE);                    // Write compacted vreg to stream
    }
}
vstar(ureg, dstPtr);  // Finalize the compact output stream
```

### Key Insight

- **vsqz semantics**: `vsqz(dstReg, srcReg, executeMask, MODE_STORED)` takes all lanes where `executeMask` is true and packs them to the front of `dstReg`. It returns the post-squeeze count as an SSA i32 (stored in a scalar register).
- **vstur semantics**: `vstur(ureg, dstReg, dstPtr, POST_UPDATE)` writes the **entire squeezed vreg** to the stream — but only the first `count` elements are meaningful (the rest are garbage from inactive lanes). The stream position advances by `count * sizeof(T)`.
- **vstar semantics**: `vstar(ureg, dstPtr)` flushes the UnalignReg stream and finalizes the output. Without this, the gathered elements would be incomplete.
- **sprclr + SPR_AR_VALUE**: The scalar address register (SPR) is cleared before the loop to reset the streaming position. `vstur` uses `POST_UPDATE` to auto-advance `dstPtr`.
- **Comparison with TQuant**: Both patterns write compact data to UB, but TQuant uses `vstus` (explicit count) because the count is known at compile-time (groupsPerRow). TGather uses `vsqz+vstur` because the count is dynamic (depends on how many lanes satisfy the mask predicate at runtime).

---

## 6. Use Case 5: Radix TopK GatherConcat — Accumulator Scan & Select

**Semantic**: In the xson radix topk implementation (Phase 5), the algorithm scans elements across multiple tile chunks, selects GT (greater-than) and EQ (equal) elements using `TGATHER`, then **accumulates** the selected indices into a growing output buffer using `TCONCAT_IMPL`. This is the most complex unaligned pattern — combining gather, squeeze, concat, and byte-count tracking.

### Phase 5: Per-Chunk Gather + Concat Accumulation

```cpp
// Six-arg TCONCAT_IMPL: concatenate a gathered segment into an accumulator
// TCONCAT_IMPL(segTmp, gtSeg, dstG, idxGtOut, idxGtAcc, concatG)
//   segTmp    — output segment tile (holds concatenated result)
//   gtSeg     — input gathered GT segment
//   dstG      — destination accumulator tile
//   idxGtOut  — output byte-count (updated by concat to reflect new total)
//   idxGtAcc  — input byte-count accumulator (where to start writing in dstG)
//   concatG   — scratch tile for concat internals

for (unsigned sub = 0; sub < nSubChunks; ++sub) {
    // Step 1: Gather GT elements from current chunk
    TGATHER<GatherDstTile, GatherSrcTile, PackedU16Tile, ConcatTile, TmpCmpTile, CmpMode::GT>(
        dstG, srcG, packedThrU, concatG, tmpG, base + sub);
    
    // Step 2: Concatenate gathered GT indices into accumulator
    TCONCAT_IMPL(segTmp, gtSeg, dstG, idxGtOut, idxGtAcc, concatG);
    
    // Step 3: Copy result back for next iteration
    TMOV(gtSeg, segTmp);
    TMOV(idxGtAcc, idxGtOut);
}
```

### Final Five-Arg Merge: GT + EQ into Single Output

```cpp
// After all chunks are processed, merge GT and EQ segments
// TCONCAT_IMPL(mergedIdx, gtSeg, eqSeg, idxGtAcc, idxEqAcc)
//   mergedIdx — final output tile
//   gtSeg     — accumulated GT indices
//   eqSeg     — accumulated EQ indices
//   idxGtAcc  — byte count of GT segment (acts as insertion offset for EQ)
//   idxEqAcc  — byte count of EQ segment
TCONCAT_IMPL(mergedIdx, gtSeg, eqSeg, idxGtAcc, idxEqAcc);
```

### TCONCAT_IMPL Physical Implementation

The concat operation physically uses `vlds + vscatter` for the second segment (writing at non-contiguous positions starting from `validCols0`):

```cpp
// From TConcat.hpp — the scatter path for concatenating src1 after src0
for (uint16_t i = 0; i < validRows; ++i) {
    // First segment: src0 — written at aligned row offsets [0..validCols0)
    for (uint16_t j = 0; j < repeatTimes0; ++j) {
        vlds(vreg_0, src0Ptr, i * RowStride0 + j * ElementsPerRepeat, NORM);
        vsts(vreg_0, dstPtr, i * RowStrideDst + j * ElementsPerRepeat, distValue, preg);
    }
    
    mem_bar(VST_VLD);  // Barrier between aligned write and scatter write
    
    // Second segment: src1 — scattered at positions [validCols0..validCols0+validCols1)
    for (uint16_t j = 0; j < repeatTimes1; ++j) {
        vlds(vreg_1, src1Ptr, i * RowStride1 + j * ElementsPerRepeat, NORM);
        vci(vreg_idx, (IndexScalar)(i * RowStrideDst + validCols0 + j * ElementsPerRepeat), INC_ORDER);
        vscatter(vreg_1, dstPtr, (RegTensor<UnsignedIndexScalar> &)vreg_idx, preg);
    }
}
```

### Key Insight

- **Byte-count accumulator tracking**: `idxGtAcc` and `idxGtOut` are scalar registers that track how many bytes have been concatenated so far. They serve as the "insertion offset" for the next chunk's gather results.
- **TGATHER → TCONCAT_IMPL pipeline**: Each chunk produces a sparse gather result (using `vsqz+vstur` internally), then `TCONCAT_IMPL` concatenates it at the current accumulator position. The loop builds the output incrementally.
- **mem_bar(VST_VLD)**: A memory barrier between the aligned `vsts` writes (first segment) and the `vscatter` writes (second segment) ensures ordering — the first segment must be visible in UB before the scatter can write after it.
- **vscatter for concat**: Unlike `vstus` which streams sequentially, `vscatter` writes to specific byte-offset positions computed by `vci` (vector index generation). This allows concatenating at arbitrary positions within a row.
- **Final merge**: The five-arg `TCONCAT_IMPL` merges GT and EQ segments — `idxGtAcc` tells where the EQ segment starts (immediately after all GT elements). This produces the final topk result as one contiguous sequence.

---

## 7. pto.vmi Surface Mapping

At the pto.vmi level, these five use cases map to surface ops that **never expose** the aligned/unaligned distinction:

| Use Case | pto.vmi Surface Op | Surface Signature | Hidden Complexity |
|----------|--------------------|-------------------|-------------------|
| TExtract | `TEXTRACT` | `textract(dst_tile, src_tile, row, col, shape)` | vldas+vldus vs vlds decision based on col alignment |
| TInsert | `TINSERT` | `tinsert(dst_tile, src_tile, row, col, shape)` | vstus+vstas vs vsts decision based on col/stride alignment |
| TQuant | `TQUANT` (reduce-max part) | `tquant_reduce_max(src_tile) → packed_max_tile` | vstus compact streaming + padding + vstas flush |
| TGather | `TGATHER` | `tgather(dst_tile, src_tile, mask)` | vsqz predicate compact + vstur/vstar streaming |
| GatherConcat | `TCONCAT` + `TGATHER` | `tgather+ tconcat(acc, segment, offset)` | vscatter at computed offsets + byte-count accumulator |

**Principle**: The pto.vmi programmer writes `TEXTRACT(dst, src, row, col, shape)` and the compiler decides:
- If `col * sizeof(T) % 32 == 0` → use `vlds+vsts` (aligned path)
- If `col * sizeof(T) % 32 != 0` → use `vldas+vldus` + `vsts` (unaligned load path)

Similarly for `TINSERT`, `TQUANT`, `TGATHER` — the surface op is one thing, the physical implementation varies.

---

## 8. pto.as Lowering Decision Summary

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

### Decision 4: Squeeze-Stream vs Scatter for Gather

```
TGATHER(dst, src, mask):
  ├─ GATHER_COL (column gather, fixed stride)
  │   → vlds + vsts with predicate mask  [aligned gather]
  └─ GATHER_ROW (row gather, variable selection)
      → sprclr + vlds + pand + vsqz + vstur per chunk + vstar  [squeeze-stream]
```

### Decision 5: Concat at Computed Offset

```
TCONCAT_IMPL(acc, segment0, segment1, offset0, offset1):
  → vlds + vsts for segment0 (aligned, at offset 0)
  → mem_bar(VST_VLD)
  → vlds + vci + vscatter for segment1 (at computed offset)
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
| `vscatter` | Scatter Write | `(vreg, basePtr, indexVec, preg)` | Write vreg elements to UB at positions computed by index vector |
| `vci` | Vector Index Generate | `(vreg_idx, startVal, INC_ORDER)` | Generate sequential byte-offset indices for scatter |
| `sprclr` | Address Register Clear | `(sprValue)` | Clear scalar address register for streaming position reset |
