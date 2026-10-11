# 40-Core Wide-SIMD Low-Level API Reference Manual

This document details the low-level SIMD intrinsics, memory hierarchy, DMA data movement, and synchronization primitives available on the wide-SIMD compute platform.

---

## 1. Hardware Architecture & Memory Hierarchy

- **SIMD Compute Units**: 40 parallel cores.
- **On-Chip Scratchpad (SPAD)**:
  - Each core has a dedicated on-chip memory space (~192 KB capacity).
  - All on-chip tensors must reside within the allocated SPAD arena.
  - Alignment: All buffers and DMA transfers require 32-byte alignment (8 `float` elements).
- **External Global Memory**:
  - Bound via `MemRef<T>`, accessible by all cores.
  - Reads and writes between Global Memory and SPAD must go through DMA block copies.

---

## 2. Buffer Management Primitives

### 2.1 Arena & Buffer Declarations
```cpp
ScratchArena pipe;                                   // On-chip memory allocator
ScratchBlock<BufferLocation::WORK> blockBuf;         // SPAD allocation handle
Scratch<float> localTensor;                          // Local tensor view (on-chip)
MemRef<float> globalMem;                             // Global memory reference
```

### 2.2 Buffer Reservation & Acquisition
```cpp
// Reserve bytes on the scratchpad (size must be a multiple of 32 bytes)
pipe.ReserveBlock(blockBuf, buffer_size_in_bytes);

// Acquire tensor pointer
localTensor = blockBuf.Get<float>();

// Slicing / Sub-views:
Scratch<float> subTensor = localTensor[offset];
```

---

## 3. Data Movement (DMA) Primitives

### 3.1 Continuous Block Copy
Copies a contiguous 1D array of elements between Global Memory and SPAD.
```cpp
// Global Memory -> SPAD:
BlockCopy(localDst, globalSrc, count);

// SPAD -> Global Memory:
BlockCopy(globalDst, localSrc, count);
```

### 3.2 Strided 2D Matrix Block Copy (`BlockCopyPartial`)
Copies 2D rectangular tiles with strided rows.
```cpp
BlockCopyParams prm;
prm.blockCount = num_rows;                                // Number of rows to transfer
prm.blockLen   = row_elements * sizeof(float);            // Bytes per row
prm.srcStride  = (stride_src - row_elements) * sizeof(float); // Gap between consecutive rows in src
prm.dstStride  = (stride_dst - row_elements) * sizeof(float); // Gap between consecutive rows in dst

// Global Memory -> SPAD:
BlockCopyPartial(localDst, globalSrc, prm);

// SPAD -> Global Memory:
BlockCopyPartial(globalDst, localSrc, prm);
```

---

## 4. Synchronization Primitives

### 4.1 Memory & Pipeline Fence
Ensures that all prior DMA transfers and SIMD vector operations have completed before downstream instructions execute.
```cpp
Fence();
```

---

## 5. SIMD Vector Compute Intrinsics

All vector instructions operate on `Scratch<float>` operands in SPAD and execute with wide hardware vector parallelism.

### 5.1 Element-Wise Binary Operations
Operates on two vector operands element-by-element (`dst[i] = a[i] OP b[i]`):
```cpp
Add(dst, a, b, count);  // dst[i] = a[i] + b[i]
Sub(dst, a, b, count);  // dst[i] = a[i] - b[i]
Mul(dst, a, b, count);  // dst[i] = a[i] * b[i]
Div(dst, a, b, count);  // dst[i] = a[i] / b[i]
Max(dst, a, b, count);  // dst[i] = std::max(a[i], b[i])
Min(dst, a, b, count);  // dst[i] = std::min(a[i], b[i])
```

### 5.2 Vector-Scalar Operations
Applies a scalar value to every element of a vector (`dst[i] = a[i] OP scalar`):
```cpp
MulScalar(dst, a, scalar, count);  // dst[i] = a[i] * scalar
AddScalar(dst, a, scalar, count);  // dst[i] = a[i] + scalar
MaxScalar(dst, a, scalar, count);  // dst[i] = std::max(a[i], scalar)
MinScalar(dst, a, scalar, count);  // dst[i] = std::min(a[i], scalar)
```

### 5.3 Element-Wise Unary Math Operations
```cpp
Sqrt(dst, a, count);   // dst[i] = sqrt(a[i])
Abs(dst, a, count);    // dst[i] = abs(a[i])
Exp(dst, a, count);    // dst[i] = exp(a[i])
Ln(dst, a, count);     // dst[i] = log(a[i])
```

### 5.4 Buffer Initialization
```cpp
Fill(dst, 0.0f, count);  // Fills count elements with constant value
```

### 5.5 Scalar Accessors
When scalar read or write is required on a SPAD tensor:
```cpp
float val = localTensor.GetValue(index); // or ScalarRead(localTensor, index)
localTensor.SetValue(index, val);        // or ScalarWrite(localTensor, index, val)
```

---

## 6. High-Precision Numerical Patterns

For algorithms sensitive to accumulation error (such as dot products in Householder reflections or Givens rotations):

```cpp
struct DF {
    float hi;
    float lo;
};

// TwoSum (Knuth): exact sum of two floats
inline DF TwoSum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return {s, e};
}

// TwoProd (Dekker): exact product of two floats
inline DF TwoProd(float a, float b) {
    float p = a * b;
    float e = std::fma(a, b, -p);
    return {p, e};
}
```
