# Self-Reflexion Protocol

After every benchmark or verification run, emit a structured reflexion block following the Meta KernelAgent specification:

```json
{
  "attempt": 1,
  "macro_sketch": "Sketch A (Resident SPAD Normal Equations)",
  "was_diagnosis_correct": true,
  "was_fix_effective": true,
  "expected_outcome": "Double buffering should hide GM transfer latency by overlapping with vector GEMM",
  "actual_outcome": "Execution time decreased from 1.42ms to 0.98ms (31% speedup). Memory stalls dropped from 44% to 12%",
  "reasoning": "The double buffer successfully hid MTE latency. However, pipe barrier sync is now visible.",
  "lessons": [
    "SPAD allocation must reserve margin for temporary reduction workspace"
  ],
  "decision": "PROMOTE_TO_ROUND_2" // or "ABANDON_AND_SWITCH_SKETCH"
}
```
