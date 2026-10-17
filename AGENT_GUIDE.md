# Autonomous HPC Kernel Optimization Guide for AI Agents

Welcome, AI Engineer / Coding Agent.

Your objective in this repository is to implement and optimize high-performance numerical compute kernels with **100% numerical accuracy** and **maximum hardware execution throughput**.

---

## 1. Quick Start & Resources
- **Playbooks & Skills**: Read [`agent_tools/skills/`](agent_tools/skills/) for detailed playbooks on memory tiling, boundary handling, and numerical stability.
- **Prompt Protocols**: Check [`agent_tools/prompts/`](agent_tools/prompts/) for architectural exploration and bottleneck classification.
- **Reference Traces**: View [`agent_tools/traces/sample_optimization_trace/`](agent_tools/traces/sample_optimization_trace/) to see how multi-round beam-search optimization works in practice.

---

## 2. Core Operational Workflow

### Phase 1: Macro-Sketch Exploration (Do not tune parameters yet!)
Propose **3 orthogonal architectural paradigms**:
1. **Resident SPAD / On-Chip State**: Maximum on-chip cache reuse.
2. **Double-Buffered Streaming Pipeline**: Overlap memory transfers with compute.
3. **Algorithmic Reformulation**: Math formulation tailored for SIMD/vector hardware.

### Phase 2: The 3-Try Early Abandonment Rule
- Allocate **AT MOST 3 iterations** per structural sketch.
- If a sketch does not show a steep performance descent or hits a hardware ceiling (e.g. synchronization overhead, bank conflicts), **ABANDON IT IMMEDIATELY**.
- Avoid local-minima traps: do not endlessly tweak tile sizes on a flawed pipeline.

### Phase 3: Hardware Verification Checklist
Before committing code, verify:
- [ ] Numerical residual meets tolerance across square, rectangular, and small edge cases.
- [ ] All memory reads/writes are aligned to vector width (e.g., 32-byte / 512-bit).
- [ ] Tail elements (non-multiple of tile size) are correctly masked.
- [ ] No buffer overflows or memory leaks in scratchpad / local allocations.
