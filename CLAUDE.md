# Claude Code & AI Agent Architecture Guide

## Project Overview
This repository contains a high-performance numerical compute kernel.

## Workflow Rules for AI Assistants
1. **Examine Available Playbooks**: Consult `agent_tools/skills/` before generating kernel code.
2. **Propose 3 Macro-Sketches**:
   - Never settle for the first naive loop implementation.
   - Propose 3 orthogonal structural sketches (Resident On-Chip, Double-Buffered Streaming, Algorithmic Reformulation).
3. **Apply the 3-Try Rule**:
   - Spend at most 2-3 iterations optimizing a branch. If performance stalls, prune the branch and jump to an orthogonal sketch.
4. **Enforce Boundary Safety & Precision**:
   - Check non-power-of-two dimensions and tail masking.
   - Use compensated summation (Kahan / Dekker) or iterative refinement when solving linear systems or eigen-decompositions.
