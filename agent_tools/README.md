# Agent Playbook & HPC Kernel Optimization Suite

This directory contains battle-tested optimization playbooks, prompt templates, and real-world traces collected from cutting-edge HPC agent frameworks (Meta KernelAgent, tensormux/kernel-skills, and Sakana AI CUDA Engineer).

## Directory Structure
- `skills/`: Comprehensive engineering playbooks:
  - `cuda/`: Bank conflict elimination, shared memory tiling, warp divergence.
  - `triton/`: Autotuning block sizes, persistent kernels.
  - `patterns/`: Boundary condition handling, numerical stability, test harness creation.
  - `portability/`: SIMD & accelerator translation guides.
- `prompts/`:
  - `macro_sketch_exploration.md`: Proposing 3 orthogonal architectural spawn points.
  - `early_abandonment_guide.md`: The 3-try pruning rule.
  - `bottleneck_analysis.md`: Roofline & resource classification.
  - `reflexion_protocol.md`: Structured self-reflection.
- `templates/`: Meta KernelAgent Jinja prompt templates.
- `traces/`: Real-world optimization trace demonstrating multi-round performance acceleration.
