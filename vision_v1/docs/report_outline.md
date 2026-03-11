# Report Outline

Suggested title:

`RefStoryRank: Learning to Select Style-Consistent Storyboards from Reference-Guided Diffusion Samples`

Sections:

1. Introduction
2. Related Work
3. Method
4. Experiments
5. Discussion
6. Conclusion

Core experiments:

- Prompt-only independent generation.
- IP-Adapter reference-guided generation.
- IP-Adapter + trained reranker (ours).
- Ablation on beam size, number of candidates, and negative sampling strategy.
- Error analysis on failure modes: character drift, color drift, layout collapse, prompt mismatch.
