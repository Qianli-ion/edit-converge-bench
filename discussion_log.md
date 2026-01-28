# Round-Trip Image Editing Benchmark: Discussion Log

## Session: 2026-01-26

---

## High-Level Questions

### Q1: Core Motivation
**Question**: What problem are you trying to solve that existing benchmarks don't address?

**Response**:
- Real users don't edit once—they iterate: "make apple green" → "a bit more green" → "add yellow" → "move right"
- Users often don't know what they want upfront; they explore via iteration
- Current benchmarks test single-shot edit quality, not **quality degradation over iterations**
- Concern is not just semantic drift, but **technical quality decay**: compression artifacts, blurriness, detail loss
- If quality degrades, the editing model becomes less useful as a creative tool

**Core insight**: Evaluating iterative robustness, not just single-edit fidelity.

---

### Q2: Round-Trip vs Linear Sequence
**Question**: Why round-trip specifically, rather than just chaining N random edits?

**Response**:
1. **Practical**: Original image serves as clear reference → enables reference-based IQA (MSE, SSIM, etc.) to measure exactly what they're supposed to measure
2. **Secondary**: Also reveals whether "reversible" edits are actually reversible
3. **Deeper thesis - Convergence in Human-Machine Interaction**:
   - A good editing system should help users **converge** toward their goal
   - Bad system: edit X causes unintended side-effect Y → fix Y causes side-effect Z → **divergence**
   - Image quality degradation is a specific case of irreversible divergence
   - You can't easily "undo" accumulated artifacts—they compound
   - Round-trip tests **controllability**: can the user steer the system predictably?

**Key insight**: This isn't just about image quality—it's about whether the editing system is *steerable* and *predictable* for iterative workflows.

---

### Q3: Convergence as Core Thesis
**Question**: Should quality divergence and semantic divergence be measured separately or unified?

**Response**:
- **"Convergence" is the right framing** for the benchmark name
- Concern: Need a mathematical/technical definition that aligns with intuition
- Approach: **(A) Two separate metrics**, but ideally unified under a generic "convergence" metric
- Quality divergence could be one *dimension* of overall convergence
- Goal: Define convergence formally, then show quality/semantic as contributing factors

**Open problem**: How to mathematically define "convergence" for editing systems?

---

### Q4: Formalizing Convergence
**Question**: What does "convergence" mean mathematically?

**Response**:
- Initial instinct: **Banach Contraction Mapping Theorem**
- Not fully articulated yet, but something about this framing resonates

---

### Q5: Banach Contraction Intuition
**Question**: Which interpretation of contraction mapping fits?

**Response**:
- **Interpretation B is closest**: User-goal convergence
- Challenge: User's goal is hard to define a priori
- Reframe: Convergence = system reaches a state where user is **satisfied/happy**
- Don't need to define goal explicitly—just observe whether system enables reaching satisfactory state

**Tension**: For a benchmark, need something measurable without human in the loop.

---

### Q6: Round-Trip as Proxy for Convergence
**Question**: Is round-trip a proxy for general convergence, or measuring something more specific?

**Response**:
- Quality degradation in round-trip is **one specific way** convergence is hurt
- Key insight: Editing assumes image quality stays comparable before/after
- If after 10 iterations artifacts become obvious → image is unusable
- Even if **semantically correct**, quality failure = convergence failure
- The "medium" (image quality) must survive iteration, not just the "content" (semantics)

**Analogy**: Like editing a document where each save introduces typos. Content might be right, but medium is corrupted.

---

### Q7: Independence of Quality vs Semantic Dimensions
**Question**: Are semantic control and quality preservation truly independent failure modes?

**Response**:
- **Yes, they are distinct failure modes**
- Real example: **Gemini's Nano Banana** - excellent semantic control, but after ~5 edits shows substantial artifacts (color, sharpness)
- Type C (good quality, poor semantics) is harder to find in practice
- But theoretically distinct and worth measuring separately

**Implication**: Benchmark needs separate metrics for both dimensions.

---

### Q8: Benchmark Structure
**Question**: What does one test case look like?

**Confirmed Structure**:
```
Test Case:
  - Input: Source image I₀
  - Edit pair: (prompt_forward, prompt_backward)
  - Iterations: N round-trips

Execution:
  I₀ → edit(forward) → I₁ → edit(backward) → I₁'
     → edit(forward) → I₂ → edit(backward) → I₂'
     → ... (N times)

Measurements at each n:
  - Quality: PSNR/SSIM/LPIPS(I₀, Iₙ')
  - Semantic: Forward edit happened? Backward reversed it?
```

**Response**:
- Structure is correct
- **Vary N** and plot error vs N (degradation curve)
- Different models may show different curve shapes (linear, exponential, plateau)

---

### Q9: Edit Pair Design
**Question**: How to design edit pairs? Categories, difficulty, prompt generation?

**Response**:

**A) Attribute edits are tricky**:
- "black → blonde → black" - final black may differ from original black
- This is semantically okay (both are valid "black")
- This is **semantic consistency**, not quality consistency
- For **quality consistency**, edit pairs need clear, unambiguous reversibility

**B) Difficulty levels**:
- Difficulty correlates with **subjectivity/ambiguity** of edit
- Also depends on **information destruction** in first edit
- Example: "turn entire image black" destroys almost all info → recovery is impossible
- This is a fundamental limit, not model failure

**Key insight**: Need to distinguish between:
- Edits that are *theoretically* reversible (info-preserving)
- Edits that *destroy information* by design (not fair to test)

**C) Prompt generation**:
- Start with **LLM-generated** inverse prompts
- Fall back to **template-based** if LLM is too flexible/inconsistent

---

### Q10: Information Preservation Spectrum
**Question**: Where to draw the line for "fair" test cases?

**Response**:
- **Start with clearly reversible edits** (left side of spectrum)
- Need a **mechanism to identify/classify** which edits are reversible
- Interesting case: **"information adding" edits**
  - Example: blurred → clarify → blur again
  - "Clarify" hallucinates details (adds fake info)
  - Could still test whether the round-trip is consistent
  - Even if clarified details aren't "real," can they be consistently blurred back?

**Key insight**: Two types of non-reversible edits:
1. Info-destroying: clear → blur (loses info)
2. Info-hallucinating: blur → clarify (invents info)

Both directions could be interesting to test, with different expectations.

---

### Q11: Reversibility Classification
**Question**: How to classify/identify reversible edits?

**Response**:
- **A) Human annotation**: Use only for **verification** of scaled mechanisms, not primary method
- **B) Heuristic rules**: Most viable route
  - But ambiguous edits like "color A → color B" can be ill-defined
  - "red" vs "orange" is ambiguous
  - Pantone or Hex codes are precise, BUT unclear if models follow them well

**Tension identified**:
- Precise specs (hex codes) → unambiguous → good for benchmark definition
- But models may not follow precise specs → testing prompt-following, not convergence

---

### Q12: Unambiguous AND Model-Supported Edits
**Question**: What edits are unambiguous AND well-supported? Should we filter out forward-edit failures?

**Response**:
- **Criterion**: If edit is clear enough for human to execute → fair for benchmark
- **No need to filter forward-edit failures**
- **Key insight**: Prompt adherence failures **roll up into** convergence failures
  - If model can't follow prompts → edits are effectively random → never converges
  - No need to separate prompt adherence from convergence
  - One unified metric captures both

**Elegant simplification**: "Let the metric speak" rather than pre-filtering.

**Analogy**: Random edits = zero convergence by definition.

---

### Q13: Source Images
**Question**: What source images should the benchmark use?

**Response**:
- **Broad set** preferred
- **Leverage existing benchmarks** (EditBench, COCO, etc.) if satisfactory
- If not, curate new set
- **Key criterion**: Images commonly used for editing = images with **specific intention**
  - Not random images, but realistic editing targets
- **Mix of synthetic and real images**

---

### Q14: Metrics - Quality vs Semantic
**Question**: What specific metrics for each dimension?

**Response**:

**Quality preservation**:
- **Reference-based metrics**: PSNR, SSIM, LPIPS
- Key advantage of round-trip: we have the original as reference
- This is the primary reason round-trip design was chosen

**Semantic consistency**:
- **Start with CLIP** similarity
- **Eventually**: VLM-based judge for more nuanced assessment
- VLM quantification is non-trivial → detail for later

---

### Q15: Reporting Results
**Question**: How should results be reported? Headline number, thresholds, worst-case?

**Response**:

**1. Headline number**:
- **Aggregated convergence score** that's easy to understand
- "Half-life" (iterations until X% degradation) is one option
- Should convey meaning/intention clearly

**2. Thresholds**:
- Binary pass/fail **not useful** without compelling narrative
- Need a good story before defining "convergent enough"

**3. Worst-case**:
- Include as **supplementary analysis**
- Headline should capture **overall convergence across the board**

**4. Breakdowns**:
- By edit type (object add/remove, color, style)
- By image category (faces, objects, scenes)
- These are secondary to headline but valuable for understanding

---

### Q16: Scope and Baselines
**Question**: What models to test, scale for v1, and v1 vs future scope?

**Response**:

**A) Models to test** (latest as of 2025):
| Model | Provider | Notes |
|-------|----------|-------|
| Gemini 2.5 Flash Image (Nano Banana) | Google | #1 on LMArena Image Edit Arena |
| FLUX Kontext (dev/pro/max) | Black Forest Labs | 12B params, strong character preservation |
| Qwen-Image-Edit | Alibaba | 20B, SOTA on GEdit/ImgEdit benchmarks |
| GPT-Image-1 | OpenAI | 85% accuracy, but tends to over-modify |
| Seedream 3.0/4.0 | ByteDance | #1 on Artificial Analysis Arena |

**B) Scale for v1**:
- ~100 images
- Multiple edit types (TBD exact categories)
- Vary N for degradation curves

**C) v1 vs Future**:
- **v1**: Quality metrics (PSNR/SSIM/LPIPS) + Semantic via CLIP
- **Future**: VLM-based judge, larger scale, more edit types

---

### Q17: Edit Categories for v1
**Question**: What edit categories for v1?

**Response**:
- **Focus on clearly reversible edits** for v1
- Gives cleaner signal on quality degradation
- Avoids confounding with semantic ambiguity

**v1 Edit Categories**:
1. Add/remove accessories (glasses, hat, earrings)
2. Add/remove objects in scene
3. Flip/rotate (geometric transforms)

**Future**: Expand to partially reversible (color, expression, lighting)

---

### Q18: Benchmark Naming and Deliverables
**Question**: Benchmark name and what are the deliverables?

**Response**:

**Name**: **ConvergeBench**

**Deliverables**:
1. **Academic paper** - methodology, results, analysis
2. **Open-source evaluation harness** - reproducible toolkit for community

---

---

## Key Decisions

| Aspect | Decision |
|--------|----------|
| Core thesis | Editing systems should enable user convergence; round-trip tests controllability |
| Two dimensions | Quality preservation + Semantic consistency |
| Why round-trip | Original as reference → reference-based IQA works cleanly |
| Structure | I₀ → edit → I₁ → inverse_edit → I₁' → ... (vary N, plot curves) |
| v1 Edits | Clearly reversible: add/remove accessories, objects, geometric |
| v1 Scale | ~100 images, existing datasets if suitable |
| Metrics | Quality: PSNR/SSIM/LPIPS; Semantic: CLIP (VLM judge later) |
| Headline | Interpretable convergence score (half-life or similar) |
| Models | Gemini Flash, FLUX Kontext, Qwen-Image-Edit, GPT-Image-1, Seedream |
| Name | ConvergeBench |
| Output | Paper + open-source eval harness |

---

## Final Plan: ConvergeBench

### 1. Motivation & Thesis

**Problem**: Current image editing benchmarks evaluate single-shot edit quality, but real users edit iteratively. They don't know exactly what they want upfront—they explore through repeated edits: "make this greener" → "a bit more" → "now move it left." If image quality degrades over iterations (artifacts, blur, color drift), the editing system becomes unusable as a creative tool.

**Core Thesis**: A good editing system enables **convergence**—users can iteratively steer toward their goal without the system fighting back through:
- **Quality divergence**: Accumulated artifacts/degradation (irreversible)
- **Semantic divergence**: Unintended side effects (edit X also changes Y)

**Key Insight**: Round-trip editing (edit → inverse edit) provides a clean evaluation framework because:
1. The original image serves as ground truth reference
2. Reference-based IQA metrics (PSNR, SSIM, LPIPS) measure exactly what they're supposed to
3. Failure to return to origin indicates system "friction" that blocks convergence

**Theoretical Framing**: Inspired by Banach contraction mapping—a convergent system should act as a contraction toward the user's goal, not introduce compounding errors.

---

### 2. Benchmark Design

#### 2.1 Test Case Structure

```
Input:
  - Source image: I₀
  - Edit pair: (prompt_forward, prompt_backward)
  - Max iterations: N

Execution:
  For n = 1 to N:
    I_n = edit(I_{n-1}', prompt_forward)      # Forward edit
    I_n' = edit(I_n, prompt_backward)         # Backward edit (inverse)

Measurements at each n:
  - Quality: PSNR(I₀, I_n'), SSIM(I₀, I_n'), LPIPS(I₀, I_n')
  - Semantic: CLIP_similarity(I₀, I_n')
```

#### 2.2 Edit Categories (v1 - Clearly Reversible)

| Category | Forward Prompt | Backward Prompt | Rationale |
|----------|----------------|-----------------|-----------|
| Accessories | "add glasses" | "remove glasses" | Clear object, reversible |
| Accessories | "add hat" | "remove hat" | Clear object, reversible |
| Accessories | "add earrings" | "remove earrings" | Clear object, reversible |
| Scene objects | "add a chair on the left" | "remove the chair" | Localized change |
| Scene objects | "add a red ball" | "remove the red ball" | Specific object |
| Geometric | "flip horizontally" | "flip horizontally" | Mathematically exact inverse |
| Geometric | "rotate 90° clockwise" | "rotate 90° counter-clockwise" | Mathematically exact inverse |

**Prompt Generation**:
- Start with template-based for v1 (ensures clean inverses)
- Explore LLM-generated inverses in future versions

#### 2.3 Source Images

- **Scale**: ~100 images for v1
- **Sources**: Leverage existing editing benchmarks (EditBench, I2EBench, etc.) if suitable
- **Criteria**: Images with specific editing intention (not random stock photos)
- **Mix**: Synthetic + real images
- **Diversity**: Faces, objects, scenes

#### 2.4 Metrics

**Quality Preservation (Primary)**:
| Metric | What it measures | Implementation |
|--------|------------------|----------------|
| PSNR | Pixel-level reconstruction | Standard |
| SSIM | Structural similarity | Standard |
| LPIPS | Learned perceptual similarity | AlexNet/VGG backbone |

**Semantic Consistency**:
| Metric | What it measures | Implementation |
|--------|------------------|----------------|
| CLIP similarity | Semantic distance | CLIP ViT-L/14 |
| (Future) VLM Judge | Holistic assessment | GPT-4V / Gemini |

#### 2.5 Headline Score: Convergence Half-Life

**Definition**: Number of round-trips until quality degrades by X% from baseline.

```
Convergence Half-Life = min{n : metric(I₀, I_n') < threshold × metric(I₀, I₀)}
```

Where `metric` could be SSIM (higher = better, so threshold < 1) or LPIPS (lower = better, so threshold > 1).

**Interpretation**: "Model A sustains 12 round-trips before noticeable degradation; Model B only 4."

**Alternative**: Area under the degradation curve (AUC) for a single number.

---

### 3. Evaluation Protocol

#### 3.1 Per-Model Evaluation

1. For each (image, edit_pair) combination:
   - Run N round-trips (N = 1, 2, 3, 5, 10, 15, 20)
   - Record metrics at each n

2. Aggregate across test cases:
   - Mean degradation curve per edit category
   - Mean degradation curve overall
   - Compute half-life / AUC

#### 3.2 Reporting

**Primary**:
- Degradation curves (metric vs N) per model
- Headline convergence score (half-life or AUC)
- Model ranking table

**Secondary**:
- Breakdown by edit category
- Breakdown by image type
- Worst-case analysis (which edits degrade fastest)

---

### 4. Models to Evaluate

| Model | Provider | Access | Priority |
|-------|----------|--------|----------|
| Gemini 2.5 Flash Image (Nano Banana) | Google | API | High |
| FLUX Kontext (dev/pro/max) | Black Forest Labs | API / Local | High |
| Qwen-Image-Edit | Alibaba | API / Local | High |
| GPT-Image-1 | OpenAI | API | High |
| Seedream 3.0/4.0 | ByteDance | API | Medium |

---

### 5. Deliverables

#### 5.1 Academic Paper

**Structure**:
1. Introduction: Problem of iterative editing, convergence thesis
2. Related Work: Existing benchmarks, cycle consistency in GANs/diffusion
3. ConvergeBench: Design, metrics, protocol
4. Experiments: Results across models
5. Analysis: What causes divergence? Quality vs semantic breakdown
6. Discussion: Implications for model design
7. Conclusion

**Target Venue**: NeurIPS / CVPR / ICLR (Datasets & Benchmarks track)

#### 5.2 Open-Source Evaluation Harness

```
convergebench/
├── data/
│   ├── images/              # Source images
│   └── edit_pairs.json      # (forward, backward) prompt pairs
├── models/
│   ├── gemini.py            # API wrapper
│   ├── flux.py              # API/local wrapper
│   ├── qwen.py              # API/local wrapper
│   └── ...
├── metrics/
│   ├── quality.py           # PSNR, SSIM, LPIPS
│   └── semantic.py          # CLIP similarity
├── eval/
│   ├── run_roundtrip.py     # Core evaluation loop
│   ├── compute_metrics.py   # Aggregate results
│   └── plot_curves.py       # Visualization
├── results/                 # Output directory
└── README.md
```

**Features**:
- Easy to add new models
- Configurable N, edit categories
- Automatic curve plotting and reporting
- Reproducible with fixed seeds

---

### 6. Roadmap

#### Phase 1: Foundation (v0.1)
- [ ] Curate 100 source images
- [ ] Define 20-30 edit pairs (templates)
- [ ] Implement evaluation harness skeleton
- [ ] Integrate 2-3 models (Gemini, FLUX, Qwen)
- [ ] Run initial experiments, validate metrics

#### Phase 2: Full v1
- [ ] Expand to all 5 target models
- [ ] Refine edit categories based on Phase 1 findings
- [ ] Compute headline scores, generate curves
- [ ] Write paper draft

#### Phase 3: Release
- [ ] Clean up codebase for public release
- [ ] Create documentation and examples
- [ ] Submit paper
- [ ] Release benchmark publicly

#### Future (v2+)
- [ ] VLM-based semantic judge
- [ ] Expand to partially reversible edits (color, expression)
- [ ] Scale to 500+ images
- [ ] Public leaderboard

---

### 7. Open Questions for Future Exploration

1. **Convergence metric refinement**: Is half-life the best headline? Alternatives: AUC, threshold-based pass rate
2. **Edit difficulty scoring**: Can we automatically classify edit reversibility?
3. **Cross-edit consistency**: Does doing edit A then B then undo both return to origin?
4. **Model-specific failure modes**: What causes each model to diverge?
5. **Theoretical bounds**: Is there a fundamental limit to convergence for diffusion-based editing?

---

*Document generated: 2026-01-27*
*Status: Ready for implementation*
