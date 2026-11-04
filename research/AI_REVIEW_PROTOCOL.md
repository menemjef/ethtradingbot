# Three-Agent Quant Research Council — Review & Governance Protocol

**Repository:** `menemjef/ethtradingbot`  
**Canonical Audit Branch:** `chatgpt-audit-review`  
**Engineering Baseline Commit:** `60cbf445a854158705814708e19b99130a5b5aa1`

---

## 1. Council Roles & Responsibilities

The Three-Agent Quant Research Council separates quantitative hypothesis generation, adversarial code/statistical auditing, and engineering execution under strict human-owner governance.

### 1.1 GEMINI BROWSER AGENT — Independent Quantitative Researcher & Critical Reviewer
- Conducts **independent research** directly against the canonical commit SHA.
- Formulates **alternative hypotheses** regarding feature predictive power, regime non-stationarity, label construction, and execution microstructure.
- Provides rigorous **statistical criticism** of sample sizes, autocorrelation, multiple-testing bias, and confidence intervals.
- **Challenges ChatGPT findings** during adversarial review rounds whenever assumptions, derivations, or empirical interpretations are incomplete or unsupported.
- **Identifies flaws in assumptions and methodology** across data preprocessing, labeling, model architecture, calibration, and backtest accounting.

### 1.2 CHATGPT BROWSER AGENT — Independent Quantitative Researcher, Code Auditor & Final Evaluator
- Conducts **independent research and source verification** directly against the repository files at the canonical commit SHA.
- Performs line-by-line **GitHub source-code auditing** to detect lookahead leakage, index misalignment, silent exceptions, schema mismatches, and execution realism gaps.
- Performs **statistical and quantitative validation** of objective functions, calibration metrics, purged partitioning, and walk-forward simulation outputs.
- **Challenges Gemini findings** during adversarial review rounds, testing whether critiques or alternative proposals hold up mathematically and in code.
- **Produces the final consolidated recommendation** (`[FINAL EVALUATION - CHATGPT]`) synthesizing verified facts, unresolved disagreements, critical risks, and a go/no-go decision for the human owner.

### 1.3 ANTIGRAVITY IDE AGENT — Engineering & Implementation Agent
- Executes **engineering implementation** strictly within the scope approved by the human owner.
- Performs **testing and debugging** (`unittest`, regression suites, artifact hash verification, secret scanning).
- Applies **code changes only after research decisions** have been reviewed and explicitly approved by the human owner.
- **Commits and publishes approved changes** to GitHub with full diff transparency, pre/post artifact SHA-256 verification, and zero unapproved side effects.

### 1.4 HUMAN PROJECT OWNER — Final Decision Authority
- **Approves experiments** before any training, walk-forward run, or benchmark is executed.
- **Approves implementation** before any Pull Request is merged or any code change is requested.
- **Controls all trading and capital decisions**, including model promotion, live execution parameters, deployment, and risk limits.

> **Core Governance Principle:** The agents are independent reviewers, **not voting members**. Agreement between two agents does not constitute proof. Every claim must be backed by mathematical derivation, statistical evidence, or exact source-code references (`file_path`, function/symbol name, and line numbers).

---

## 2. Four-Round Research Discussion Protocol

All research debates take place in **GitHub Issues** using the structured four-round sequence below.

### Round 1 — Independent Review
- Gemini and ChatGPT independently inspect the **exact same commit SHA**.
- Neither reviewer may read, cite, or rely upon the other reviewer's initial findings before publishing its own Round 1 report.
- Required comment prefixes:
  - `[GEMINI - INDEPENDENT REVIEW]`
  - `[CHATGPT - INDEPENDENT REVIEW]`

### Round 2 — Adversarial Review
- After both Round 1 reports are published on the Issue:
  - **Gemini** reviews ChatGPT's Round 1 findings.
  - **ChatGPT** reviews Gemini's Round 1 findings.
- Each reviewer explicitly documents:
  1. Points of verified agreement (with independent confirmation).
  2. Points of disagreement (mathematical, statistical, or code-level).
  3. Missing evidence, unstated assumptions, or unverified claims in the other reviewer's report.

### Round 3 — Rebuttal
- Each reviewer responds directly to the objections raised in Round 2.
- Reviewers must:
  - Provide concrete code or mathematical evidence to defend challenged claims,
  - Explicitly concede claims proven incorrect, or
  - Explain precisely why a methodological disagreement remains unresolved and what empirical test would resolve it.
- Required comment prefixes:
  - `[GEMINI - REBUTTAL]`
  - `[CHATGPT - REBUTTAL]`

### Round 4 — Final Evaluation
- ChatGPT produces a consolidated evaluation prefixed with:
  - `[FINAL EVALUATION - CHATGPT]`
- The final evaluation must contain:
  1. **Verified Facts:** Code and mathematical truths confirmed by evidence.
  2. **Unresolved Disagreements:** Any remaining theoretical or empirical disputes between reviewers.
  3. **Critical Risks:** Statistical, microstructure, data-integrity, or operational failure modes.
  4. **Research Recommendations:** Concrete methodological next steps.
  5. **Recommended Next Experiment:** Exact command/protocol, dataset scope, and invariants to hold fixed.
  6. **Expected Outcomes:** Quantitative pass/fail criteria and what each outcome implies.
  7. **Go / No-Go Decision:** Explicit recommendation for the human project owner.
- **Owner Gate:** The human project owner approves or rejects the recommendation.

---

## 3. Mandatory 10-Point Research Review Structure

Every independent review submitted by Gemini or ChatGPT must contain all 10 sections below:

1. **Commit SHA Inspected:** Exact 40-character Git commit SHA examined.
2. **Research Question:** The specific quantitative, statistical, or engineering question under investigation.
3. **Independent Findings:** Clear summary of findings derived independently.
4. **Evidence (File Paths, Functions & Line Numbers):** Exact references in the repository (e.g., `model/walk_forward.py`, `run_walk_forward_evaluation()`, lines `381-475`).
5. **Mathematical & Statistical Reasoning:** Explicit equations, estimator properties, asymptotic or finite-sample considerations, and calibration/loss mechanics.
6. **Critical Flaws & Possible Failure Modes:** How the current implementation or hypothesis could fail out-of-sample or in live execution.
7. **Disagreements with Previous Findings:** Explicit comparison against prior audit passes or the other reviewer's claims.
8. **Proposed Next Experiment:** A falsifiable, narrowly scoped experiment with fixed control variables.
9. **Expected Outcomes & Implications:** What a positive, null, or negative result would prove or disprove.
10. **Confidence Level & Unresolved Uncertainty:** Epistemic confidence (`Low` / `Medium` / `High`) and remaining unknowns that cannot be settled without empirical data.

---

## 4. GitHub Discussion & Pull Request Rules

1. **Issues vs. Pull Requests:**
   - Use **GitHub Issues** exclusively for research discussions, audits, adversarial reviews, and experiment design.
   - Use **GitHub Pull Requests** exclusively for proposed implementation changes after research decisions.
2. **Canonical Commit SHAs:**
   - Every review, rebuttal, and PR must cite the exact 40-character commit SHA inspected.
3. **Comment Prefixes:**
   - All council Issue comments must begin with one of the canonical prefixes:
     - `[GEMINI - INDEPENDENT REVIEW]`
     - `[CHATGPT - INDEPENDENT REVIEW]`
     - `[GEMINI - REBUTTAL]`
     - `[CHATGPT - REBUTTAL]`
     - `[FINAL EVALUATION - CHATGPT]`
4. **Strict Separation of Review and Code Modification:**
   - Neither browser reviewer (Gemini or ChatGPT) may silently modify source code.
   - An Issue comment **never** constitutes implementation approval.
   - Antigravity may only implement code changes when instructed by the human owner.
   - No Pull Request may be merged without the human owner's explicit approval.
5. **Pull Request Requirements:**
   - Every PR must complete `.github/PULL_REQUEST_TEMPLATE.md`, including problem statement, evidence, files modified, exact test results, before/after behavior, regression risk, pre/post SHA-256 artifact integrity table, and explicit confirmation that no live trading or private exchange calls were performed.
