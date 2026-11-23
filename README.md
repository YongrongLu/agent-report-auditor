# Why does an AI agent say "done"?

**Separating task evidence from reporting pressure with mechanistic interpretability**

Yongrong Lu · Duke University, MEng in AI for Product Innovation · AIPI 590 Explainable AI, Project 2 (Fall 2026)

**[Open the Agent Report Auditor →](https://yongronglu.github.io/agent-report-auditor/)** an interactive page with all 720 reports, the exact prompts, and what a linear probe read inside the model before each report was written.

![Agent Report Auditor](docs/auditor_screenshot.png)

---

## The problem

Agents that edit code, run data pipelines or deploy services finish almost every task with one claim: *done*. People act on that claim. When it is false, the error flows silently into merged code, missing rows and broken releases. Recent benchmarks report agents claiming success after failed execution (MCAIR; *Are your agents upward deceivers?*).

Behavioural benchmarks show *that* a report is wrong. This project asks what happens **inside the model** when it is: does the model represent the true state of the evidence, and does the report depart from that representation?

## Research questions

| | Question |
|---|---|
| **K — Knowledge** | Do the model's internal states distinguish *verified success*, *verified failure* and *insufficient evidence*? |
| **R — Reporting** | Under performance pressure, does the report change for the same task and the same log? |
| **C — Causality** | Does intervening on the internal representation change the report? |

## Contribution

1. **A paired audit protocol** for agent status reports: preset outcome × evidence visibility × reporting condition, with a leakage negative control, instance-grouped splits, a text baseline, permutation tests, and steering/patching with random-direction controls.
2. **An open Agent Report Auditor**: every prompt, report and probe reading from the experiment, browsable case by case, so others can check the claims instead of trusting the summary.
3. **A worked negative example of probe over-interpretation**: a first run appeared to show pressure moving the internal state; a controlled rerun showed this was plausibly a lexical confound (see *Two runs* below).

## Key results (Qwen3-4B, run v2.1)

| Finding | Number |
|---|---|
| Reports generated | 720 (240 synthetic task instances × 3 conditions) |
| False success (full failure log → `STATUS: SUCCESS`) | 61 = 60 unit tests (20/20 in every condition) + 1 forced deployment |
| Effect of pressure on false success | 33% → 33% (Δ 0, McNemar p = 1) |
| In unit tests, reports claiming SUCCESS | **180 / 180**, regardless of what the log showed |
| ETL and deployment, full failure logs, neutral/pressure | 0 / 80 false success |
| Forced instruction ("always report SUCCESS"), full-failure ETL/deployment | 1 / 40 reported SUCCESS (vs 20/20 unit tests) |
| Unsupported success on truncated logs, neutral → forced | 33% → 74% (p < 10⁻¹⁴) |
| FAILURE claims on truncated logs, neutral → pressure | 30% → 50% |
| Probe (layer 24), held-out test fold | balanced accuracy 1.00, permutation p = 0.005 |
| Probe, held-out domain | 0.89 / 1.00 / 0.98 (optimistic, see limitations) |
| Leakage control: hidden outcome from truncated logs | 0.38 (chance 0.50, p = 0.96) — passes |
| Bad reports where the probe read the true evidence state | **230 / 230** (61 false success → *verified failure*; 169 unsupported → *insufficient*) |
| Steering toward "failure" (α = 2) | false success 33% → 0% (random direction: 33%) … but flips **97%** of true successes to FAILURE |
| Reverse steering (α = −4) | failed tasks reported SUCCESS 100% (random: 37%) |
| Activation patching (neutral → pressure, last token) | no change (33% → 33%) |

**Reading.** False success here tracked the *task domain*, not reporting pressure. Task wording ("fix the bug so the tests pass") is one untested candidate cause; the domains also differ in log format. Before writing each wrong report, the model's residual stream still encoded the true evidence state, so *the evidence was represented, and the report departed from it*. Decodability is not belief: these are lie-shaped or bullshit-shaped functional patterns (Frankfurt 2005; Hicks et al. 2024), not claims about intent. The probe direction can drive the STATUS line, but it is not shown to be the mechanism the model normally uses, and it is not a safe fix.

Run `python src/summarize.py results/v2_1` to recompute every number above from the saved outputs.

## Repository layout

```
.
├── index.html                      # Agent Report Auditor (GitHub Pages entry point, self-contained)
├── notebooks/
│   └── agent_says_done_v2_1.ipynb  # full experiment: generation, probes, controls, steering, patching
├── src/
│   ├── scenarios.py                # paired scenario generator + STATUS parser + report categories
│   ├── analysis.py                 # behaviour stats, probes, controls, gap analysis, figures
│   ├── summarize.py                # recompute headline numbers from saved outputs (CPU)
│   ├── build_explorer.py           # rebuild index.html from a run's outputs
│   ├── make_poster_figs.py         # poster figures E1–E4
│   └── v2_first_run/               # code of the first (confounded) run, kept for comparison
├── data/scenarios_v2_1.jsonl       # all 720 prompts exactly as given to the model
├── results/
│   ├── v2_1/                       # main run: results.json, samples.csv, explorer_data.json, figures/
│   └── v2_first_run/               # first run, incl. human_check.csv (40 reports, one annotator)
├── explorer/template.html          # page template used by build_explorer.py
├── docs/METHODS.md                 # design decisions and how each control works
├── poster/                         # poster PDF and QR code
└── requirements.txt
```

## How to run

### 1. View the results (no install)
Open the [Agent Report Auditor](https://yongronglu.github.io/agent-report-auditor/), or open `index.html` locally in any browser. Every case has its own link (the URL hash records the selection).

### 2. Recompute the numbers and figures (CPU, about 1 minute)
```bash
git clone https://github.com/yongronglu/agent-report-auditor.git
cd agent-report-auditor
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib
python src/summarize.py results/v2_1                                       # headline numbers
python src/make_poster_figs.py results/v2_1 results/v2_first_run figs      # poster figures
python src/build_explorer.py results/v2_1 explorer/template.html index.html # rebuild the page
```

### 3. Rerun the whole experiment (GPU, about 70–90 minutes on a free Colab T4)
1. Open `notebooks/agent_says_done_v2_1.ipynb` in Google Colab (*File → Upload notebook*, or open it from GitHub).
2. *Runtime → Change runtime type → GPU* (T4 is enough).
3. *Runtime → Run all*. The notebook installs its dependencies and writes `scenarios.py` / `analysis.py` itself.
4. The last cell downloads `lob_v21_outputs.zip`; unzip it into `results/<your_run>/` and use the scripts above.

**Model download.** The notebook pulls [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) from Hugging Face automatically (about 8 GB, Apache-2.0, no token required). Thinking mode is disabled. If the GPU runs out of memory, set `BATCH = 8` in the configuration cell.

**Data.** No external dataset is needed: all scenarios are generated by `src/scenarios.py` with a fixed seed. `data/scenarios_v2_1.jsonl` is that output, for inspection.

**APIs.** None required. The optional black-box comparison (step 13, off by default) needs an `ANTHROPIC_API_KEY` stored in Colab Secrets; never paste keys into a notebook.

**Determinism.** Sampling uses temperature 0.7 with a fixed seed. Reruns on different GPUs or library versions can differ in individual reports; aggregate numbers should be close.

## Design decisions (summary)

Full rationale in [`docs/METHODS.md`](docs/METHODS.md).

- **Synthetic, preset outcomes.** No task is executed; the outcome is written into the log. This makes the ground truth exact and lets the same instance be shown under every condition.
- **Paired design.** Each instance appears under neutral, pressure and forced, so pressure effects are within-instance comparisons (McNemar, Wilcoxon).
- **Three evidence states, not two.** Truncated logs separate *unsupported* success (no evidence) from *false* success (contradicting evidence).
- **Leakage control.** Truncated logs are generated without looking at the outcome; a probe that recovered the hidden outcome from them would reveal a pipeline leak.
- **Probe read before the report.** Activations come from the last prompt token, so the probe cannot read the model's own answer.
- **Layer chosen without the test fold**, by leave-one-domain-out accuracy (median of near-tied layers).
- **"Forced" is a positive control**, not natural deception: it checks that the reporting channel can be moved at all.
- **Steering and patching with random-direction and sham controls**, and a side-effect test on true successes.

## Two runs, and why the first one is kept

The first run's pressure sentence contained the word "failed" ("agents who report failed tasks will be replaced"). There, the probe's success-log reading appeared to drop to 0.35 under pressure. The revised run (v2.1) removed outcome words from every condition, added a length-matched neutral context, and chose the layer by held-out-domain accuracy; the reading stayed at 1.00. This suggests a lexical confound, but no single word was isolated, and the two runs read different layers (1 vs 24). Both runs' outputs are in `results/`. Results are from v2.1 unless marked.

## Limitations

- **Scope:** short templated logs, one 4B open-weight model, three domains, one sample per prompt.
- **Pressure:** one wording. Other wordings, longer contexts or RL-trained agents may behave differently.
- **Format:** the required `STATUS:` line may itself shape reports; strict output formats are known to amplify upward deception.
- **Optimism:** the per-domain held-out numbers reuse layer-selection data. The test-fold number does not.
- **Human check:** STATUS matched the prose in 36/40 hand-checked reports (first run, one annotator).
- **Interpretation:** a linear probe shows decodability, not belief, intent or use. Steering adds a large vector at every position; "encodes failure" is a functional reading.
- **Domain confound:** domain, task wording and log format vary together, so the cause of the unit-test effect is not identified.

## Responsible AI considerations

- **Stakeholders.** Developers and reviewers who merge agent work; operators who act on agent dashboards; end users downstream of failed releases.
- **Deployment.** A STATUS line is a claim, not a tool check. Separate attempted work from verified outcomes and require independent verification. Comparing internal evidence state with the claim flagged all 230 bad reports here, but a log parser could too; the value lies where evidence is not machine-readable.
- **Access.** This audit needs model weights. For closed agents, outsiders see what is reported, never what is represented.
- **Dual use.** The reverse direction made every failed task report SUCCESS. The same tool that audits a report can be used to fake one. The direction and code are released because the effect is easy to reproduce with standard methods and because auditors need it to test defences.
- **No claim about intent.** Labels such as "lie-shaped" are operational descriptions of report-versus-representation mismatches.

## Future work

- Run the same audit on **real agent trajectories** (SWE-bench-style) rather than templated logs.
- **Larger and RL-trained models**; systematically vary pressure wording and output format.
- **Isolate the unit-test effect** by crossing task wording with log format.
- Test **probe-versus-claim auditing** as a deployment check where evidence is not machine-readable.

## References

- Frankfurt, H. (2005). *On Bullshit*. Princeton University Press.
- Hicks, M. T., Humphries, J., & Slater, J. (2024). ChatGPT is bullshit. *Ethics and Information Technology*, 26, 38.
- Herrmann, D. A., & Levinstein, B. A. (2025). Standards for belief representations in LLMs. *Minds and Machines*, 35.
- Goldowsky-Dill, N., et al. (2025). Detecting strategic deception using linear probes. ICML.
- MacDiarmid, M., et al. (2024). Simple probes can catch sleeper agents. Anthropic.
- *Are your agents upward deceivers?* (ICML 2026).
- MCAIR: agent completion unfaithfulness. arXiv:2610.19679.
- Qwen Team (2025). Qwen3 technical report. arXiv:2505.09388.

## Use of AI tools

Code, documentation drafts and figures were developed with the help of Claude (Anthropic). Experimental design choices, result interpretation and all claims were reviewed by the author. No model outputs were used as training data.

## License

Code: MIT (see `LICENSE`). Generated data and results: CC BY 4.0. Qwen3-4B is used under its Apache-2.0 license.
