---
title: C.O.R.E. AI
emoji: ⛏️
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
python_version: "3.11"
app_file: app_premium.py
pinned: false
---
# C.O.R.E. AI

**Collaborative Operational Risk Evaluator** — a four-agent AI system that produces an institutional-grade ESG due-diligence report for a mining prospect in minutes instead of weeks.

Built for the **lablab.ai AMD AI Hackathon 2026 — Track 1: AI Agents \& Agentic Workflows**, running on an **AMD Instinct MI300X** GPU via the AMD Developer Cloud.

\---

## Table of Contents

* [What It Does](#what-it-does)
* [Why It Matters](#why-it-matters)
* [How It Works](#how-it-works)
* [Repository Layout](#repository-layout)
* [Quick Start](#quick-start)
* [Using the App](#using-the-app)
* [Using the CLI](#using-the-cli)
* [Adding Your Own Prospects](#adding-your-own-prospects)
* [Configuration Reference](#configuration-reference)
* [Standards Referenced](#standards-referenced)
* [Tech Stack](#tech-stack)
* [License \& Acknowledgements](#license--acknowledgements)

\---

## What It Does

Pick a mining prospect from the sidebar, click **Run Evaluation**, and four specialist AI agents run in sequence on AMD MI300X to produce a complete pre-Scoping screening report:

1. **Senior Exploration Geologist** — JORC 2012-compliant resource classification, deposit analogue comparison, ore body geometry, ARD/metallurgical pre-screen.
2. **Mining \& Environmental Engineer** — IFC PS1–PS8, GISTM, ICMM, and ILO 169 / FPIC ESG pre-assessment with explicit RESOLVABLE / BORDERLINE / POTENTIALLY FATAL verdicts on every material risk.
3. **Senior Mining Economist** — Scoping-level CAPEX/OPEX with the mandatory ±40% uncertainty label, Bear / Base / Bull DCF scenarios, ESG cost itemisation, and a project-financing structure proposal.
4. **Chief Operating Officer** — synthesises the three upstream reports and issues a single binding decision: **PROCEED to PEA / PFS / DFS** or **REJECT**, with measurable Binding Conditions and Stage-Reversal Triggers.

The output is rendered as a colour-coded decision banner, an ESG risk radar chart, key-rationale cards, and per-agent tabbed sections. The full report is downloadable as Markdown.

## Why It Matters

A traditional Scoping-stage consulting report from SRK, Hatch, or an equivalent firm costs **USD 50,000 – 200,000** and takes **6–12 weeks**. Most prospects screened at this stage are abandoned anyway, so the capital is spent only to learn that the prospect is not worth advancing.

At the same time, project finance under the **Equator Principles (EP4)**, **IFC Performance Standards**, **GISTM**, and **SASB Metals \& Mining** is mandatory for an industry that secures over USD 3 trillion in annual financing. A single missed flag (FPIC, biodiversity offset, tailings consequence category) can derail a multi-billion-dollar project at the final investment decision.

C.O.R.E. AI compresses this evaluation into a sub-ten-minute run on a single MI300X, while preserving the full standards-compliant structure that institutional reviewers expect.

\---

## How It Works

### Cognitive-Demand Routing

Rather than calling one general-purpose model four times, each agent is matched to the open-weight model whose architecture best fits its cognitive demands. All four models stay resident in the MI300X's 192 GB HBM3 unified memory pool, so there is no GPU swapping between agents.

|Agent|Default Model (env var)|Cognitive Demand|Why this Model|
|-|-|-|-|
|Senior Exploration Geologist|**Mixtral 8x22B** (`AMD\_MODEL\_GEO`)|Multi-domain JORC / CRIRSCO recall, ore-body geometry, deposit-analogue lookup|Mixture-of-Experts excels at activating broad multi-domain knowledge in a single pass|
|Mining \& Environmental Engineer|**Mistral Small 24B** (`AMD\_MODEL\_ENV`)|Strict template compliance across the IFC / GISTM / ICMM / ILO regulatory stack|Best instruction-following at this size; precisely fills structured templates without drift|
|Senior Mining Economist|**Qwen 2.5 72B** (`AMD\_MODEL\_ECON`)|Numerical reasoning, three-scenario DCF, ESG cost itemisation, financing structure|Strongest open-weight model for quantitative reasoning and disciplined financial prose|
|Chief Operating Officer|**Llama 3.3 70B Instruct** (`AMD\_MODEL\_COO`)|Synthesis across three reports, eight-condition fatal-flaw test, stage-gate decisioning|Best executive-style synthesis at the largest open-weight tier|

All defaults can be overridden in `.env` without touching the code (see [Configuration Reference](#configuration-reference)).

### Sequential Pipeline

```
Prospect Briefing
      |
      v
\[Geologist] -> \[Env Engineer] -> \[Economist] -> \[COO] -> Decision
      |              |                 |           |
   guardrail     guardrail         guardrail   guardrail
   (retry)       (retry)           (retry)     (retry)
```

CrewAI runs the agents in order. Each agent receives the upstream agents' outputs as context, so the COO sees all three specialist reports before making its decision.

### Self-Correcting Guardrails

Every agent's output passes through a deterministic Python validator before it is allowed downstream. If validation fails, the error message is fed back into the agent and the task is retried automatically. Guardrails currently enforce:

* **Section completeness** — every required header (e.g. `Tailings \& ARD/AMD Risk:`, `ESG Red Flags \& Resolvability Assessment:`, `DCF Valuation:`, `Key Rationale:`) must be present in order.
* **JORC discipline** — the Geologist's *Resource Confidence Classification* must explicitly assign either an Inferred Mineral Resource or a JORC clause 18 Exploration Target. A single discovery hole cannot support Indicated or Measured.
* **Real analogues** — the Geologist's *Deposit Analogue Comparison* must name at least one real producing or advanced-stage mine.
* **Geometry numbers** — the Geologist's *Ore Body Geometry Estimate* must include at least one explicit dimension in metres.
* **No LaTeX, no equations** — the Economist must write in plain business prose. Math notation (`\\frac`, `$$`, `\\sum`) is rejected.
* **Scoping uncertainty label** — every CAPEX/OPEX figure must carry the `±40%` Scoping-level uncertainty label.
* **Three DCF scenarios** — Bear, Base, and Bull must all be present.
* **Stage-gate hygiene** — the COO cannot conflate PEA, PFS, and DFS. Strings like `PEA (Pre-Feasibility Study)` are a hard fail.
* **Recommendation grammar** — the COO output must contain a line of the exact form `Recommendation: PROCEED to <PEA|PFS|DFS>` or `Recommendation: REJECT`.
* **Three numbered rationale points** — `Key Rationale:` must contain exactly three numbered points.
* **No anti-meta commentary** — phrases like *"Note: I have followed…"* are stripped and rejected; the report is the report, not a compliance acknowledgement.
* **Hard-rule contradiction check (COO)** — for bulk deposits with a drilled intersection ≥ 30 m, fatal flaw (a) *Capital-Geology Mismatch* must NOT be marked Triggered. The 114 m porphyry case is the canonical non-trigger example; if the COO trips this it is forced to retry.

### COO Decision Framework

The COO applies a three-step framework:

**Step 1 — Cross-report consistency.** Confirm the economist's mining method matches the geologist's intersection width, confirm the NPV is built on demonstrated ore volume rather than assumed tonnage, confirm capital is proportionate to current data confidence. If any are inconsistent, name the error before continuing.

**Step 2 — REJECT TEST.** REJECT is issued immediately if **any single one** of these eight conditions holds:

|#|Condition|Trigger|
|-|-|-|
|a|Capital-Geology Mismatch|Bulk deposit with < 30 m intersection, or narrow-vein with < 10 m true width|
|b|Mining Method Mismatch|Open-pit selected on a sub-30 m ore zone|
|c|Price Sub-Economic|Grade × tonnage sub-economic at any realistic long-run commodity price|
|d|Environment Unresolvable|Engineer flags `FATAL FLAW DETECTED` or rates water / habitat / ARD as POTENTIALLY FATAL|
|e|Social Licence Blocked|Documented, active indigenous opposition with no negotiated pathway|
|f|Jurisdiction Catastrophic|Armed conflict, rule-of-law breakdown, or near-certain expropriation within payback|
|g|Logistics Un-bridgeable|No road, rail, port, or grid power within economic reach|
|h|Financial Negative|Strongly negative NPV at the most optimistic realistic price and lowest defensible discount rate|

**Step 3 — STAGE SELECTION** (only if no REJECT condition fires):

* **PROCEED to PEA** if Inferred resources justify a definition campaign.
* **PROCEED to PFS** only if resources are at Indicated.
* **PROCEED to DFS** only if resources are at Measured + Indicated.

Stage-skipping is forbidden. Every PROCEED carries specific Binding Conditions and bright-line Stage-Reversal Triggers; every REJECT states exactly what evidence would need to change to reopen the file.

\---

## Repository Layout

```
AMD Hackathon/
├── coreai\_.py             # Agents, prompts, guardrails, evaluation pipeline (CLI + library)
├── app\_premium.py         # Streamlit dashboard (imports run\_evaluation from coreai\_)
├── mining\_prospects.xlsx  # Optional external dataset (overrides built-in via env var)
├── requirements.txt       # Python dependencies
├── .env.example           # Template for environment configuration
├── .streamlit/            # Streamlit theme / config
└── README.md
```

The two files that matter:

* **`coreai\_.py`** — pure-Python evaluation engine. Constructs the four LLMs (`\_build\_llms`), agents (`\_build\_agents`), tasks with attached guardrails (`\_build\_tasks`), runs the CrewAI sequential pipeline, and assembles the final Markdown report (`run\_evaluation`). Also exposes `list\_prospects()` for picker UIs and supports CLI usage.
* **`app\_premium.py`** — Streamlit front-end. Imports `run\_evaluation`, `list\_prospects`, and `\_strip\_inline\_markdown` from `coreai\_`. Provides a live per-agent progress checklist (worker thread + queue so the UI ticks while CrewAI runs), an ESG risk radar (Plotly), a colour-coded decision banner, key-rationale cards, per-agent tabbed sections, downloadable Markdown reports, and a session-local report history.

A built-in five-prospect dataset is embedded in `coreai\_.py` so the system runs out of the box with no spreadsheet:

|Prospect|Country|Commodity|Setting|
|-|-|-|-|
|Eagles Nest (Ring of Fire)|Canada|Ni-Cu-PGE|Ultramafic sill|
|Mawson (Fraser Range)|Australia|Ni-Cu-Co|Mafic intrusion under cover|
|Saddle North (Golden Triangle)|Canada|Cu-Au|Alkalic porphyry|
|Emmie Bluff (Gawler Craton)|Australia|Cu-Co|Sedimentary (Tapley Hill)|
|Obelisk (Paterson Province)|Australia|Cu-Au|Hydrothermal metasedimentary|

\---

## Quick Start

### 1\. Provision an inference server

The system expects an OpenAI-compatible chat-completion endpoint that serves the four open-weight models. The reference setup is **Ollama on an AMD MI300X droplet** in the AMD Developer Cloud:

```bash
curl -fsSL https://ollama.com/install.sh | sh
echo 'OLLAMA\_HOST=0.0.0.0:11434' | sudo tee -a /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl restart ollama

ollama pull mixtral:8x22b
ollama pull mistral-small
ollama pull qwen2.5:72b
ollama pull llama3.3
```

Verify the GPU is being used:

```bash
rocminfo | grep gfx          # should show gfx942 (MI300X)
ollama ps                    # should show 100% GPU
```

Any other OpenAI-compatible endpoint (vLLM, LM Studio, a hosted provider) works as long as the four model names resolve.

### 2\. Install the client

```bash
git clone <this-repo>
cd "AMD Hackathon"
pip install -r requirements.txt
```

`requirements.txt` pulls in: `crewai`, `langchain-openai`, `python-dotenv`, `pandas`, `openpyxl`, `pydantic`, `streamlit`, `plotly`.

### 3\. Configure `.env`

Copy `.env.example` to `.env` and fill in your endpoint:

```env
AMD\_API\_BASE\_URL="http://<droplet-ip>:11434/v1"
OPENAI\_API\_KEY="ollama-no-key-needed"

AMD\_MODEL\_GEO=mixtral:8x22b
AMD\_MODEL\_ENV=mistral-small
AMD\_MODEL\_ECON=qwen2.5:72b
AMD\_MODEL\_COO=llama3.3
```

Both `AMD\_API\_BASE\_URL` and `OPENAI\_API\_KEY` are required — `coreai\_.py` raises a clear error at startup if either is missing.

### 4\. Run

**Streamlit dashboard:**

```bash
streamlit run app\_premium.py
```

**CLI:**

```bash
python coreai\_.py "Saddle\_North\_Golden\_Triangle"
```

\---

## Using the App

`streamlit run app\_premium.py` opens a dashboard with three regions:

* **Sidebar** — agent pipeline reference, prospect picker, **Run Evaluation** button, and a session-local report history (each entry is a click-to-restore prior run).
* **Main pane during a run** — a live per-agent checklist that updates as each task completes (Geologist → Env Engineer → Economist → COO), with elapsed-seconds counter and pipeline status.
* **Main pane after a run** — colour-coded decision banner (green PROCEED-PEA, blue PROCEED-PFS, purple PROCEED-DFS, red REJECT), prospect metric cards, an ESG risk radar across six axes (Ecological, Water \& Tailings, ARD/AMD, Social Licence, Permitting, ESG Finance), three Key Rationale cards, and four tabs containing the full Geological / Environmental / Financial / COO sections. A **Download Full Report (.md)** button exports the assembled Markdown.

Selecting a different prior run from the history sidebar swaps the active report without re-running the pipeline.

## Using the CLI

`coreai\_.py` doubles as a command-line tool:

```bash
# Run an evaluation and write the report to outputs/<prospect>\_<date>.md
python coreai\_.py "Saddle\_North\_Golden\_Triangle"

# Run with no file output, print to stdout only
python coreai\_.py "Saddle\_North\_Golden\_Triangle" --no-write

# Use a custom output directory
python coreai\_.py "Saddle\_North\_Golden\_Triangle" --output-dir reports

# List available prospect names and exit
python coreai\_.py --list

# No name -> defaults to the first prospect in the dataset
python coreai\_.py
```

You can also import it as a library:

```python
from coreai\_ import list\_prospects, run\_evaluation

for p in list\_prospects():
    print(p\["name"], "->", p\["display"])

report\_md = run\_evaluation("Eagles\_Nest\_Ring\_of\_Fire")
print(report\_md)
```

`run\_evaluation` accepts an optional `progress\_callback(role: str)` that fires as each agent finishes (`"geology"`, `"env"`, `"economy"`, `"coo"`). The Streamlit app uses this hook to drive its live checklist via a thread-safe queue.

## Adding Your Own Prospects

The default five-prospect dataset is hardcoded inside `coreai\_.py` so the system runs with zero external files. To use your own data, point the `MINING\_PROSPECTS\_PATH` environment variable at a `.csv` or `.xlsx` file with these required columns:

```
Prospect\_Name, Country, Target\_Commodity, Geological\_Setting,
Best\_Drill\_Intersection, Est\_Depth\_to\_Target\_m,
Logistics\_Distance\_to\_Grid\_km, Environmental\_Context
```

`mining\_prospects.xlsx` in the repo root is a working example. Field values use underscore-joined tokens (e.g. `15m\_at\_2.0\_Percent\_Ni`) — the briefing renderer converts underscores back to spaces when building the prompt, so keep that format.

## Configuration Reference

|Variable|Purpose|Default|
|-|-|-|
|`AMD\_API\_BASE\_URL`|OpenAI-compatible endpoint URL (required)|—|
|`OPENAI\_API\_KEY`|API key for the endpoint (required; any non-empty string for Ollama)|—|
|`AMD\_MODEL\_GEO`|Model name served to the Geologist|`mixtral:8x22b`|
|`AMD\_MODEL\_ENV`|Model name served to the Environmental Engineer|`mistral-small`|
|`AMD\_MODEL\_ECON`|Model name served to the Economist|`qwen2.5:72b`|
|`AMD\_MODEL\_COO`|Model name served to the COO|`llama3.3`|
|`AMD\_API\_BASE\_URL\_GEO` / `\_ENV` / `\_ECON` / `\_COO`|Per-role base-URL overrides for split deployments (different ports/servers)|falls back to `AMD\_API\_BASE\_URL`|
|`MINING\_PROSPECTS\_PATH`|Path to a custom CSV/XLSX dataset|unset (uses the built-in 5 prospects)|

\---

## Standards Referenced

|Standard|Domain|Used By|
|-|-|-|
|JORC 2012 / CRIRSCO|Resource classification|Geologist|
|NI 43-101 (CIM 2014)|Resource classification (Canada)|Geologist|
|IFC Performance Standards (PS1–PS8)|Environmental \& social safeguards|Env Engineer, Economist|
|GISTM (2020)|Tailings storage facility management|Env Engineer|
|ICMM 10 Principles|Industry conduct|Env Engineer|
|IRMA Standard for Responsible Mining|Responsible mining certification|Env Engineer, Economist|
|ILO Convention 169 / UNDRIP / DRIPA / Native Title Act|Indigenous rights, FPIC|Env Engineer|
|Equator Principles (EP4)|Project finance|Economist|
|SASB Metals \& Mining|ESG disclosure|Economist|
|TNFD / TCFD|Nature- and climate-related financial disclosure|Env Engineer, Economist|
|Fraser Institute IAI|Jurisdiction risk premium|Economist|

## Tech Stack

* **Agentic framework**: [CrewAI](https://www.crewai.com/) — sequential `Process.sequential` with structured context passing between tasks and per-task `guardrail=` callbacks.
* **Models**: Mixtral 8x22B, Mistral Small, Qwen 2.5 72B, Llama 3.3 70B Instruct (all open-weight).
* **Inference runtime**: [Ollama](https://ollama.com/) on AMD MI300X via [ROCm](https://www.amd.com/en/products/software/rocm.html) (any OpenAI-compatible server works).
* **Validation**: [Pydantic](https://docs.pydantic.dev/) schema for the financial briefing + custom regex-based guardrails.
* **UI**: [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/python/) for the ESG risk radar.
* **Compute**: AMD Instinct MI300X (192 GB HBM3) on the AMD Developer Cloud.

## License \& Acknowledgements

This is a hackathon submission for the **lablab.ai AMD AI Hackathon 2026 — Track 1: AI Agents \& Agentic Workflows**.

* **AMD** — Developer Cloud and Instinct MI300X compute.
* **CrewAI** — multi-agent orchestration framework.
* **Mistral, Mixtral, Qwen, and Llama** open-weight communities.
* **lablab.ai** for organising the hackathon.
* **Hugging Face** for providing the space to upload the web app.



License: MIT License

