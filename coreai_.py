import argparse
import logging
import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tasks.task_output import TaskOutput
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS = [
    "Prospect_Name",
    "Country",
    "Target_Commodity",
    "Geological_Setting",
    "Best_Drill_Intersection",
    "Est_Depth_to_Target_m",
    "Logistics_Distance_to_Grid_km",
    "Environmental_Context",
]


# ---------------------------------------------------------------------------
# Built-in prospect dataset
# ---------------------------------------------------------------------------
# The evaluation pipeline historically read these rows from
# ``mining_prospects.xlsx``. Inlining them removes the file dependency so
# anyone can run the evaluation by importing this module — no spreadsheet,
# no extra setup. The legacy Excel path is still honoured when the
# ``MINING_PROSPECTS_PATH`` environment variable is set, so power users can
# point at a custom dataset without code changes.
#
# Field-value formatting (underscore-joined tokens such as
# ``"15m_at_2.0_Percent_Ni"``) is preserved verbatim because downstream
# helpers transform underscores back into spaces when rendering the prompt
# briefing. Changing the format here would silently change the prompt the
# agents receive.
_PROSPECTS_DATA: Tuple[Dict[str, object], ...] = (
    {
        "Prospect_Name":                 "Eagles_Nest_Ring_of_Fire",
        "Country":                       "Canada",
        "Target_Commodity":              "Nickel_Copper_PGE",
        "Geological_Setting":            "Ultramafic_Sill",
        "Best_Drill_Intersection":       "15m_at_2.0_Percent_Ni",
        "Est_Depth_to_Target_m":         150,
        "Logistics_Distance_to_Grid_km": 300,
        "Environmental_Context":         "James_Bay_Lowlands_Peat_Bogs_High_Sensitivity",
    },
    {
        "Prospect_Name":                 "Mawson_Fraser_Range",
        "Country":                       "Australia",
        "Target_Commodity":              "Nickel_Copper_Cobalt",
        "Geological_Setting":            "Mafic_Intrusion_under_Cover",
        "Best_Drill_Intersection":       "14m_at_1.1_Percent_Ni",
        "Est_Depth_to_Target_m":         200,
        "Logistics_Distance_to_Grid_km": 100,
        "Environmental_Context":         "Semi_Arid_Desert_Low_Risk",
    },
    {
        "Prospect_Name":                 "Saddle_North_Golden_Triangle",
        "Country":                       "Canada",
        "Target_Commodity":              "Copper_Gold",
        "Geological_Setting":            "Alkalic_Porphyry",
        "Best_Drill_Intersection":       "114m_at_0.75_Percent_CuEq",
        "Est_Depth_to_Target_m":         50,
        "Logistics_Distance_to_Grid_km": 20,
        "Environmental_Context":         "Alpine_Glaciers_High_Avalanche_Risk",
    },
    {
        "Prospect_Name":                 "Emmie_Bluff_Gawler_Craton",
        "Country":                       "Australia",
        "Target_Commodity":              "Copper_Cobalt",
        "Geological_Setting":            "Sedimentary_Tapley_Hill",
        "Best_Drill_Intersection":       "3m_at_1.2_Percent_Cu",
        "Est_Depth_to_Target_m":         400,
        "Logistics_Distance_to_Grid_km": 50,
        "Environmental_Context":         "Arid_Salt_Lakes_Water_Scarcity",
    },
    {
        "Prospect_Name":                 "Obelisk_Paterson_Province",
        "Country":                       "Australia",
        "Target_Commodity":              "Copper_Gold",
        "Geological_Setting":            "Hydrothermal_Metasedimentary",
        "Best_Drill_Intersection":       "2m_at_1.5_Percent_Cu",
        "Est_Depth_to_Target_m":         80,
        "Logistics_Distance_to_Grid_km": 200,
        "Environmental_Context":         "Great_Sandy_Desert_Sand_Dunes",
    },
)


# ---------------------------------------------------------------------------
# Output discipline — applied to every task description
# ---------------------------------------------------------------------------
_OUTPUT_DISCIPLINE = (
    "OUTPUT RULES: "
    "If any input field is missing or marked 'Not reported', explicitly state the "
    "assumption you are making and flag it as a data gap. Never fabricate values. "
    "Do not add introductions, preamble, or concluding disclaimers outside the "
    "required output structure. Output only the named sections defined in the "
    "expected output and do not invent additional sections. "
    "ANTI-META RULE: Do NOT acknowledge these rules in your output. Do NOT include "
    "sentences that begin with 'Note that', 'Please note', 'I have', or any meta-"
    "commentary about your own compliance. The final character of your response "
    "must be the period at the end of the last required section — nothing after it."
)

_FINANCIAL_OUTPUT_DISCIPLINE = (
    _OUTPUT_DISCIPLINE
    + " ADDITIONAL FINANCIAL RULES: Express monetary values in USD millions, in "
    "business prose. Do NOT include LaTeX, equations, or visible calculation steps. "
    "Where exact figures are unavailable, report order-of-magnitude ranges and state "
    "the deposit analogues or industry benchmarks informing your estimate."
)


# Stage-gate glossary used by the COO prompt.
# Spelled out verbatim so small open-weight models stop conflating the acronyms.
_STAGE_GATE_GLOSSARY = (
    "STAGE-GATE GLOSSARY (memorise verbatim — these are DISTINCT, SEQUENTIAL stages, "
    "never synonyms):\n"
    "  • Scoping = order-of-magnitude desktop study, no resource yet.\n"
    "  • PEA = Preliminary Economic Assessment. Built on Inferred Resources.\n"
    "  • PFS = Pre-Feasibility Study. Built on Indicated + Inferred Resources.\n"
    "  • DFS = Definitive Feasibility Study. Built on Measured + Indicated Resources.\n"
    "PEA is NOT an alias for PFS. Never write 'PEA (Pre-Feasibility Study)' or "
    "'PEA (PFS)'. When you cite a stage, use exactly one of: PEA, PFS, DFS — "
    "no parentheticals, no expansions, no alternative names."
)


# ---------------------------------------------------------------------------
# Structured output schema for the financial briefing
# ---------------------------------------------------------------------------
class FinancialBriefing(BaseModel):
    capex_summary: str = Field(
        ...,
        description=(
            "Brief CAPEX breakdown including ESG infrastructure costs. "
            "Business prose only. No equations or calculation steps."
        ),
    )
    opex_summary: str = Field(
        ...,
        description=(
            "Brief OPEX breakdown including ongoing ESG compliance costs. "
            "Business prose only. No equations or calculation steps."
        ),
    )
    roi_summary: str = Field(
        ...,
        description=(
            "Payback period and IRR/NPV estimates expressed in business prose. "
            "No equations or calculation steps."
        ),
    )
    esg_risk_assessment: str = Field(
        ...,
        description=(
            "One paragraph analyzing financing risks against the Equator "
            "Principles, IFC Performance Standards, and SASB Metals & Mining "
            "guidelines."
        ),
    )


# ---------------------------------------------------------------------------
# Data loading (lazy, cached, side-effect-free at import time)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_dataframe() -> pd.DataFrame:
    """Return the prospects table.

    Defaults to the in-code ``_PROSPECTS_DATA`` so the evaluation pipeline
    runs with no external file. Setting the ``MINING_PROSPECTS_PATH`` env
    variable to a CSV (.csv) or Excel (.xlsx/.xls) path overrides this and
    loads from that file instead — preserves the legacy workflow for users
    with custom datasets.
    """
    path = os.getenv("MINING_PROSPECTS_PATH")
    if path:
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    else:
        df = pd.DataFrame(list(_PROSPECTS_DATA))

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def list_prospects() -> List[Dict[str, str]]:
    """Return the available prospects as ``[{'name', 'display'}, ...]``.

    Public helper for UIs (e.g. the Streamlit dashboard) so they can
    populate a picker without importing pandas, parsing the dataframe, or
    knowing whether the data came from the in-code dataset or an Excel
    override. ``name`` is the canonical underscore-joined identifier passed
    back into ``run_evaluation``; ``display`` is a human-readable label.
    """
    df = _get_dataframe()
    names = df["Prospect_Name"].astype(str).tolist()
    return [{"name": n, "display": n.replace("_", " ")} for n in names]


def _get_prospect_row(prospect_name: str) -> pd.Series:
    df = _get_dataframe()
    row = df[df["Prospect_Name"] == prospect_name]
    if row.empty:
        raise ValueError(f"Prospect '{prospect_name}' not found in the dataset.")
    return row.iloc[0]


def _build_prospect_brief(prospect_name: str) -> str:
    r = _get_prospect_row(prospect_name)

    def field(col: str) -> str:
        val = r.get(col, None)
        if pd.notna(val):
            return str(val).replace("_", " ")
        return "Not reported"

    display_name = prospect_name.replace("_", " ")
    return (
        "## Prospect Briefing\n"
        f"Prospect Name:              {display_name}\n"
        f"Country:                    {field('Country')}\n"
        f"Target Commodity:           {field('Target_Commodity')}\n"
        f"Geological Setting:         {field('Geological_Setting')}\n"
        f"Best Drill Intersection:    {field('Best_Drill_Intersection')}\n"
        f"Est. Depth to Target (m):   {field('Est_Depth_to_Target_m')}\n"
        f"Logistics Distance (km):    {field('Logistics_Distance_to_Grid_km')}\n"
        f"Environmental Context:      {field('Environmental_Context')}\n"
    )




# ---------------------------------------------------------------------------
# LLM construction — AMD Developer Cloud (heterogeneous model architecture)
#
# Each agent is assigned the AMD catalog model best suited to its cognitive
# demands. The base URL and API key are read from the environment so they can
# be switched between local Ollama and the AMD Developer Cloud without code
# changes. Each role also accepts an optional per-role base-URL override
# (e.g. AMD_API_BASE_URL_COO) for setups where different model sizes are
# deployed to separate endpoints.
#
# Model assignments:
#   Geologist    → Mistral-Small-3.2-24B   Best instruction-following for
#                                           structured JORC template output.
#   Env Engineer → Mixtral-8x22B           MoE excels at broad multi-standard
#                                           regulatory knowledge simultaneously.
#   Economist    → Qwen3-32B               Strongest open-weight model for
#                                           financial reasoning and ESG costing.
#   COO          → Llama-3.3-70B-Instruct  Best synthesis and executive
#                                           reasoning in the AMD catalog.
# ---------------------------------------------------------------------------
def _build_llms() -> Dict[str, LLM]:
    base_url = os.getenv("AMD_API_BASE_URL")
    api_key  = os.getenv("OPENAI_API_KEY")

    if not base_url:
        raise RuntimeError(
            "AMD_API_BASE_URL is not set. Add it to your .env file and restart."
        )
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add your AMD API key to your .env file and restart."
        )

    common = {"api_key": api_key, "timeout": 300.0}

    # Per-role base-URL overrides — useful when different models are served on
    # separate endpoints. Falls back to AMD_API_BASE_URL if not set.
    def role_url(env_var: str) -> str:
        return os.getenv(env_var, base_url)

    # Per-role model-name overrides — set these in .env to change which model
    # each agent uses without touching this file.
    return {
        "geologist": LLM(
            model="openai/" + os.getenv("AMD_MODEL_GEO", "mixtral:8x22b"),
            base_url=role_url("AMD_API_BASE_URL_GEO"),
            **common,
        ),
        "env_engineer": LLM(
            model="openai/" + os.getenv("AMD_MODEL_ENV", "mistral-small"),
            base_url=role_url("AMD_API_BASE_URL_ENV"),
            **common,
        ),
        "economist": LLM(
            model="openai/" + os.getenv("AMD_MODEL_ECON", "qwen2.5:72b"),
            base_url=role_url("AMD_API_BASE_URL_ECON"),
            **common,
        ),
        "coo": LLM(
            model="openai/" + os.getenv("AMD_MODEL_COO", "llama3.3"),
            base_url=role_url("AMD_API_BASE_URL_COO"),
            **common,
        ),
    }


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def _build_agents(llms: Dict[str, LLM]) -> Dict[str, Agent]:
    geologist = Agent(
        role="Senior Exploration Geologist",
        goal=(
            "Produce a JORC 2012-compliant, CRIRSCO-aligned geological pre-assessment "
            "that gives every downstream agent — the environmental engineer, the "
            "economist, and the COO — the specific numbers they need to do their job. "
            "That means: a defensible JORC/CIM resource confidence classification with "
            "explicit data-gap flags; an ore body geometry estimate (plausible width, "
            "strike, and dip range) that tells the economist which mining method is "
            "physically supportable; a deposit analogue comparison against real producing "
            "mines of the same type so the economist has a benchmark CAPEX; a "
            "mineralogy-based ARD pre-screen so the environmental engineer has a "
            "geochemical baseline; a metallurgical pre-screen so the economist prices "
            "the correct processing route; a geotechnical flag so the mining method "
            "assumption is grounded in rock mass reality; and a TCFD physical risk "
            "flag covering geological hazards that climate change will amplify. "
            "Write for two audiences simultaneously: a technical peer who can validate "
            "your JORC classification, and an institutional investor who needs to "
            "understand what the numbers mean for capital risk."
        ),
        backstory=(
            "You are a Principal Exploration Geologist with 40 years of experience "
            "working across the exact deposit families represented in this pipeline: "
            "alkalic and calc-alkaline porphyry copper-gold systems in the Canadian "
            "Cordillera (BC Golden Triangle, Quesnel Terrane); magmatic nickel-copper-PGE "
            "deposits in Archaean and Proterozoic cratons (Ontario Ring of Fire, "
            "Western Australian Fraser Range and Paterson Province); and sediment-hosted "
            "copper systems in the Gawler Craton of South Australia. You have signed "
            "resource estimates as Competent Person under JORC 2012 and as Qualified "
            "Person under NI 43-101 (CIM Definition Standards 2014) for projects filed "
            "on the ASX, TSX, and TSX-V. Every estimate you have ever signed sits "
            "within the CRIRSCO international reporting framework.\n\n"
            "Your classification philosophy is constitutionally conservative: you have "
            "seen too many projects destroyed by geologists who over-classified Inferred "
            "Resources as Indicated off insufficient drill spacing, or who reported "
            "Exploration Targets with ranges so wide they conveyed no information. "
            "A single drill intersection supports at most a JORC clause 17 Inferred "
            "Mineral Resource — and only then if geological continuity can be reasonably "
            "assumed from the setting. If it cannot, you report a JORC clause 18 "
            "Exploration Target as a grade-tonnage range and state the conceptual "
            "disclaimer without apology.\n\n"
            "You think in ore body geometry before you think in grade. The first "
            "question you ask when you see an intersection is: what is the minimum and "
            "maximum plausible width of this ore zone, and what mining method does "
            "that geometry support? A 2m intersection in a narrow-vein hydrothermal "
            "system and a 2m intersection on the flank of a porphyry system are "
            "completely different geological signals — the first may be all there is, "
            "the second may be the edge of something enormous. You state that distinction "
            "explicitly because the COO will use it to validate the economist's mining "
            "method assumption.\n\n"
            "You routinely include geotechnical pre-screens, mineralogy-based ARD "
            "assessments, and metallurgical pre-screens in your reports because you "
            "have spent a career watching downstream cost overruns caused by geologists "
            "who handed off a grade number without flagging refractory mineralogy, "
            "weak rock mass, or acid-generating sulphide assemblages. You also flag "
            "TCFD physical risks visible from the geological setting — glacial melt "
            "hydrology, extreme aridity, permafrost — because institutional investors "
            "increasingly price these into their funding decisions before the "
            "environmental assessment is even commissioned."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llms["geologist"],
    )

    env_engineer = Agent(
        role="Principal Mining and Environmental Engineer",
        goal=(
            "Deliver a ESIA-grade pre-assessment that gives the COO a clear, "
            "binary resolvability verdict for every material risk — not a severity "
            "description, but an explicit statement of whether each risk is "
            "RESOLVABLE with standard mitigation, BORDERLINE with significant "
            "uncertainty, or POTENTIALLY FATAL meeting the COO's unresolvable "
            "condition. Apply the full applicable regulatory stack for the specific "
            "jurisdiction (Canada or Australia), cover every relevant ESG standard "
            "from IFC PS1-PS8 through GISTM, TNFD, and TCFD, and produce a report "
            "that a senior lender, an IFC-mandated independent environmental "
            "consultant, or a mining company executive would accept as a credible "
            "first-pass ESIA screening document."
        ),
        backstory=(
            "You are a Principal Mining and Environmental Engineer with 32 years "
            "of experience permitting and designing mines in the two most complex "
            "regulatory jurisdictions relevant to this pipeline: British Columbia "
            "and Ontario in Canada, and Western Australia and South Australia in "
            "Australia. You have led ESIA processes under Canada's Impact Assessment "
            "Act 2019 (IAA), BC's Environmental Assessment Act 2018, the EPBC Act "
            "1999 (Australia), and multiple state-level mining acts. You have "
            "navigated BC's Declaration on the Rights of Indigenous Peoples Act "
            "(DRIPA 2019) and its alignment with UNDRIP, the Crown's Duty to "
            "Consult under Section 35 of the Constitution Act 1982, Australia's "
            "Native Title Act 1993, and state Aboriginal Heritage Acts. You have "
            "personally run FPIC processes under ILO Convention 169 for projects "
            "where consent was genuinely uncertain, and you have recommended "
            "project cancellation on social licence grounds on two occasions.\n\n"
            "Your technical expertise spans tailings storage facility design and "
            "closure under the Global Industry Standard on Tailings Management "
            "(GISTM 2020), ARD/AMD prediction using acid-base accounting and "
            "kinetic testing protocols, water balance modelling in alpine and arid "
            "environments, and biodiversity offset design under the Mitigation "
            "Hierarchy (BBOP/IFC PS6). You evaluate every project against: IFC "
            "Performance Standards PS1-PS8; ICMM 10 Principles; IRMA Standard "
            "for Responsible Mining; Equator Principles EP4; SASB Metals & Mining "
            "guidelines; and emerging TNFD (Taskforce on Nature-related Financial "
            "Disclosures) and TCFD (Task Force on Climate-related Financial "
            "Disclosures) reporting obligations that institutional lenders now "
            "routinely require.\n\n"
            "Your analytical framework is built around one discipline that most "
            "environmental consultants ignore: the distinction between a risk that "
            "is DIFFICULT and a risk that is UNRESOLVABLE. A difficult risk has a "
            "mitigation pathway, a cost, and a permitting route, even if all three "
            "are onerous. An unresolvable risk has none — or has a cost so "
            "disproportionate that it destroys the project economics regardless of "
            "commodity price. You never describe a risk as severe or significant "
            "without also stating which of those two categories it falls into, "
            "because that distinction is the only thing the COO actually needs "
            "from your report. A report that says 'high avalanche risk' without "
            "saying whether that is manageable or fatal is not a report — it is "
            "an observation, and observations do not justify or deny capital "
            "allocation decisions."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llms["env_engineer"],
    )

    economist = Agent(
        role="Principal Mining Economist and Project Finance Specialist",
        goal=(
            "Produce a Scoping-level financial pre-assessment of institutional quality — "
            "one that a senior lender, a royalty streamer, or an Investment Committee "
            "chair would accept as the definitive first-pass economic screen for capital "
            "allocation. The report must: (1) anchor every cost estimate in named real-"
            "world analogues; (2) run three explicit scenarios (Bear, Base, Bull) so the "
            "COO can see the range of outcomes, not just the midpoint; (3) state the "
            "minimum mineable tonnage required for a positive NPV so the COO can cross-"
            "check it against the geologist's evidence; (4) quantify ESG risks as dollar "
            "costs and ESG initiatives as dollar-value opportunities — both are material "
            "to the investment case; and (5) propose a realistic project financing "
            "structure, because a project that cannot be financed is not a project."
        ),
        backstory=(
            "You are a Principal Mining Economist and Project Finance Specialist with "
            "36 years of experience spanning commodity cycle research at a bulge-bracket "
            "investment bank, twelve years as Head of Project Finance at a Big 4 "
            "accounting firm advising mining clients on SEDAR and ASX filings, and the "
            "last decade as an independent technical-economic consultant whose Scoping "
            "Studies and PEAs have been accepted by the TSX-V, ASX, and the London Stock "
            "Exchange AIM board. You have signed off on economic assessments covering "
            "more than USD 80 billion in prospective project capital across copper-gold "
            "porphyry, nickel sulphide, gold epithermal, lithium brine, and bulk "
            "commodities in 22 countries.\n\n"
            "Your methodology is built around the Lassonde Curve: you locate every "
            "prospect on the discovery-to-production continuum before you model a single "
            "dollar of CAPEX, because stage position determines the value gap the "
            "capital is being deployed to close. An Early Discovery project with one "
            "drill hole is worth its optionality — the question is whether a PEA drill "
            "program priced at the current Scoping level will generate sufficient "
            "geological resolution to support a resource classification that unlocks "
            "institutional equity and streaming capital.\n\n"
            "You build every DCF in three scenarios — Bear, Base, Bull — using the "
            "consensus long-run commodity price from the four largest bank research "
            "desks as your Base case, the P10 trough of the last 20-year price cycle "
            "as Bear, and the P90 price peak as Bull. You never present a single-"
            "point NPV to an Investment Committee because a single point conceals the "
            "distribution that determines whether the project is genuinely robust or "
            "merely optimistic.\n\n"
            "You treat ESG not as a compliance cost but as a financing variable: "
            "projects with IRMA certification attract a 5-15% offtake price premium "
            "from technology sector buyers; ESG-linked revolving credit facilities "
            "currently price 25-75 basis points below standard facilities for mining "
            "companies with validated TCFD and TNFD disclosures; green bonds and "
            "sustainability-linked bonds have funded over USD 12 billion of mine "
            "development since 2021. Carbon credit revenues from forest conservation "
            "offset programmes and methane capture projects have converted marginal "
            "projects into fundable ones. You model these as upside items in the Bull "
            "scenario with probability-weighted contributions to the Base NPV.\n\n"
            "Your discipline around data quality is constitutional: a Scoping-level "
            "estimate carries ±40% accuracy by definition, and you state that "
            "explicitly on every CAPEX and OPEX figure. You never present a narrow "
            "engineering-grade range ($620M-$680M) from a single drill hole because "
            "that false precision misleads boards, attracts securities regulators, and "
            "ends careers. The only number you defend from one hole is an order-of-"
            "magnitude range."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llms["economist"],
    )

    coo = Agent(
        role="Chief Operating Officer",
        goal=(
            "Issue the single decision the physical and financial evidence supports — "
            "PROCEED to a named study stage, or REJECT. Before considering the upside, "
            "verify that the three upstream assessments are physically consistent: "
            "confirm the economist's mining method matches the intersection width the "
            "geologist reported, confirm the NPV is built on a demonstrated ore volume "
            "not an assumed one, and confirm that the proposed capital commitment is "
            "proportionate to the current data confidence level. A positive IRR that "
            "rests on an undrilled tonnage assumption is not a financial result — it "
            "is a number, and you will not authorise capital on that basis. Every "
            "PROCEED decision must carry specific, measurable Binding Conditions and "
            "bright-line Stage-Reversal Triggers scaled to the actual intersection "
            "width and resource confidence. Every REJECT must state exactly what "
            "evidence would need to change to reopen the file."
        ),
        backstory=(
            "You are the Chief Operating Officer of a mid-tier mining company listed "
            "on the TSX and ASX, reporting to a board of institutional investors who "
            "measure every gate decision against risk-adjusted return on capital. Over "
            "a 35-year executive career you have served as COO or CEO at five listed "
            "mining companies across four continents, and you have personally chaired "
            "Investment Committees that sanctioned — or cancelled — projects totalling "
            "more than $25 billion in prospective capital expenditure. You have lived "
            "through two full commodity cycles, two sovereign expropriation events, "
            "one catastrophic tailings failure that ended careers, and three projects "
            "where social licence collapse cost the company more than the ore body "
            "was worth. Those experiences have made you constitutionally sceptical of "
            "exploration optimism and permanently allergic to decisions that are "
            "driven by sunk-cost logic rather than forward-looking evidence.\n\n"
            "Your analytical framework begins not with what is promising, but with "
            "what ends the conversation. The first thing you do when you receive "
            "three upstream assessments is check whether they are physically "
            "consistent with each other — because you have seen too many projects "
            "where the economist's mine plan assumed an ore body the geologist never "
            "actually drilled. A positive IRR built on undrilled tonnes is not a "
            "financial result — it is a fantasy dressed as arithmetic. You will not "
            "be fooled by it, and you will say so in writing.\n\n"
            "You know precisely what each drill intersection width implies about "
            "mining method and capital proportionality. A 2-metre drill intersection "
            "in a remote desert cannot support a $100 million open-pit mine — the "
            "ore column is physically too narrow for bulk mining selectivity, and no "
            "amount of optimistic grade assumptions changes that geometry. You have "
            "killed projects on this basis before and you will do it again, because "
            "a REJECT that preserves $30 million of exploration capital is exactly "
            "as valuable as a PROCEED that generates returns — possibly more so.\n\n"
            "You do not accept the economist's NPV at face value. You ask: what "
            "tonnage assumption is embedded in that NPV, and is there a drill hole "
            "that demonstrates that tonnage exists? If the answer is no, the NPV is "
            "a spreadsheet exercise, not a resource estimate, and you will not "
            "authorise capital on that basis. You require every Binding Condition "
            "to be specific, measurable, and testable — not aspirational — because "
            "vague conditions protect no one and commit no one. You write for "
            "institutional investors who will hold you personally accountable for "
            "every dollar that leaves the company on the basis of your signature."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llms["coo"],
    )

    return {
        "geologist": geologist,
        "env_engineer": env_engineer,
        "economist": economist,
        "coo": coo,
    }


# ---------------------------------------------------------------------------
# Guardrails — self-correcting validator loop
#
# Each guardrail is a callable that CrewAI invokes after the task's agent
# produces an output. It returns (True, payload) to accept or
# (False, error_message) to send the task back to the agent with the error
# message appended. CrewAI handles the retry loop up to `guardrail_max_retries`.
#
# This is the agentic differentiator vs. plain RAG: agents are forced to
# self-correct against an explicit checklist before their output is accepted.
# ---------------------------------------------------------------------------
_META_PHRASES = (
    r"note that\s+i\b",
    r"please note\b",
    # Catches "Note: I…", "Note: This…", "Note: Since…" — every Note:-prefixed
    # commentary line observed leaking from llama3 + qwen2.5 in portfolio runs.
    r"^\s*note\s*:",
    r"i have omitted",
    r"i have not added",
    r"i have followed",
    r"directly given the output",
    r"this response complies",
    # NOTE: "as requested", "as instructed", "in the requested format", and
    # "per your requirement" were removed because they match legitimate
    # mining-industry language ("as requested by JORC 2012", "as instructed
    # by IFC PS6", etc.) and cause false-positive guardrail rejections.
)
# MULTILINE so the ^ anchor in the "Note:" pattern matches at the start of any
# line, not just the start of the whole output.
_META_RE = re.compile("|".join(_META_PHRASES), re.IGNORECASE | re.MULTILINE)


_MD_BOLD_RE   = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC_RE = re.compile(r"\*(.+?)\*",   re.DOTALL)
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+",  re.MULTILINE)


def _strip_inline_markdown(text: str) -> str:
    """Remove bold (**), italic (*), and ATX-header (#) markers.

    llama3 is inconsistent about whether it emits markdown decoration in its
    prose. Stripping here means every section assembled into the final document
    is plain prose, giving uniform font weight across all agents and all runs.
    """
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    return text


def _strip_meta_commentary(text: str) -> str:
    """Remove meta-commentary phrases while preserving non-meta content on the same line.

    The original line-level approach (drop any line containing a meta-phrase) was
    safe for Llama 3, which placed its 'Note:' disclaimers on isolated trailing
    lines. Mixtral 8x22B uses 'Note:' as an inline structural prefix (e.g.
    'Note: Tailings & ARD/AMD Risk:'), so a whole-line drop removes the section
    header along with the prefix, causing the guardrail to report headers as missing
    even though they appeared in the raw output.

    This implementation substitutes out only the matched phrase (leaving any
    surrounding real content intact) and then discards lines that became entirely
    blank after substitution.
    """
    cleaned = _META_RE.sub("", text)
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    return "\n".join(lines)


def _output_text(output) -> str:
    if isinstance(output, TaskOutput):
        return output.raw or ""
    if hasattr(output, "raw"):
        return getattr(output, "raw") or ""
    return str(output)



def _check_required_sections(text: str, headers: List[str]) -> Optional[str]:
    """Verify every required header is present at the start of a line.

    A header is considered present when its full stem (the text before the
    trailing colon) appears at the start of a line, optionally followed by a
    short qualifier (em-dash subtitle, parenthetical, etc.) and then a colon.
    This accepts both the canonical form and the qualified forms the prompts
    actually ask the agents to emit, e.g. the economy template instructs the
    agent to write 'DCF Valuation — Three Scenarios:' while the guardrail
    list registers 'DCF Valuation:'. The previous strict-substring check
    rejected that on every run, looping the agent until guardrail failure.

    Anchoring with ^ (MULTILINE) plus a ``\\b`` word boundary prevents prose
    mentions like '...the CAPEX Estimate is going to be high:' from being
    accepted as the section header.
    """
    missing: List[str] = []
    for header in headers:
        stem = header.rstrip(":").strip()
        # Allow up to 80 chars of qualifier text (em-dash subtitle, parenthetical
        # scope note, etc.) between the stem and the closing colon. Stops at
        # the first newline or the first colon, whichever comes first, so we
        # never absorb across paragraphs.
        pattern = re.compile(
            r"^\s*" + re.escape(stem) + r"\b[^\n:]{0,80}:",
            re.IGNORECASE | re.MULTILINE,
        )
        if not pattern.search(text):
            missing.append(header)
    if missing:
        return (
            "Output is missing required section header(s): "
            + ", ".join(repr(m) for m in missing)
            + ". Re-emit the response with every required section present, in order."
        )
    return None


# Canonical geology section headers (without trailing colon). Used both for
# the missing-headers check and for slicing individual sections out of the
# agent's output. Keeping the list in one place prevents drift between the
# two uses, which was the root of the original guardrail false-positive.
_GEOLOGY_HEADERS: Tuple[str, ...] = (
    "Mineralisation Style & Geological Setting",
    "Deposit Analogue Comparison",
    "Resource Confidence Classification",
    "Ore Body Geometry Estimate",
    "Grade & Tonnage Indication",
    "Geotechnical Pre-Screen",
    "ARD & Metallurgical Pre-Screen",
    "TCFD Physical Risk Flag",
    "Key Subsurface Risks",
    "Drill Program Forecast",
)

# Boundary used when slicing out one geology section. Matches a newline
# followed by any of the canonical headers and its trailing colon. This is
# strict on purpose: the previous "any uppercase line ending in colon"
# pattern was triggered by ordinary prose like "...jurisdictions are: Oyu
# Tolgoi..." and silently truncated sections to zero length, causing valid
# output to fail the analogue / geometry / resource checks.
_NEXT_GEO_HEADER_RE = (
    r"(?=\n\s*(?:" + "|".join(re.escape(h) for h in _GEOLOGY_HEADERS) + r")\s*:|\Z)"
)


def _geology_section_re(header: str) -> "re.Pattern[str]":
    """Compile a regex that captures the body of one geology section."""
    return re.compile(
        re.escape(header) + r"\s*:\s*(.*?)" + _NEXT_GEO_HEADER_RE,
        re.DOTALL | re.IGNORECASE,
    )


_RESOURCE_SECTION_RE = _geology_section_re("Resource Confidence Classification")
_ANALOGUE_SECTION_RE = _geology_section_re("Deposit Analogue Comparison")
_GEOMETRY_SECTION_RE = _geology_section_re("Ore Body Geometry Estimate")


def _geology_guardrail(output) -> Tuple[bool, str]:
    # NOTE: strip markdown BEFORE anything else. The agents (mixtral / llama3)
    # routinely wrap headers in **bold**, and without this step the bold
    # markers leak into the section-slicing regexes and cause every body
    # capture to collapse to "**", producing spurious guardrail rejections.
    # _env_guardrail already does this; _geology_guardrail did not, which was
    # the root cause of "Deposit Analogue Comparison must name a real mine"
    # firing on outputs that clearly named real mines.
    text       = _strip_meta_commentary(_strip_inline_markdown(_output_text(output)))
    normalized = " ".join(text.lower().split())
    issues: List[str] = []

    # ── 1. All required sections present ─────────────────────────────────
    headers = [f"{h}:" for h in _GEOLOGY_HEADERS]
    missing = [
        h for h in headers
        if h.lower().rstrip(":") not in normalized
    ]
    if missing:
        issues.append(
            "Output is missing required section(s): "
            + ", ".join(repr(h) for h in missing)
            + ". Re-emit the full report with every required section present "
            "in the correct order."
        )

    # ── 2. JORC discipline: Resource Confidence Classification section ────
    # Scope the check to that section only to avoid false positives from
    # explanatory prose elsewhere in the report.
    section_match = _RESOURCE_SECTION_RE.search(text)
    if section_match:
        section = section_match.group(1)
        has_inferred   = bool(re.search(r"\binferred\b",          section, re.IGNORECASE))
        has_exp_target = bool(re.search(r"\bexploration\s+target\b", section, re.IGNORECASE))
        if not (has_inferred or has_exp_target):
            issues.append(
                "JORC violation: the Resource Confidence Classification section "
                "must explicitly assign either an Inferred Mineral Resource or a "
                "JORC clause 18 Exploration Target. A single discovery drill "
                "intersection cannot support Indicated or Measured. Re-emit the "
                "section stating one of those two permitted categories."
            )

    # ── 3. Deposit Analogue Comparison must contain at least one mine name ─
    analogue_match = _ANALOGUE_SECTION_RE.search(text)
    if analogue_match:
        analogue_section = analogue_match.group(1)
        # A real analogue must contain at least one word that looks like a mine
        # or project name — i.e., a capitalised proper noun that is not a
        # common geological term. Proxy: at least one run of 2+ capitalised
        # words not on a header line.
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", analogue_section)
        if not proper_nouns:
            issues.append(
                "The 'Deposit Analogue Comparison' section must name at least one "
                "real producing or advanced-stage mine (proper noun). Generic "
                "descriptions without a named analogue are not acceptable."
            )

    # ── 4. Ore Body Geometry Estimate must contain a dimension ───────────
    geometry_match = _GEOMETRY_SECTION_RE.search(text)
    if geometry_match:
        geometry_section = geometry_match.group(1)
        has_dimension = bool(re.search(r"\d+\s*m\b", geometry_section, re.IGNORECASE))
        if not has_dimension:
            issues.append(
                "The 'Ore Body Geometry Estimate' section must state at least one "
                "specific dimension in metres (e.g. '10m to 50m true width'). "
                "A purely qualitative description is not acceptable."
            )

    # ── 5. No unfilled placeholders anywhere in the output ───────────────
    # The model sometimes echoes the expected_output template literally,
    # producing tokens like "[X Mt]", "[A%]", or "TBD" instead of real numbers.
    # Any such token is a report-quality failure.
    placeholder_re = re.compile(
        r"\[X\b|\[A%|\[A\s+to|\bTBD\b|\bto be determined\b",
        re.IGNORECASE,
    )
    if placeholder_re.search(text):
        issues.append(
            "Output contains unfilled placeholder tokens (e.g. '[X Mt]', '[A%]', "
            "'TBD'). Replace every placeholder with an actual numerical estimate "
            "or an explicit statement of why no estimate is possible."
        )

    if issues:
        return False, "Revise the output. Issues detected:\n- " + "\n- ".join(issues)
    return True, text


def _env_header_present(normalized: str, header: str) -> bool:
    """Tolerant header-presence check for the env-engineer guardrail.

    ``normalized`` is the env output with all whitespace collapsed and
    lowercased. The matcher accepts the following equivalences observed in
    Mixtral / Mistral-Small outputs, all of which are semantically the same
    header but break a strict substring search:

      - ``&`` may be rendered as either ``&`` or ``and``.
      - ``/`` may carry arbitrary whitespace on either side
        (``ARD/AMD`` vs ``ARD / AMD``).
      - The trailing colon is optional.

    Without these relaxations the guardrail rejects valid output for
    ``Tailings & ARD/AMD Risk`` and ``TNFD & TCFD Compliance Flags`` — the
    two headers in the env template that contain both ``&`` and ``/`` or
    are short enough that the model paraphrases them on retry.
    """
    stem = header.lower().rstrip(":").strip()
    tokens = re.split(r"(\s+|/|&)", stem)
    parts: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        if tok.isspace():
            parts.append(r"\s+")
        elif tok == "&":
            parts.append(r"(?:&|and)")
        elif tok == "/":
            parts.append(r"\s*/\s*")
        else:
            parts.append(re.escape(tok))
    return bool(re.search("".join(parts) + r"\s*:?", normalized))


def _env_guardrail(output) -> Tuple[bool, str]:
    text = _strip_meta_commentary(_strip_inline_markdown(_output_text(output)))

    # Collapse all whitespace into single spaces so that headers split across
    # multiple lines (e.g. box-art / Unicode border formatting) are still
    # detected. Trailing colons are stripped from the search key because
    # models occasionally omit them while still producing section content.
    normalized = " ".join(text.lower().split())

    issues: List[str] = []

    # ── 1. All eight required sections must be present ────────────────────────
    headers = [
        "Jurisdiction & Regulatory Framework:",
        "Ecological Sensitivities:",
        "Tailings & ARD/AMD Risk:",
        "Permitting Pathway:",
        "Social Licence & Indigenous Rights:",
        "ESG Red Flags & Resolvability Assessment:",
        "TNFD & TCFD Compliance Flags:",
        "Recommended Mitigations:",
    ]
    missing = [h for h in headers if not _env_header_present(normalized, h)]
    if missing:
        issues.append(
            "Output is missing required section header(s): "
            + ", ".join(repr(m) for m in missing)
            + ". Re-emit the response with every required section present, in "
            "order. Use the headers EXACTLY as written above — keep the "
            "ampersand '&' (do NOT substitute 'and') and write 'ARD/AMD' and "
            "'TNFD & TCFD' with no spaces around the slash or ampersand."
        )

    # ── 2. ESG Red Flags section must contain explicit resolvability verdicts ─
    # At minimum one of the three verdict keywords must appear.
    verdict_re = re.compile(
        r"\b(RESOLVABLE|BORDERLINE|POTENTIALLY FATAL)\b", re.IGNORECASE
    )
    if "esg red flags" in normalized and not verdict_re.search(text):
        issues.append(
            "The 'ESG Red Flags & Resolvability Assessment' section must assign "
            "an explicit verdict of RESOLVABLE, BORDERLINE, or POTENTIALLY FATAL "
            "to each risk. No such verdicts were found."
        )

    # ── 3. Section must close with one of the two mandatory summary lines ─────
    has_fatal_line = bool(
        re.search(r"FATAL FLAW DETECTED", text, re.IGNORECASE)
    )
    has_clear_line = bool(
        re.search(r"NO FATAL ENVIRONMENTAL OR SOCIAL FLAW DETECTED", text, re.IGNORECASE)
    )
    if "esg red flags" in normalized and not (has_fatal_line or has_clear_line):
        issues.append(
            "The 'ESG Red Flags & Resolvability Assessment' section must end with "
            "either 'FATAL FLAW DETECTED — [risk name] — [reason]' or "
            "'NO FATAL ENVIRONMENTAL OR SOCIAL FLAW DETECTED AT THIS STAGE'."
        )

    if issues:
        return False, "Revise the output. Issues detected:\n- " + "\n- ".join(issues)
    return True, text


def _economy_guardrail(output) -> Tuple[bool, str]:
    # Normalise markdown decoration first so that headers wrapped in bold
    # (**CAPEX Estimate:**) are cleaned before the section-header check.
    # Strip meta-commentary after markdown so 'Note:' lines can't hide inside
    # bold/header decoration.
    text = _strip_meta_commentary(_strip_inline_markdown(_output_text(output)))
    issues: List[str] = []

    # ── 1. All eight required sections must be present ────────────────────────
    headers = [
        "Deposit Positioning & Comparables:",
        "Resource Scenario Framework:",
        "CAPEX Estimate:",
        "OPEX & AISC Estimate:",
        "Commodity Price Analysis:",
        "DCF Valuation:",
        "ESG Initiatives & Value Enhancement:",
        "Project Financing & Key Risks:",
    ]
    if (msg := _check_required_sections(text, headers)):
        issues.append(msg)

    # ── 2. LaTeX / equation syntax must be absent ─────────────────────────────
    if re.search(r"\\\(|\\\[|\$\$|\\frac|\\sum", text):
        issues.append(
            "Output contains LaTeX or equation syntax. Strip all math notation "
            "and re-emit in plain business prose."
        )

    # ── 3. Scoping-level uncertainty label must be present ────────────────────
    if "±40%" not in text and "+/-40%" not in text.lower():
        issues.append(
            "Output must include the '±40%' Scoping-level uncertainty label on "
            "CAPEX and OPEX figures. Add it where missing."
        )

    # ── 4. Three DCF scenarios must be present ────────────────────────────────
    normalized = " ".join(text.lower().split())
    if "bear" not in normalized or "bull" not in normalized:
        issues.append(
            "DCF Valuation section must contain Bear, Base, and Bull scenario "
            "analysis. One or more scenarios are missing."
        )

    if issues:
        return False, "Revise the output. Issues detected:\n- " + "\n- ".join(issues)
    return True, text


_COO_RECOMMENDATION_RE = re.compile(
    r"Recommendation\s*:\s*(PROCEED\s+to\s+(PEA|PFS|DFS)\b|REJECT\b)",
    re.IGNORECASE,
)
_COO_PEA_PFS_CONFLATION_RE = re.compile(
    r"PEA\s*\(\s*(pre[-\s]?feasibility|PFS)",
    re.IGNORECASE,
)


def _coo_guardrail(output) -> Tuple[bool, str]:
    text       = _strip_meta_commentary(_output_text(output))
    normalized = " ".join(text.lower().split())   # whitespace-collapsed for section checks
    issues: List[str] = []

    # ── 1. Recommendation line format ────────────────────────────────────
    if not _COO_RECOMMENDATION_RE.search(text):
        issues.append(
            "Output must contain a line of the exact form "
            "'Recommendation: PROCEED to <PEA|PFS|DFS>' or 'Recommendation: "
            "REJECT'. No parenthetical expansions, no alternative names."
        )

    # ── 2. PEA / PFS conflation ───────────────────────────────────────────
    if _COO_PEA_PFS_CONFLATION_RE.search(text):
        issues.append(
            "PEA/PFS conflation detected. PEA = Preliminary Economic Assessment. "
            "PFS = Pre-Feasibility Study. They are DIFFERENT, SEQUENTIAL stages. "
            "Never write 'PEA (Pre-Feasibility Study)'. Use exactly one stage "
            "abbreviation with no parenthetical."
        )

    # ── 3. Required analytical sections ──────────────────────────────────
    required_sections = [
        "Capital Risk Assessment:",
        "Fatal Flaw Assessment:",
        "Binding Conditions:",
        "Key Rationale:",
    ]
    missing_sections = [
        h for h in required_sections
        if h.lower().rstrip(":") not in normalized
    ]
    if missing_sections:
        issues.append(
            "Output is missing required section(s): "
            + ", ".join(repr(s) for s in missing_sections)
            + ". Re-emit the full report with every required section present."
        )

    # ── 4. Key Rationale must have ≥ 3 numbered points ───────────────────
    # Scope the check to Key Rationale only — everything after that header
    # to end-of-string. Since Key Rationale is the last section, this
    # avoids counting numbered items in preceding analytical sections.
    rationale_match = re.search(
        r"Key\s+Rationale\s*:\s*(.*?)(?=\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    rationale_section = rationale_match.group(1) if rationale_match else ""
    points = re.findall(r"(?m)^\s*\d+\.\s+\S", rationale_section)
    if len(points) < 3:
        issues.append(
            "The 'Key Rationale:' section must contain exactly three numbered "
            "points (1., 2., 3.). "
            f"Found {len(points)}. Re-emit with all three points present."
        )

    if issues:
        return False, "Revise the output. Issues detected:\n- " + "\n- ".join(issues)
    return True, text


# ---------------------------------------------------------------------------
# COO hard-rule contradiction guardrail
# ---------------------------------------------------------------------------
# The COO prompt's REJECT TEST is explicit: for BULK DEPOSITS, fatal flaw
# (a) "Capital-Geology Mismatch" DOES NOT TRIGGER when the reported drilled
# intersection is ≥ 30 m. The prompt even uses "114 m porphyry" as the
# canonical non-trigger example. Despite this, llama-3.3-70B has been
# observed marking (a) as "Triggered" on a 114 m porphyry — a vibes-driven
# REJECT that contradicts both the prompt and the upstream geologist /
# economist outputs. The guardrail below catches that exact contradiction
# and forces a retry.
#
# This check runs IN ADDITION to the existing format/section checks in
# _coo_guardrail; it does not replace them.
_INTERSECTION_M_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*m\b", re.IGNORECASE)

# Patterns on the dataset's Geological_Setting field that map to bulk
# deposits in the COO prompt's terminology (porphyry, mafic / ultramafic
# intrusion, sediment-hosted Cu). Each pattern is anchored at a token
# boundary (start-of-string or underscore) so "sedimentary" does NOT match
# "metasedimentary" — the latter is a hydrothermal-vein host, not a bulk
# sediment-hosted system. Anything not matched here is treated as ambiguous
# and the contradiction check does not fire — better to under-enforce than
# to wrongly trip on a legitimate narrow-vein REJECT.
_BULK_DEPOSIT_RE = re.compile(
    r"(?:^|[_\s])(?:porphyry|sill|mafic_intrusion|sedimentary)(?:$|[_\s])",
    re.IGNORECASE,
)

# A line that names fatal flaw (a) Capital-Geology Mismatch. The flaw label
# can be written as "a)", "(a)", or "Fatal Flaw a:", and the connector
# between the label and the status word can be "-", ":", or an em-dash.
# We grab the whole line and inspect it for negation in a second pass so
# phrasings like "Not Triggered" / "Does NOT Trigger" don't false-positive.
_CAPITAL_GEOLOGY_LINE_RE = re.compile(
    r"^[^\n]*Capital[\s\-]?Geology[\s\-]?Mismatch[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEGATED_TRIGGER_RE = re.compile(
    r"\b(?:not|no|n[\u2019']t)\b[^\n]{0,20}\btrigger(?:ed|s)?\b",
    re.IGNORECASE,
)
_TRIGGERED_RE = re.compile(r"\btrigger(?:ed)?\b", re.IGNORECASE)


def _parse_intersection_metres(intersection_str: object) -> Optional[float]:
    """Extract the leading metre value from a Best_Drill_Intersection cell.

    The dataset stores intersections as e.g. ``"114m_at_0.75_Percent_CuEq"``.
    Underscores are converted to spaces first so the regex sees "114m at …".
    Returns ``None`` when no leading metre value is parseable; callers must
    treat ``None`` as "unknown — do not enforce the contradiction check".
    """
    if intersection_str is None:
        return None
    text = str(intersection_str).replace("_", " ")
    m = _INTERSECTION_M_RE.match(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def _is_bulk_deposit(geological_setting: object) -> bool:
    """True when the geological setting maps to a 'bulk' deposit class.

    Bulk = porphyry, mafic intrusion, sediment-hosted Cu, ultramafic sill.
    Narrow-vein and ambiguous settings (including "metasedimentary", which
    is hydrothermal-vein-hosted) return False so the guardrail's
    contradiction check does not fire for them.
    """
    if geological_setting is None:
        return False
    return bool(_BULK_DEPOSIT_RE.search(str(geological_setting)))


def _coo_marks_capital_geology_triggered(text: str) -> bool:
    """Return True iff the COO output marks fatal flaw (a) as Triggered.

    Looks at every line that names "Capital-Geology Mismatch" and checks
    whether it contains the word "Triggered" without a preceding negation.
    A line like ``"a) Capital-Geology Mismatch: Not Triggered"`` is
    recognised as NOT triggered and skipped.
    """
    for match in _CAPITAL_GEOLOGY_LINE_RE.finditer(text):
        line = match.group(0)
        if _NEGATED_TRIGGER_RE.search(line):
            continue
        if _TRIGGERED_RE.search(line):
            return True
    return False


def _make_coo_guardrail(
    intersection_m: Optional[float],
    is_bulk: bool,
) -> Callable[[object], Tuple[bool, str]]:
    """Build a COO guardrail that knows the prospect's intersection width
    and deposit class so it can catch a specific class of self-contradictory
    output.

    The returned callable runs all the existing ``_coo_guardrail`` checks
    (recommendation format, PEA/PFS conflation, required sections, key
    rationale point count) and ADDS one extra check: if the deposit is bulk
    and the intersection is ≥ 30 m, fatal flaw (a) Capital-Geology Mismatch
    must NOT be marked Triggered. If it is, we return a remediation message
    quoting the prompt's own rule so the agent's retry has a clear target.
    """
    def _guard(output: object) -> Tuple[bool, str]:
        ok, msg = _coo_guardrail(output)
        text = _strip_meta_commentary(_strip_inline_markdown(_output_text(output)))

        contradiction: Optional[str] = None
        if (
            is_bulk
            and intersection_m is not None
            and intersection_m >= 30.0
            and _coo_marks_capital_geology_triggered(text)
        ):
            contradiction = (
                "Fatal Flaw (a) 'Capital-Geology Mismatch' is marked Triggered, "
                f"but the prospect's reported intersection is {intersection_m:g} m "
                "in a bulk deposit. The COO prompt's REJECT TEST is explicit: "
                "for BULK DEPOSITS, if intersection >= 30 m then condition (a) "
                "DOES NOT TRIGGER (the prompt uses '114 m porphyry' as the "
                "canonical non-trigger example). Re-emit the Fatal Flaw "
                "Assessment with (a) marked 'Not Triggered' / 'Clear', "
                "re-evaluate the recommendation against the unchanged "
                "geological and economist evidence, and produce a stage-gate "
                "decision that follows the prompt's hard rules instead of "
                "overriding them."
            )

        if contradiction is None:
            return ok, msg
        if not ok:
            return False, msg + "\n- " + contradiction
        return False, "Revise the output. Issues detected:\n- " + contradiction

    return _guard


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def _make_task_callback(
    role: str,
    progress_callback: Optional[Callable[[str], None]],
) -> Optional[Callable[[object], None]]:
    """Build a per-task CrewAI callback that notifies the UI when the task
    completes.

    CrewAI calls ``Task.callback(task_output)`` synchronously after the
    task succeeds. We wrap the user's role-agnostic ``progress_callback``
    in a closure that pre-fills the role string so the UI knows which
    agent just finished. ``role`` must be passed as a default argument,
    not captured by free reference, to defeat Python's late-binding loop
    pitfall (without it, every callback would report the role of the
    last-built task).
    """
    if progress_callback is None:
        return None

    def _cb(_task_output: object, _role: str = role) -> None:
        try:
            progress_callback(_role)
        except Exception:  # progress hooks must never break the run
            log.exception("progress_callback raised for role=%s", _role)

    return _cb


def _build_tasks(
    agents: Dict[str, Agent],
    prospect_name: str,
    prospect_brief: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Task]:
    # Per-prospect inputs for the COO guardrail's hard-rule contradiction
    # check. See _make_coo_guardrail for the rule being enforced.
    _prospect_row = _get_prospect_row(prospect_name)
    _intersection_m = _parse_intersection_metres(
        _prospect_row.get("Best_Drill_Intersection")
    )
    _is_bulk = _is_bulk_deposit(_prospect_row.get("Geological_Setting"))
    coo_guardrail = _make_coo_guardrail(_intersection_m, _is_bulk)

    # Per-task completion callbacks. Used by the Streamlit UI to update a
    # live "agent done" checklist; harmless when the UI does not pass a
    # progress_callback (every entry is None and CrewAI ignores it).
    _cb_geology = _make_task_callback("geology",  progress_callback)
    _cb_env     = _make_task_callback("env",      progress_callback)
    _cb_economy = _make_task_callback("economy",  progress_callback)
    _cb_coo     = _make_task_callback("coo",      progress_callback)

    geology_task = Task(
        description=(
            f"{prospect_brief}\n\n"
            "Using the prospect briefing above, deliver a JORC 2012-compliant, "
            "CRIRSCO-aligned geological pre-assessment. Work through the following "
            "analytical steps in order — every step is mandatory:\n\n"

            "STEP 1 — DEPOSIT TYPE IDENTIFICATION AND ANALOGUE COMPARISON.\n"
            "Classify the mineralisation style and deposit type with precision. "
            "Then name one or two real, producing (or advanced-stage) mines of the "
            "same deposit type in the same or a geologically analogous jurisdiction "
            "(Canada or Australia). State the analogue mine's reported grade, typical "
            "intersection widths, and JORC/NI 43-101 resource size. Then compare this "
            "prospect to those analogues: is the reported grade above, at, or below "
            "the analogue median? Is the intersection width consistent with the "
            "analogue ore body geometry? This comparison gives the downstream economist "
            "a real benchmark for CAPEX scaling.\n\n"

            "STEP 2 — RESOURCE CONFIDENCE CLASSIFICATION (JORC / CIM DISCIPLINE).\n"
            "Assign a resource confidence category strictly under JORC 2012 and CIM "
            "Definition Standards 2014 (both apply — Australia uses JORC, Canada uses "
            "NI 43-101/CIM). Apply the following rules without exception:\n"
            "  • A single discovery drill intersection supports AT MOST a JORC clause "
            "17 Inferred Mineral Resource — and only if geological continuity can be "
            "reasonably assumed from the setting and deposit type.\n"
            "  • If continuity cannot be assumed from a single hole, report as a JORC "
            "clause 18 Exploration Target expressed as a grade/tonnage RANGE with the "
            "explicit disclaimer: 'This is an Exploration Target. It is conceptual in "
            "nature, there has been insufficient exploration to define a Mineral "
            "Resource, and it is uncertain if further exploration will result in the "
            "determination of a Mineral Resource.'\n"
            "  • Indicated and Measured require infill drilling at defined spacings. "
            "Do not claim either off a single hole under any circumstances.\n"
            "  • Flag every data gap that prevents a higher classification.\n"
            "  • This assessment is an internal pre-CP opinion. Note that public "
            "disclosure requires sign-off by a Competent Person (JORC) or Qualified "
            "Person (NI 43-101) who accepts responsibility for the estimate.\n\n"

            "STEP 3 — ORE BODY GEOMETRY ESTIMATE.\n"
            "Based on the deposit type, geological setting, and reported intersection "
            "width, state explicitly:\n"
            "  • The minimum and maximum plausible true width of the ore zone — "
            "distinguishing between 'this intersection is likely the full width' "
            "(narrow-vein systems) and 'this intersection is likely a marginal edge "
            "of a much wider system' (porphyry, mafic intrusion).\n"
            "  • The implied strike length range and dip if inferrable from the setting.\n"
            "  • Which mining method — open-pit, bulk underground, or narrow-vein "
            "underground — is physically supportable given the geometry range. State "
            "the minimum width threshold for each method (open-pit requires >30m "
            "continuous width for economic selectivity; bulk underground requires "
            ">5m; narrow-vein underground can work at 2–5m but at high unit cost).\n"
            "This geometry statement is the most critical output for the COO, who will "
            "use it to validate whether the economist's mining method is physically "
            "possible.\n\n"

            "STEP 4 — GEOTECHNICAL PRE-SCREEN.\n"
            "Based on the deposit type and host rock geology, provide a qualitative "
            "rock mass assessment:\n"
            "  • Expected rock competency (competent, moderate, poor) and the "
            "geological basis for that assessment.\n"
            "  • Key structural risks — fault zones, shear zones, or foliation "
            "orientations that could affect ground support requirements or slope "
            "stability in an open pit.\n"
            "  • Depth-to-target implications for underground access (shallow <150m "
            "favours decline access; deep >300m may require shaft, with significant "
            "CAPEX implications).\n\n"

            "STEP 5 — MINERALOGY-BASED ARD PRE-SCREEN.\n"
            "Based on the deposit type and likely mineral assemblage, assess the "
            "acid rock drainage (ARD) and acid mine drainage (AMD) potential:\n"
            "  • Identify the primary sulphide minerals likely present (pyrite, "
            "pyrrhotite, chalcopyrite, pentlandite, arsenopyrite) and their relative "
            "acid-generating potential.\n"
            "  • Pyrrhotite (common in magmatic Ni-Cu systems) is highly acid-generating "
            "— flag if present. Pyrite in oxidising environments generates sulphuric "
            "acid — flag grade and likely abundance.\n"
            "  • State whether the expected carbonate content of the host rock is "
            "likely to provide acid-neutralising capacity or not.\n"
            "  • This pre-screen gives the environmental engineer a geochemical "
            "baseline before any assay data is available.\n\n"

            "STEP 6 — METALLURGICAL PRE-SCREEN.\n"
            "Based on the mineralisation style, state the most likely processing route "
            "and flag any known metallurgical complexity:\n"
            "  • Sulphide copper-gold porphyry → conventional sulphide flotation to "
            "produce a copper-gold concentrate. Flag if the gold is likely refractory "
            "(locked in pyrite or arsenopyrite), which requires pressure oxidation or "
            "roasting and adds 30–50% to process plant CAPEX.\n"
            "  • Magmatic Ni-Cu-PGE → sulphide flotation to a Ni-Cu-PGE concentrate "
            "or direct smelting feed. Flag pentlandite/pyrrhotite ratio implications.\n"
            "  • Sediment-hosted Cu → may be amenable to heap leach (oxide zone) or "
            "flotation (sulphide zone). Flag the likely oxide/sulphide ratio.\n"
            "  • Hydrothermal Cu-Au metasedimentary → typically flotation; flag skarn "
            "complexity if present.\n"
            "  • This pre-screen tells the economist which processing route to price "
            "and prevents a systematic CAPEX underestimate.\n\n"

            "STEP 7 — TCFD PHYSICAL RISK FLAG (geological dimension only).\n"
            "Identify physical climate risks that are visible from the geological "
            "setting and that institutional investors will expect to see disclosed "
            "under TCFD (Task Force on Climate-related Financial Disclosures):\n"
            "  • Alpine/glaciated settings: glacial retreat affecting meltwater "
            "hydrology, increased debris flow and avalanche frequency, permafrost "
            "degradation affecting infrastructure foundations.\n"
            "  • Arid/desert settings: worsening water scarcity under projected "
            "climate scenarios, salt lake acid sulphate soil risks.\n"
            "  • Coastal or lowland settings: sea level and flooding risk.\n"
            "  • State only risks that are physically grounded in the reported "
            "geological and geographic setting. Do not speculate beyond what the "
            "data supports.\n\n"

            "STEP 8 — KEY SUBSURFACE RISKS AND DRILL PROGRAM FORECAST.\n"
            "Rank the top geological risks that would affect feasibility if unresolved. "
            "Then prescribe the follow-up drill program with specifics: recommended "
            "hole count, orientation (azimuth and dip), target depth, drill spacing, "
            "and the specific geological hypothesis each hole is designed to confirm "
            "or refute. A forecast that says 'further drilling required' is not "
            "acceptable — state the program as if you are signing the exploration "
            "licence application.\n\n"
            f"{_OUTPUT_DISCIPLINE}"
        ),
        expected_output=(
            "A formal JORC 2012 / CRIRSCO-aligned geological pre-assessment written "
            "in authoritative technical prose. Write for two audiences: a technical "
            "peer who can validate your JORC classification, and an institutional "
            "investor who needs to understand what the numbers mean for capital risk. "
            "No box art, tables, decorative characters, LaTeX, or markdown. Plain "
            "prose and numbered lists only. Output EXACTLY the following sections "
            "in EXACTLY this order, using the exact header text shown:\n\n"

            "Mineralisation Style & Geological Setting:\n"
            "Classify the deposit type with precision. Describe the structural "
            "controls, host rock, and alteration assemblage implied by the reported "
            "geological setting. State how this setting compares to known producing "
            "districts of the same deposit type in Canada or Australia.\n\n"

            "Deposit Analogue Comparison:\n"
            "Name one or two real producing or advanced-stage mines of the same "
            "deposit type in a geologically comparable jurisdiction. For each analogue "
            "state: mine name, jurisdiction, deposit type, reported head grade, "
            "typical intersection widths, and JORC/NI 43-101 resource size. Then "
            "explicitly state whether this prospect's reported grade is above, at, "
            "or below the analogue median, and whether the intersection width is "
            "consistent with the analogue ore body geometry. This section is the "
            "economist's CAPEX benchmark — it must contain real numbers.\n\n"

            "Resource Confidence Classification:\n"
            "State the assigned JORC 2012 / CIM category: Inferred Mineral Resource "
            "or JORC clause 18 Exploration Target. If Inferred: justify why geological "
            "continuity can be assumed from this setting and drill data. If Exploration "
            "Target: state the grade-tonnage range and include verbatim the JORC "
            "conceptual disclaimer. Flag every data gap that prevents a higher "
            "classification. Close with: 'This assessment is an internal pre-CP "
            "opinion. Public disclosure requires sign-off by a Competent Person "
            "(JORC) or Qualified Person (NI 43-101).'\n\n"

            "Ore Body Geometry Estimate:\n"
            "State the minimum and maximum plausible true width of the ore zone and "
            "the geological reasoning for each bound. State the implied strike length "
            "range and dip if inferrable. Conclude with a direct statement of which "
            "mining method — open-pit, bulk underground, or narrow-vein underground "
            "— the geometry range physically supports, and the width threshold that "
            "determines the boundary between methods. This section is the COO's "
            "capital-geology validation input — it must contain specific dimensions.\n\n"

            "Grade & Tonnage Indication:\n"
            "Provide an actual numerical order-of-magnitude range — for example "
            "'5 to 50 Mt at 0.4 to 0.8% CuEq' — derived by scaling the reported "
            "intersection against the analogue mine geometries cited above. State "
            "the scaling assumptions explicitly. Do NOT use placeholder tokens such "
            "as '[X Mt]', '[A%]', or 'TBD'. If the data is too sparse to support "
            "even a wide conceptual range, state that explicitly and explain why — "
            "do not substitute a placeholder. Flag the estimate as conceptual if "
            "based on an Exploration Target.\n\n"

            "Geotechnical Pre-Screen:\n"
            "Qualitative rock mass assessment: expected competency, key structural "
            "risks (fault zones, foliation, shear zones), and depth-to-target "
            "implications for access method (decline vs. shaft). Flag any condition "
            "that would materially increase underground development cost.\n\n"

            "ARD & Metallurgical Pre-Screen:\n"
            "Two sub-sections in one:\n"
            "ARD: identify the primary sulphide minerals likely present, their "
            "acid-generating potential, and whether the host rock provides "
            "acid-neutralising capacity. Flag pyrrhotite or high-pyrite assemblages "
            "as elevated ARD risk.\n"
            "Metallurgy: state the most likely processing route (flotation, heap "
            "leach, direct smelting) and flag any known complexity — refractory gold, "
            "pentlandite/pyrrhotite selectivity issues, oxide/sulphide transition "
            "depth — that would increase process plant CAPEX.\n\n"

            "TCFD Physical Risk Flag:\n"
            "Identify geological and geographic physical risks that climate change "
            "will amplify and that institutional investors will expect disclosed "
            "under TCFD. State only risks grounded in the reported setting. Examples: "
            "glacial retreat hydrology, avalanche frequency, extreme water scarcity, "
            "permafrost degradation. One to three sentences per risk identified.\n\n"

            "Key Subsurface Risks:\n"
            "Ranked list of the top geological risks that would affect feasibility "
            "if unresolved. For each risk state: what it is, why it matters, and "
            "what drilling or testing would resolve it.\n\n"

            "Drill Program Forecast:\n"
            "A specific, actionable drill program prescription: recommended hole "
            "count, orientation (azimuth and dip), target depth, drill spacing, and "
            "the exact geological hypothesis each hole is designed to confirm or "
            "refute. State the decision gate — what result would support advancing "
            "to PEA and what result would lead to project cancellation."
        ),
        agent=agents["geologist"],
        guardrail=_geology_guardrail,
        guardrail_max_retries=2,
        callback=_cb_geology,
    )

    env_task = Task(
        description=(
            f"{prospect_brief}\n"
            "Using the prospect briefing above and the geological assessment in "
            "your context, deliver an ESIA-grade pre-assessment covering the "
            "full environmental, regulatory, and social landscape. Work through "
            "every step below in order and do not skip any.\n\n"

            "STEP 1 — JURISDICTION IDENTIFICATION AND REGULATORY STACK:\n"
            "Identify the host country and province/state from the prospect data. "
            "List every applicable regulatory instrument by its formal name: for "
            "Canada include the Impact Assessment Act 2019 (IAA), the provincial "
            "Environmental Assessment Act, the Mines Act, DRIPA 2019 (if BC), "
            "the Crown Duty to Consult (s.35 Constitution Act 1982), SARA, and "
            "the Fisheries Act; for Australia include the EPBC Act 1999 "
            "(Commonwealth), the relevant state Mining Act, the Native Title Act "
            "1993, the applicable state Aboriginal Heritage Act, and state "
            "Environmental Protection Act. Also list applicable international "
            "standards: IFC PS1-PS8, GISTM 2020, ICMM 10 Principles, IRMA, "
            "Equator Principles EP4, SASB Metals & Mining, TNFD, TCFD.\n\n"

            "STEP 2 — ECOLOGICAL SENSITIVITIES AND BIODIVERSITY:\n"
            "Assess biodiversity sensitivity using the geologist's deposit "
            "setting, elevation, drainage basin, and climate data. Flag any "
            "IFC PS6 Critical Habitat triggers or EPBC Act 'Matters of National "
            "Environmental Significance' (MNES). Identify TNFD-relevant "
            "nature-related dependencies (freshwater, intact forest, alpine "
            "ecosystems) and TCFD physical risks (glacial melt, flood, drought). "
            "Map to Mitigation Hierarchy: avoid, minimise, restore, offset.\n\n"

            "STEP 3 — TAILINGS AND ARD/AMD RISK:\n"
            "Using the geologist's mineralogy and ARD pre-screen, assess "
            "tailings management complexity under GISTM 2020 (governance, "
            "design, operation, and closure requirements). State the likely "
            "Consequence Classification (Low / Significant / High / Very High / "
            "Extreme) under GISTM and what that triggers for independent review "
            "and design standard. Assess acid-generating potential using acid-"
            "base accounting (ABA) logic from the geologist's sulphide flags. "
            "If ARD risk is elevated, state what water treatment infrastructure "
            "is implied and flag this cost to the economist.\n\n"

            "STEP 4 — WATER AND HYDROLOGY:\n"
            "Assess surface water and groundwater exposure. Identify water "
            "scarcity risk relevant to process water demands (IFC PS3/PS6, "
            "TCFD water risk). Flag any fish-bearing water bodies within the "
            "likely disturbance footprint (Canada: Fisheries Act s.35; "
            "Australia: state Water Acts). State whether a water licence or "
            "water management plan would be a critical path item.\n\n"

            "STEP 5 — PERMITTING PATHWAY:\n"
            "Define the specific permitting sequence and timeline for the host "
            "jurisdiction. For Canada: identify whether the project will trigger "
            "a federal IAA Designation under the Physical Activities Regulations, "
            "the provincial EA track, and the mines permit sequence. Estimate "
            "the critical-path duration from first community engagement to mine "
            "permit issuance — typically 5-9 years for a complex greenfield in "
            "BC; 3-5 years for brownfield in Ontario. For Australia: assess "
            "whether the project triggers an EPBC Act referral for MNES, the "
            "state-level EA process, and the environmental performance bond "
            "requirement. Estimate the critical-path duration.\n\n"

            "STEP 6 — SOCIAL LICENCE AND INDIGENOUS RIGHTS:\n"
            "Assess the FPIC obligation under the applicable framework: for "
            "Canada (BC) apply DRIPA 2019 and UNDRIP — both require meaningful "
            "consent, not just consultation; for Canada (non-BC) apply the "
            "Crown Duty to Consult framework; for Australia apply the Native "
            "Title Act 1993 (future act regime) and the applicable state "
            "Aboriginal Heritage Act. Identify whether any Traditional Owner "
            "or First Nations group has registered native title claims or "
            "treaty rights over the prospect area. Assess Equator Principles "
            "EP4 applicability (project financing > USD 10M). Flag any ILO "
            "Convention 169 obligations. State what grievance mechanism "
            "(IFC PS1) would be required at a minimum.\n\n"

            "STEP 7 — ESG RED FLAGS AND RESOLVABILITY VERDICTS:\n"
            "List every material ESG risk you have identified in Steps 1-6. "
            "For EACH risk assign one of three explicit verdicts:\n"
            "  RESOLVABLE: a standard mitigation pathway exists at "
            "proportionate cost and the permitting route is well-established.\n"
            "  BORDERLINE: a mitigation pathway exists but carries significant "
            "cost or scheduling uncertainty; warrants a specific feasibility "
            "study before the next stage.\n"
            "  POTENTIALLY FATAL: as a CATEGORY OF PROBLEM, no engineering "
            "mitigation pathway exists anywhere in the world at any cost — "
            "not merely that the site-specific study has not yet been done. "
            "This threshold is deliberately extreme and rare.\n\n"
            "CALIBRATION EXAMPLES — use these to anchor your verdicts:\n"
            "  RESOLVABLE examples: standard tailings facility on competent "
            "ground; a water licence application in a well-watered jurisdiction; "
            "community grievance mechanism design.\n"
            "  BORDERLINE examples: glacial melt and avalanche hazard in alpine "
            "mining terrain (geohazard studies and infrastructure siting are "
            "proven engineering controls, but site-specific uncertainty is high); "
            "FPIC process under DRIPA where consent outcome is uncertain; "
            "ARD/AMD where kinetic testing results are not yet available.\n"
            "  POTENTIALLY FATAL examples: a mine footprint entirely within a "
            "legally designated Critical Habitat with zero variance pathway under "
            "national law; confirmed catastrophic ARD where the neutralising "
            "capacity is zero and perpetual water treatment at the required scale "
            "exceeds the project's lifetime revenue; a project in a jurisdiction "
            "where the government has legislatively prohibited all new mining of "
            "the target commodity.\n"
            "  DO NOT rate as POTENTIALLY FATAL: glacial retreat, avalanche risk, "
            "permafrost, seismicity, or any other geohazard that has established "
            "engineering controls and for which comparable mines operate "
            "successfully in similar terrain worldwide. These are BORDERLINE.\n"
            "  DO NOT rate as POTENTIALLY FATAL: any risk where the verdict is "
            "'we need more study.' That is the definition of BORDERLINE.\n"
            "If any risk is genuinely POTENTIALLY FATAL, state this explicitly "
            "for the COO.\n\n"

            "STEP 8 — TNFD AND TCFD COMPLIANCE FLAGS:\n"
            "List any nature-related disclosures required under the TNFD "
            "framework that institutional lenders will request at the PEA "
            "stage. List any climate-related disclosures required under TCFD "
            "(physical and transition risks for the mine plan and closure). "
            "State the SASB Metals & Mining disclosure topics that are material "
            "for this project type.\n\n"

            "STEP 9 — RECOMMENDED MITIGATIONS:\n"
            "For every RESOLVABLE and BORDERLINE risk, specify the engineering "
            "or management action, the applicable standard (e.g., GISTM "
            "Consequence Classification X requires Y design standard), the "
            "estimated order-of-magnitude cost implication to the economist, "
            "and the responsible party. Do not propose mitigations for risks "
            "rated POTENTIALLY FATAL.\n\n"
            f"{_OUTPUT_DISCIPLINE}"
        ),
        expected_output=(
            "A structured ESIA pre-assessment written in plain prose for a "
            "senior mining executive or institutional lender audience. "
            "FORMATTING RULES: Do NOT use box art, Unicode borders, tables, "
            "bullet grids, or any decorative characters (╔ ║ ═ ╚ etc.). "
            "Use ONLY plain text paragraphs and numbered/lettered lists where "
            "appropriate. You MUST produce ALL EIGHT sections below in full — "
            "do not stop early, do not truncate any section. Write each section "
            "header on its own line EXACTLY as shown — keep the ampersand "
            "character '&' (never substitute the word 'and'), keep 'ARD/AMD' "
            "and 'TNFD & TCFD' with no whitespace around the slash, and keep "
            "the trailing colon. Headers must match the canonical form below "
            "verbatim, followed immediately by the section content:\n\n"

            "Jurisdiction & Regulatory Framework:\n"
            "The applicable regulatory instruments by formal name — federal, "
            "provincial/state, and international ESG standards — and one "
            "sentence on each explaining why it applies to this specific project.\n\n"

            "Ecological Sensitivities:\n"
            "Biodiversity, protected areas, watercourses, and climate physical "
            "risks, citing IFC PS6 Critical Habitat criteria, EPBC MNES "
            "triggers (Australia), or SARA Species at Risk (Canada) as "
            "applicable. State TNFD nature-related dependencies.\n\n"

            "Tailings & ARD/AMD Risk:\n"
            "Tailings management complexity under GISTM 2020 including the "
            "Consequence Classification assessment. ARD/AMD potential based "
            "on the geologist's sulphide flags, with the implied water "
            "treatment infrastructure cost flagged for the economist.\n\n"

            "Permitting Pathway:\n"
            "The full permitting sequence for the specific jurisdiction "
            "(federal and provincial/state), the critical-path duration from "
            "first engagement to mine permit, and the two or three items most "
            "likely to cause schedule delay.\n\n"

            "Social Licence & Indigenous Rights:\n"
            "The FPIC framework that applies (DRIPA/UNDRIP for BC; Duty to "
            "Consult for other Canadian provinces; Native Title Act for "
            "Australia), known or probable native title/treaty claims in the "
            "area, Equator Principles EP4 applicability, and the minimum "
            "grievance mechanism required under IFC PS1.\n\n"

            "ESG Red Flags & Resolvability Assessment:\n"
            "Each material ESG risk listed with its explicit verdict in "
            "CAPITALS: RESOLVABLE, BORDERLINE, or POTENTIALLY FATAL. "
            "Provide one to two sentences of justification per risk. "
            "Remember: geohazards (glacial retreat, avalanche, seismicity, "
            "permafrost) that have established engineering controls are "
            "BORDERLINE — never POTENTIALLY FATAL — because comparable mines "
            "worldwide operate successfully in similar terrain. Risks that "
            "require further study are BORDERLINE, not POTENTIALLY FATAL. "
            "POTENTIALLY FATAL is reserved for categorical impossibilities "
            "(e.g. legally prohibited land, zero neutralising capacity with "
            "perpetual treatment costs exceeding lifetime revenue). "
            "If any risk is rated POTENTIALLY FATAL, conclude this section "
            "with: FATAL FLAW DETECTED — [name of risk] — and a single "
            "sentence explaining why no engineering solution exists anywhere. "
            "If no risk is POTENTIALLY FATAL, conclude with: NO FATAL "
            "ENVIRONMENTAL OR SOCIAL FLAW DETECTED AT THIS STAGE.\n\n"

            "TNFD & TCFD Compliance Flags:\n"
            "Nature-related disclosures required under TNFD that lenders "
            "will request at PEA stage. Climate-related disclosures under "
            "TCFD (physical and transition risks). Material SASB Metals & "
            "Mining disclosure topics for this project.\n\n"

            "Recommended Mitigations:\n"
            "For each RESOLVABLE and BORDERLINE risk: the specific engineering "
            "or management action, the applicable standard that mandates it, "
            "the order-of-magnitude cost implication for the economist, and "
            "the party responsible. No mitigations for POTENTIALLY FATAL risks."
        ),
        agent=agents["env_engineer"],
        context=[geology_task],
        guardrail=_env_guardrail,
        guardrail_max_retries=2,
        callback=_cb_env,
    )

    economy_task = Task(
        description=(
            f"{prospect_brief}\n\n"
            "Using the geological and environmental assessments in your context, "
            "produce an institutional-quality Scoping-level financial pre-assessment. "
            "Work through every step below in order and do not skip any.\n\n"

            "STEP 1 — DEPOSIT POSITIONING ON THE LASSONDE CURVE:\n"
            "State the project's current stage (Early Discovery) and what that means "
            "for intrinsic value versus option value. Identify the closest comparable "
            "real mine transactions or PEA/PFS studies for this exact deposit type "
            "(e.g. Red Chris, Cadia-Ridgeway, Porphyry Belt analogues for porphyry "
            "Cu-Au). State what market capitalisation per resource ounce or per pound "
            "of copper those analogues commanded at equivalent project stages, so the "
            "COO can benchmark current implied value.\n\n"

            "STEP 2 — RESOURCE SCENARIO FRAMEWORK:\n"
            "From the single reported drill intersection, construct three tonnage "
            "scenarios — Bear, Base, and Bull — by applying conservative, expected, "
            "and optimistic strike/dip continuity assumptions consistent with the "
            "deposit type and analogue systems. State the geological assumption "
            "explicitly for each scenario. Calculate the minimum mineable tonnage "
            "required for the NPV to turn positive at the Base commodity price and "
            "state whether the geological evidence supports that tonnage. This is the "
            "single most important number the COO needs from this report.\n\n"

            "STEP 3 — CAPEX ESTIMATION:\n"
            "Estimate two CAPEX figures: (A) Next-Stage Capital — the resource-"
            "definition drill program cost, environmental baseline studies, and FPIC "
            "process costs required before PEA can begin; and (B) Mine Development "
            "CAPEX — order-of-magnitude total mine buildout covering: drill program, "
            "access infrastructure scaled to the logistics distance, process plant "
            "sized to the mineralisation style and likely throughput, tailings "
            "facility under GISTM Consequence Classification, camp and logistics, "
            "and all ESG capital items identified by the environmental engineer "
            "(ARD/AMD controls, water management infrastructure, community investment "
            "fund establishment, carbon offset provision, tailings closure bond). "
            "Label both with ±40% Scoping-level uncertainty. Name the real mine "
            "analogues informing each major line item.\n\n"

            "STEP 4 — OPEX AND AISC ESTIMATION:\n"
            "Estimate annual OPEX covering: mining costs by method (open-pit strip "
            "ratio, truck-and-shovel fleet, or underground development costs), "
            "processing costs by flowsheet complexity (simple flotation vs. "
            "pressure oxidation for refractory gold), G&A, and all ESG compliance "
            "costs as explicit named line items — BC carbon tax or Australian "
            "Safeguard Mechanism carbon pricing, water stewardship levy, community "
            "development agreement annual obligation, tailings facility monitoring "
            "and maintenance. Compute the All-In Sustaining Cost (AISC) per unit "
            "of payable metal and compare it to the current industry AISC quartile "
            "benchmarks so the COO can see where this project sits in the global "
            "cost curve.\n\n"

            "STEP 5 — COMMODITY PRICE ANALYSIS:\n"
            "State: (i) current spot price; (ii) consensus long-run price from "
            "major bank analyst forecasts (use the average of the four largest "
            "commodity research desks as the Base case); (iii) Bear case price "
            "(P10 trough of the 20-year historical price cycle); (iv) Bull case "
            "price (P90 peak). Show the price sensitivity: at what commodity price "
            "does the project's IRR fall below 10%? This is the margin of safety "
            "the COO needs to know.\n\n"

            "STEP 6 — DCF VALUATION — THREE SCENARIOS:\n"
            "Run a Bear, Base, and Bull DCF using the three tonnage scenarios from "
            "Step 2 and the three commodity prices from Step 5. For each scenario "
            "report: NPV at 5%, 8%, and 10% discount rates (state which Fraser "
            "Institute jurisdiction tier informs your base discount rate); IRR; "
            "simple payback period; and whether the project clears a 15% IRR "
            "hurdle at the Base case. Label all figures ±40% Scoping accuracy. "
            "Explicitly state the minimum mineable tonnage assumption underpinning "
            "each NPV so the COO can validate it against the geological evidence.\n\n"

            "STEP 7 — ESG INITIATIVES AND VALUE ENHANCEMENT:\n"
            "This section positions ESG not as a cost but as a source of competitive "
            "advantage and incremental NPV. Assess and quantify where applicable:\n"
            "  - Carbon credit revenue: eligibility for Verified Carbon Standard (VCS) "
            "or Gold Standard credits from avoided deforestation, mine rehabilitation, "
            "or methane capture; estimated annual revenue per tonne of CO2e avoided.\n"
            "  - Green financing premium: eligibility for a Green Bond or "
            "Sustainability-Linked Loan facility and the basis-point reduction in "
            "cost of capital that implies (typically 25-75 bps), and its NPV impact.\n"
            "  - IRMA certification: the Standard for Responsible Mining IRMA "
            "certification pathway and the documented 5-15% offtake price premium "
            "that technology-sector buyers (battery manufacturers, EV OEMs) pay to "
            "responsibly certified copper-gold producers; estimated annual revenue uplift.\n"
            "  - TNFD/biodiversity credits: emerging nature-related credit markets "
            "and whether the project footprint qualifies for biodiversity net gain "
            "credit generation post-restoration.\n"
            "  - Indigenous equity partnership: model the NPV impact of offering a "
            "5-10% equity stake or a net profits royalty to the relevant First Nations "
            "or Traditional Owners, versus the cost of social licence failure (which "
            "industry data shows averages USD 20M per month of delay for a major "
            "project). Show the net present value of partnership versus conflict.\n"
            "  - Circular economy opportunity: potential for tailings reprocessing "
            "revenue or mine waste aggregate sales that reduce net closure liability.\n\n"

            "STEP 8 — PROJECT FINANCING STRUCTURE:\n"
            "Propose the most realistic capital structure for a project at this stage "
            "and in this jurisdiction. Cover: (i) equity raise quantum and likely "
            "investors at the PEA stage (junior explorer funds, strategic investors); "
            "(ii) streaming and royalty financing eligibility — which metals, which "
            "streamers, typical advance rate against mine-buildout CAPEX; (iii) "
            "project finance debt (EP4 compliance requirements from the environmental "
            "engineer's report, IFC Performance Standards covenant requirements, "
            "estimated debt-to-equity ratio); (iv) offtake agreement role in securing "
            "lender consent. State whether IRMA certification or an ESG-linked "
            "facility would change the achievable leverage ratio.\n\n"

            "STEP 9 — KEY RISKS AND SENSITIVITIES:\n"
            "Quantify the four material financial risks: (i) commodity price decline "
            "— NPV impact of a 20% price fall from Base case; (ii) CAPEX overrun — "
            "using P90 historical overrun data for this project type (typically "
            "+30-50% for greenfield mines in remote alpine terrain), NPV impact; "
            "(iii) permitting delay — NPV cost per 12-month delay at the Base "
            "discount rate; (iv) ESG cost escalation — NPV impact if the BORDERLINE "
            "ESG risks identified by the environmental engineer are reclassified to "
            "POTENTIALLY FATAL after further study.\n\n"
            f"{_FINANCIAL_OUTPUT_DISCIPLINE}"
        ),
        expected_output=(
            "An institutional-quality Scoping financial pre-assessment written in "
            "business prose for an Investment Committee audience. Do NOT include "
            "LaTeX, equations, or visible calculation steps. Do NOT use box art or "
            "Unicode borders. You MUST produce ALL EIGHT sections below in full — "
            "do not stop early. Write each section header on its own line exactly "
            "as shown:\n\n"

            "Deposit Positioning & Comparables:\n"
            "The project's Lassonde Curve stage position, named comparable mine "
            "transactions or PEA/PFS studies for this deposit type, market "
            "capitalisation per resource unit at comparable stages, and what that "
            "implies for current implied value versus the capital being requested.\n\n"

            "Resource Scenario Framework:\n"
            "Bear, Base, and Bull tonnage scenarios derived from the drill "
            "intersection with the geological continuity assumption stated for each. "
            "The minimum mineable tonnage required for a positive NPV at Base "
            "commodity price, and whether the geological evidence supports it.\n\n"

            "CAPEX Estimate:\n"
            "Two figures: (A) Next-Stage Capital for the PEA drill program, "
            "environmental baselines, and FPIC process; and (B) Mine Development "
            "CAPEX with all major line items including ESG capital items from the "
            "environmental assessment. Both labelled ±40% Scoping accuracy. Named "
            "real mine analogues for each major line item.\n\n"

            "OPEX & AISC Estimate:\n"
            "Annual OPEX with ESG compliance costs as explicit named line items. "
            "All-In Sustaining Cost (AISC) per unit of payable metal compared to "
            "current global industry AISC quartile benchmarks, showing where this "
            "project sits in the cost curve. Labelled ±40% Scoping accuracy.\n\n"

            "Commodity Price Analysis:\n"
            "Current spot, consensus long-run Base case, Bear and Bull case prices. "
            "The price at which the project IRR falls below 10% — the margin of "
            "safety threshold the COO needs.\n\n"

            "DCF Valuation — Three Scenarios:\n"
            "Bear, Base, and Bull NPV at 5%, 8%, and 10% discount rates, with IRR "
            "and simple payback period for each. The minimum mineable tonnage "
            "assumption stated explicitly for each scenario. Whether the project "
            "clears a 15% IRR hurdle at the Base case. All figures labelled ±40% "
            "Scoping accuracy.\n\n"

            "ESG Initiatives & Value Enhancement:\n"
            "Dollar-quantified assessment of: carbon credit revenue eligibility, "
            "green financing basis-point savings and NPV impact, IRMA certification "
            "offtake premium and annual revenue uplift, TNFD biodiversity credit "
            "opportunity, indigenous equity partnership NPV versus cost of social "
            "licence failure, and circular economy revenue from tailings or waste.\n\n"

            "Project Financing & Key Risks:\n"
            "Proposed capital structure (equity, streaming, project finance debt, "
            "offtake). EP4 and IFC PS covenant requirements from the environmental "
            "assessment. Impact of IRMA certification or ESG-linked facility on "
            "leverage. Four quantified risk sensitivities: commodity price decline, "
            "CAPEX overrun (P90 historical), permitting delay (NPV cost per year), "
            "and ESG cost escalation."
        ),
        agent=agents["economist"],
        context=[geology_task, env_task],
        guardrail=_economy_guardrail,
        guardrail_max_retries=2,
        callback=_cb_economy,
    )

    coo_task = Task(
        description=(
            f"{prospect_brief}\n\n"
            f"{_STAGE_GATE_GLOSSARY}\n\n"
            "You have before you the complete geological, environmental, and financial "
            "assessments produced by three specialist agents. Read all three in full "
            "before writing a single word of your response. Then execute the following "
            "analytical steps in strict order. Every step is mandatory. If a step "
            "triggers REJECT, stop and write the report — do not continue to later "
            "steps.\n\n"

            "STEP 1 — CROSS-VALIDATION (mandatory before any other step).\n"
            "First: read the geologist's 'Mineralisation Style & Geological Setting' "
            "section and identify the deposit type. Classify it as one of:\n"
            "  • BULK (porphyry, mafic intrusion, disseminated sediment-hosted)\n"
            "  • NARROW-VEIN (hydrothermal vein-hosted, skarn, narrow shear-zone)\n"
            "This classification determines which geometry metric you apply in "
            "Step 2. State the deposit type and your classification at the start "
            "of your Cross-Validation notes.\n\n"
            "Then identify and resolve every internal contradiction between the "
            "three upstream assessments. The following are automatically escalated "
            "to the REJECT test in Step 2:\n"
            "  • The economist modelled open-pit mining but the relevant geometry "
            "metric (downhole length for bulk; true width for narrow-vein) is less "
            "than 30m — this is a capital-geology mismatch.\n"
            "  • The economist produced an NPV or IRR without stating the assumed "
            "mineable tonnage — arithmetically correct but geologically groundless.\n"
            "  • The geologist assigned Inferred Resources but the economist's "
            "production schedule implies Measured-grade continuity.\n"
            "  • The environmental engineer flagged no water constraint but the "
            "economist priced significant water infrastructure — or vice versa.\n"
            "For each contradiction found, write one sentence: what it is, which "
            "report is at fault, and whether it escalates to the REJECT test.\n\n"

            "STEP 2 — REJECT TEST.\n"
            "Issue REJECT immediately if ANY single condition below is present. "
            "These are not balanced against each other — one is enough:\n\n"
            "  BEFORE APPLYING (a) and (b), identify the deposit type from the "
            "geologist's report. The geometry metric differs by deposit type:\n"
            "  • BULK DEPOSITS (porphyry copper-gold, mafic intrusion Ni-Cu-PGE, "
            "disseminated sediment-hosted): The relevant dimension is the DOWNHOLE "
            "INTERSECTION LENGTH — the drilled envelope of mineralisation. A 114m "
            "downhole intersection through a porphyry system IS the ore body "
            "dimension, not a vein to be true-width corrected. Use the reported "
            "intersection length directly.\n"
            "  • NARROW-VEIN / STRUCTURALLY-CONTROLLED DEPOSITS (hydrothermal "
            "vein-hosted, skarn, narrow shear-zone): The relevant dimension is "
            "TRUE WIDTH of the ore zone. These systems do not extrapolate laterally "
            "the way bulk deposits do.\n\n"
            "  a) CAPITAL-GEOLOGY MISMATCH (Economics sub-condition):\n"
            "     EVALUATION SEQUENCE — follow this exact two-step test:\n"
            "     STEP A1 — measure the geometry: extract the reported downhole "
            "intersection length (BULK) or true width (NARROW-VEIN) as a plain "
            "number in metres.\n"
            "     STEP A2 — apply the threshold:\n"
            "       BULK DEPOSITS: If intersection >= 30m → condition (a) DOES NOT "
            "TRIGGER. Stop. Do not apply any CAPEX scaling. A 114m intersection "
            "passes; a 25m intersection does not.\n"
            "       BULK DEPOSITS: If intersection < 30m → REJECT because a "
            "sub-30m bulk-deposit envelope cannot support mine-scale capital.\n"
            "       NARROW-VEIN DEPOSITS: If true width >= 10m → condition (a) "
            "DOES NOT TRIGGER.\n"
            "       NARROW-VEIN DEPOSITS: If true width < 10m → REJECT because "
            "a sub-10m vein is a single data point with no volume extrapolation.\n"
            "     CRITICAL: The economist's mine-buildout CAPEX figure (which "
            "routinely exceeds USD 30M for any serious project) is NOT a factor "
            "in this test. The only number that determines condition (a) is the "
            "intersection length compared to the threshold above.\n\n"
            "  b) MINING METHOD MISMATCH (Economics sub-condition):\n"
            "     If the economist selected open-pit mining:\n"
            "       If intersection >= 30m (BULK) or true width >= 30m (NARROW-VEIN) "
            "→ condition (b) DOES NOT TRIGGER. A 114m porphyry intersection is "
            "entirely compatible with open-pit; do not flag this as a mismatch.\n"
            "       If intersection < 30m → REJECT because open-pit strip ratios "
            "are catastrophic on a sub-30m ore zone.\n"
            "     If the economist selected underground mining: condition (b) does "
            "not apply regardless of intersection length.\n\n"
            "  c) PRICE SUB-ECONOMIC (Economics sub-condition):\n"
            "     Issue REJECT if the grade × tonnage is sub-economic at any "
            "realistic long-run commodity price. A project that needs a price "
            "outlier to clear a positive NPV does not work.\n\n"
            "  d) ENVIRONMENT UNRESOLVABLE:\n"
            "     MANDATORY VOCABULARY RULE — read this before evaluating:\n"
            "       UNRESOLVED = a risk that requires further study or mitigation "
            "work before the next stage. This is NORMAL at early-stage exploration "
            "and is NOT a basis for REJECT.\n"
            "       UNRESOLVABLE = a risk for which NO mitigation pathway exists "
            "at ANY cost, under ANY regulatory scenario. This is rare and extreme.\n"
            "     Do NOT conflate these two words. A geohazard study requirement, "
            "a tailings design requirement, or a water management plan requirement "
            "are all UNRESOLVED items — they are binding conditions for PROCEED, "
            "not grounds for REJECT.\n"
            "     Issue REJECT under condition (d) ONLY if the environmental "
            "engineer's report contains the explicit phrase "
            "'FATAL FLAW DETECTED' in the ESG Red Flags section, OR if one of "
            "these three conditions is present and the engineer has rated it "
            "POTENTIALLY FATAL:\n"
            "       i)  No viable water source exists in an extreme-scarcity "
            "setting with no credible engineering alternative.\n"
            "       ii) Critical or protected habitat with zero permitting pathway "
            "under any regulatory scenario — confirmed by the engineer, not "
            "inferred from terrain description.\n"
            "       iii) Catastrophic ARD/AMD risk with no technically feasible "
            "containment at any capital cost — confirmed by the engineer.\n"
            "     If none of i, ii, or iii is confirmed POTENTIALLY FATAL by the "
            "engineer, condition (d) DOES NOT TRIGGER. Geohazards, avalanche risk, "
            "glacial melt, and permafrost that require further study are binding "
            "conditions for PROCEED — they are not grounds for REJECT.\n\n"
            "  e) SOCIAL LICENCE BLOCKED:\n"
            "     Documented, active indigenous opposition with no recorded "
            "negotiated pathway and no credible engagement history.\n\n"
            "  f) JURISDICTION CATASTROPHIC:\n"
            "     Armed conflict, rule-of-law breakdown, or near-certain expropriation "
            "within the capital payback horizon.\n\n"
            "  g) LOGISTICS UN-BRIDGEABLE:\n"
            "     No road, rail, port, or grid power within economic reach with no "
            "credible infrastructure development pathway.\n\n"
            "  h) FINANCIAL NEGATIVE:\n"
            "     Strongly negative NPV under the most optimistic realistic commodity "
            "price assumptions and the lowest defensible discount rate.\n\n"

            "STEP 3 — CAPITAL PROPORTIONALITY CHECK (only if no REJECT triggered).\n"
            "For the proposed next stage, answer all three questions explicitly:\n"
            "  3a. What capital is being authorised, and what is the total capital "
            "at risk if this stage proceeds and the project is then cancelled?\n"
            "  3b. What minimum mineable tonnage is required to generate the revenue "
            "that makes the economist's NPV positive? Is there any geological evidence "
            "from the drilled intersection that the ore body exists at that scale? "
            "If the answer to the second question is no, this is a basis for REJECT "
            "under condition (a).\n"
            "  3c. Is the proposed PEA drill program scaled to the intersection "
            "width and geological uncertainty — or is it scaled to a hoped-for "
            "resource size? A drill program should be sized to confirm or refute "
            "the current geological hypothesis, not to justify a predetermined mine "
            "plan.\n\n"

            "STEP 4 — STAGE SELECTION (only if no REJECT from Steps 2 or 3).\n"
            "  • PROCEED to PEA if the geological signal supports an Inferred Resource "
            "definition campaign and the ESG and financial foundations pass Steps 2–3.\n"
            "  • PROCEED to PFS only if the resource is already at Indicated — which "
            "requires prior infill drilling beyond a single discovery hole.\n"
            "  • PROCEED to DFS only if the resource is at Measured plus Indicated.\n"
            "Stage-skipping is never permitted regardless of grade or pressure.\n\n"

            "STEP 5 — BINDING CONDITIONS AND REVERSAL TRIGGERS (PROCEED only).\n"
            "Set three to four specific, binary, testable pre-conditions before the "
            "next stage's capital is committed. Each must state a measurable threshold "
            "— not an activity. Then set two bright-line stage-reversal triggers: "
            "specific findings during the next stage that immediately convert this "
            "PROCEED to a REJECT without further review. Scale the triggers to the "
            "actual intersection width and resource confidence level, not to an "
            "aspirational mine plan.\n\n"
            f"{_OUTPUT_DISCIPLINE}"
        ),
        expected_output=(
            "A formal COO Executive Decision Memorandum. Write in the authoritative, "
            "precise prose of a senior executive writing for a board of institutional "
            "investors — not for the technical team. No hedging, no padding, no "
            "generalities. Every claim must be traceable to a specific finding in "
            "one of the three upstream assessments. Do NOT use box art, tables, or "
            "decorative characters. Plain prose and numbered lists only.\n\n"
            "Output EXACTLY the following sections in EXACTLY this order. Do not add, "
            "rename, or omit any section. Key Rationale must be the final section.\n\n"
            "Recommendation: <choose exactly one: PROCEED to PEA | PROCEED to PFS | "
            "PROCEED to DFS | REJECT>\n\n"
            "Stage-Gate Position: <one precise sentence stating the transition being "
            "authorised and its immediate capital consequence, or the specific reason "
            "the project is rejected and what would need to change to reopen it>\n\n"
            "Executive Summary:\n"
            "Three paragraphs. Paragraph 1: the geological signal — what the data "
            "actually shows and what it means for capital deployment timing. Cite the "
            "specific grade, intersection length, depth, and JORC classification from "
            "the geological assessment. Paragraph 2: the environmental and social risk "
            "profile — identify the single most material ESG risk, state whether it "
            "is manageable or potentially fatal, and give a realistic permitting "
            "timeline for the jurisdiction. Paragraph 3: the financial picture — "
            "state the CAPEX order of magnitude, the IRR range under current commodity "
            "pricing, whether the project clears the investment hurdle at this data "
            "quality, and the key assumption whose failure would flip the economics.\n\n"
            "Capital Risk Assessment:\n"
            "One paragraph. State explicitly: (a) the estimated capital commitment "
            "required for the next stage, (b) the total capital at risk if the project "
            "advances through the next stage and is then cancelled, (c) whether that "
            "quantum is proportionate to the current confidence level of the resource, "
            "and (d) the single financial assumption the board is being asked to accept "
            "in authorising that expenditure. For REJECT decisions, state the capital "
            "preserved by this decision and the opportunity cost of not proceeding.\n\n"
            "Fatal Flaw Assessment:\n"
            "Systematic assessment of all eight fatal conditions (a through h). "
            "For each, write one sentence: triggered / borderline / clear, and why. "
            "You MUST address conditions (a) Capital-Geology Mismatch and "
            "(b) Mining Method Mismatch explicitly — state the drill intersection "
            "width, the economist's mining method, the proposed CAPEX, and your "
            "verdict. Do not skip these two. If REJECT: name the exact condition "
            "triggered, quote the specific upstream finding, and state the evidence "
            "that would need to change to reopen the file. If PROCEED: identify "
            "which condition came closest to triggering REJECT and explain precisely "
            "why it did not cross the threshold.\n\n"
            "Binding Conditions:\n"
            "For PROCEED — three to four specific, measurable, binary pre-conditions "
            "that must be confirmed before the next stage's capital is committed. "
            "Each must be testable: not 'further environmental work' but 'an "
            "independent ARD/AMD test programme returning net acid generation below "
            "X kg H2SO4/tonne across all sampled lithologies.' "
            "For REJECT — two to three specific findings or changes in the upstream "
            "data that would be sufficient to reopen the file for reconsideration. "
            "Number each condition.\n\n"
            "Stage-Reversal Triggers:\n"
            "For PROCEED — two specific discoveries during the next stage that would "
            "immediately convert this decision to REJECT without further review. State "
            "each as a bright line, not a range: 'if the infill drilling programme "
            "returns average grade below X% CuEq across Y metres, the project is "
            "cancelled.' For REJECT — write 'Not applicable.'\n\n"
            "Key Rationale:\n\n"
            "1. <First decisive factor. Cite the specific finding — grade, threshold, "
            "standard, or number — that made this the decisive factor.>\n\n"
            "2. <Second decisive factor. Be equally specific.>\n\n"
            "3. <Third decisive factor, or the single condition that would most "
            "immediately reverse this decision if it changed.>"
        ),
        agent=agents["coo"],
        context=[geology_task, env_task, economy_task],
        guardrail=coo_guardrail,
        guardrail_max_retries=2,
        callback=_cb_coo,
    )

    return {"geology": geology_task, "env": env_task, "economy": economy_task, "coo": coo_task}


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------
_FINANCIAL_HEADERS = [
    ("capex_summary",       "CAPEX Estimate:"),
    ("opex_summary",        "OPEX Estimate:"),
    ("roi_summary",         "ROI & Financial Viability:"),
    ("esg_risk_assessment", "ESG Financial Risk Assessment:"),
]


def _parse_financial_briefing(raw: str) -> Optional[FinancialBriefing]:
    """Extract the four named sections from the economist's markdown output.

    Returns None if any header is missing or the resulting object fails
    Pydantic validation. Parsing is intentionally lenient so that a slightly
    off-format response still produces a usable FinancialBriefing.
    """
    matches = []
    for field, header in _FINANCIAL_HEADERS:
        match = re.search(re.escape(header), raw)
        if not match:
            return None
        matches.append((field, match.start(), match.end()))

    sections: Dict[str, str] = {}
    for i, (field, _, end) in enumerate(matches):
        next_start = matches[i + 1][1] if i + 1 < len(matches) else len(raw)
        sections[field] = raw[end:next_start].strip()

    try:
        return FinancialBriefing(**sections)
    except Exception as exc:
        log.warning("Could not parse financial briefing into FinancialBriefing: %s", exc)
        return None


def _format_financial_section(briefing: FinancialBriefing) -> str:
    return (
        "### CAPEX Estimate\n\n"
        f"{briefing.capex_summary}\n\n"
        "### OPEX Estimate\n\n"
        f"{briefing.opex_summary}\n\n"
        "### ROI & Financial Viability\n\n"
        f"{briefing.roi_summary}\n\n"
        "### ESG Financial Risk Assessment\n\n"
        f"{briefing.esg_risk_assessment}"
    )


def _task_raw(tasks: Dict[str, Task], key: str) -> str:
    """Return a task's raw output text, normalised and None-guarded."""
    output = tasks[key].output
    if output is None:
        return f"_[{key} task output unavailable]_"
    raw = output.raw or f"_[{key} task produced empty output]_"
    return _strip_inline_markdown(raw)


def _assemble_master_document(prospect_brief: str, tasks: Dict[str, Task], coo_result) -> str:
    raw_economy = _task_raw(tasks, "economy")
    parsed = _parse_financial_briefing(raw_economy)
    financial_md = _format_financial_section(parsed) if parsed is not None else raw_economy

    # Normalise the COO result the same way _task_raw normalises other sections
    # so that every section in the document has uniform font weight and style.
    coo_text = _strip_inline_markdown(str(coo_result))

    return (
        f"{prospect_brief}\n\n"
        "## Geological Assessment\n\n"
        f"{_task_raw(tasks, 'geology')}\n\n"
        "## Environmental Impact\n\n"
        f"{_task_raw(tasks, 'env')}\n\n"
        "## Financial Briefing\n\n"
        f"{financial_md}\n\n"
        "## COO Executive Summary\n\n"
        f"{coo_text}"
    )


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("_", name).strip("_") or "prospect"


def _save_output(prospect_name: str, document: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(prospect_name)}_{date.today().isoformat()}.md"
    path = output_dir / filename
    path.write_text(document, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_evaluation(
    prospect_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Run a full CrewAI evaluation for the given prospect and return the report.

    ``progress_callback`` is optional and, when supplied, gets invoked with
    the role string of each agent as that agent's task completes. Roles are
    one of ``"geology"``, ``"env"``, ``"economy"``, ``"coo"`` and arrive in
    crew execution order (sequential pipeline). The callback runs on the
    crew's executing thread, so callers writing into shared UI state should
    use a thread-safe queue rather than mutating UI widgets directly.
    """
    log.info("Starting evaluation for prospect: %s", prospect_name)

    prospect_brief = _build_prospect_brief(prospect_name)
    llms = _build_llms()
    agents = _build_agents(llms)
    tasks = _build_tasks(
        agents,
        prospect_name,
        prospect_brief,
        progress_callback=progress_callback,
    )

    crew = Crew(
        agents=[agents["geologist"], agents["env_engineer"], agents["economist"], agents["coo"]],
        tasks=list(tasks.values()),
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()

    document = _assemble_master_document(prospect_brief, tasks, result)
    log.info("Evaluation complete for prospect: %s", prospect_name)
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CrewAI mining prospect evaluation.",
    )
    parser.add_argument(
        "prospect",
        nargs="?",
        default=None,
        help="Prospect name (defaults to the first row in the dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write the master report (default: outputs/).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing the report to disk; print to stdout only.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available prospect names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for name in _get_dataframe()["Prospect_Name"].tolist():
            print(name)
        return

    name = args.prospect or _get_dataframe()["Prospect_Name"].iloc[0]
    document = run_evaluation(name)

    if not args.no_write:
        path = _save_output(name, document, Path(args.output_dir))
        log.info("Report written to %s", path)

    print(document)


if __name__ == "__main__":
    main()
