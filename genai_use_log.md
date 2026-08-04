# GenAI Use Log — Dissertation Project

**Purpose:** Running factual record of how Claude (Anthropic) was used throughout
this project, organized by the categories required for the mandatory GenAI
chapter (per module leader email, [date]). This is raw source material —
the actual chapter should be drafted in first person, in your own reflective
voice, using this log as evidence. Update this after each working session.

---

## 1. Literature Review Support

- Reviewed three academic papers already sourced independently and uploaded
  to project knowledge (Pappalardo et al. 2019 dataset paper; Aalbers & Van
  Haaren role-classification paper; Narizuka & Yamazaki formation-clustering
  paper) to help identify which were directly relevant to specific pipeline
  steps — e.g. recognising the Aalbers & Van Haaren paper as methodologically
  relevant to the role-assignment problem before that step was built.
- When additional references were considered for the literature review
  (e.g. Duch et al. 2010, Grund 2012, Peña & Touchette 2012, Buldú et al.
  2018, Gyarmati & Anguera 2015, Gonçalves et al. 2017, McInnes et al. 2018
  UMAP paper), Claude was asked to verify each citation was real via web
  search before it was accepted for inclusion — exact title, authors, year,
  and venue were checked against actual search results, not generated from
  memory. [Note: still need to independently read and critically engage with
  each of these papers yourself before citing — verification confirms a
  citation is real, not that you've engaged with its content.]

## 2. Objective Setting and Scope Decisions

- Discussed whether "one team, one season" (Manchester City, 2017/18 Premier
  League) was sufficient scope given the 9,000-12,000 word budget. Claude
  argued for depth-over-breadth and mapped a rough word budget across
  chapters; recommended possession-episode-level (rather than match-level)
  granularity specifically to give RQ1's clustering step enough data points.
  Final decision to proceed with this scope was mine, informed by this
  discussion and by the proposal's own risk-mitigation plan.
- Discussed team choice (Man City vs. alternatives e.g. Napoli) — Claude
  gave reasoning for Man City (tactical variety, data availability, fit to
  RQ1/RQ2) and explicitly flagged a limitation (32W/4D/2L record giving low
  outcome variance for RQ3) before I confirmed the choice.

## 3. Evaluating Methodological Choices

Two major methodological decisions were developed iteratively, with Claude
proposing an approach, testing it computationally against the real dataset,
and both of us evaluating the quantitative result before accepting, revising,
or discarding it:

**Possession-episode segmentation:**
- Naive approach (episode boundary = any change in acting team) tested first;
  found to produce 584 spurious episodes in a single sample match, 67.8% of
  which lasted under 3 seconds.
- Revised to an action-event-based approach (Duel/Interruption/Foul events do
  not themselves split possession); retested — reduced to 268 segments in the
  same match, still with residual noise.
- Applied a minimum-threshold rule (>=2 passes, >=3 seconds) as a final
  filter, run across the full season: 5,397 raw episodes -> 3,048 valid
  episodes (56.5% retained), ~80 valid episodes/match.
- I reviewed and accepted this design; the specific threshold values were a
  joint judgement call, not derived from a formal optimisation.

**Player role assignment:**
- Initial design (Option A): 3x3 positional zone grid (depth x width
  tercile). Tested across the full season — found a 100% match collision
  rate (multiple players assigned the same role label within a match).
- Diagnosed as a substitution-handling problem; revised with time-segmented
  boundaries at substitution minutes. Retested — collision rate barely
  improved (83.2% of segments still collided).
- Root cause identified: a 9-zone grid can never cleanly fit 10 outfield
  players (pigeonhole principle) — this was a structural flaw in Option A,
  not a fixable parameter issue.
- Pivoted to Option B: Hungarian-algorithm (optimal one-to-one) matching of
  10 outfield players to 10 fixed reference role-slot coordinates, combined
  with the substitution time-segmentation already built. First test showed
  zero collisions (guaranteed by construction) but tactically implausible
  assignments (a deep-lying midfielder and an attacking midfielder assigned
  to central-defence slots).
- Diagnosed as a coordinate-scale mismatch (template assumed full 0-100
  pitch spread; real averaged positions cluster narrower). Fixed via
  per-segment min-max normalisation of player positions before matching.
- Final validated result: 1,504 role assignments across the season, 1
  collision out of 143 segments (99.3% collision-free), the single
  exception attributable to a rare goalkeeper substitution near a
  segment boundary.
- I reviewed each iteration's quantitative results and made the calls to
  accept, reject, or request a redesign at each stage; Claude did not choose
  the final method unilaterally.

## 4. Design and Technical Decisions

- Overall project/folder structure (data/raw, src modules, notebooks)
  suggested by Claude, adopted after review.
- Decision to keep events data as both-teams-per-match (not filtered to only
  City's own events) was made jointly, to preserve context needed for
  possession-episode segmentation.
- Choice of LaTeX document structure in Overleaf (title page, declaration,
  abstract, chapter skeleton) was built to match the official module
  guidance document, which I uploaded and Claude read directly to extract
  the actual formatting/structural requirements (word count, chapter
  headings, declaration wording) rather than relying on general assumptions.

## 5. Code Development and Testing

- I set an explicit preference early on: Claude provides code as plain
  blocks in chat with block-level (not line-by-line) explanations, and I
  type/paste it into my own notebook and `.py` files myself, rather than
  Claude generating notebook files directly. This was maintained throughout.
- Before giving me any code, Claude generally ran it independently first (in
  a separate sandboxed environment) against copies of my actual data, to
  verify it worked and to cross-check outputs against known real-world facts
  (e.g. confirming computed pass-accuracy and passes-per-match figures
  against publicly known statistics for Man City's 2017/18 season) before
  recommending I use it.
- Debugging support was used repeatedly for errors I introduced while
  manually retyping code, including:
  - `ModuleNotFoundError` caused by a folder-structure mismatch (code
    assumed a `notebooks/` + `src/` layout; my actual files were flat in one
    folder) — diagnosed via a diagnostic cell Claude asked me to run first.
  - `IndexError` in `assign_roles_for_season`, caused by a mistranscription
    (`matches_by_id = []` instead of a dictionary comprehension, and a
    missing `all_results = []` initialisation) — diagnosed by requesting and
    reading my actual file content rather than guessing.
  - Several typos from manual retyping (e.g. `'substitution'` vs
    `'substitutions'` dictionary key, `golakeepers` vs `goalkeepers` variable
    name) — identified by careful line-by-line comparison against the
    intended source.
  - A LaTeX compilation failure caused by an accidentally deleted preamble
    (missing `\documentclass`/`\usepackage` lines above `\begin{document}`).

## 6. How Prompts Were Constructed

- Prompts were generally natural, conversational requests rather than
  engineered/structured prompts (e.g. "let's start step by step", "explain
  this block", "why is this happening", "help me get the data"), reflecting
  an iterative working style rather than one-shot prompt engineering.
- I set explicit standing preferences partway through the project (e.g.
  "explain each block, not line by line"; "give me code in chat, I'll type
  it myself") which shaped how Claude responded for the remainder of the
  project.
- When debugging, I provided exact error tracebacks and screenshots rather
  than paraphrasing the problem, which allowed for precise diagnosis rather
  than guesswork.

## 7. How Outputs Were Evaluated Before Use

- Every proposed methodological change was tested against the real dataset
  and the quantitative result reviewed before acceptance (see Section 3) —
  nothing was accepted purely on the basis of Claude's explanation.
- Factual/statistical claims were spot-checked against independently
  verifiable sources: e.g. cross-checking derived statistics (season pass
  counts, top passers by volume) against known facts about the actual
  2017/18 season; requiring web-search verification of literature citations
  before accepting them into the reference list.
- Where Claude's proposed code produced results inconsistent with domain
  knowledge (e.g. the first Hungarian-matching attempt assigning defensive
  midfielders to centre-back slots), this was caught by inspecting the
  output against known player positions, not accepted at face value.
- [Add specific examples here as they arise in remaining project stages —
  e.g. clustering results, dynamics analysis, outcomes analysis.]

## 8. Decisions Not to Follow Claude's Suggestions / Independent Judgement

- [To complete: note any cases where a Claude suggestion was declined or
  substantially altered rather than accepted — e.g. any point where you
  chose a different approach than the one initially proposed. Worth
  populating this honestly, since the reflection should show independent
  critical judgement, not passive acceptance.]

---

## Chronological Session Notes

*(Brief dated entries — expand as the project continues)*

- **Project setup & data sourcing:** Guidance on obtaining the Wyscout
  dataset, project folder structure, and a Python script (written jointly)
  to filter the large `events_England.json` file down to Man City's matches
  to fit upload size constraints.
- **Possession-episode segmentation:** See Section 3.
- **Role assignment:** See Section 3.
- **GitHub repository setup:** Guidance on setting up GitHub Desktop and
  structuring the repository (`.gitignore` to exclude raw data files) to
  serve as the verifiable progress log recommended in the module leader's
  email.
- **Overleaf/LaTeX setup:** Assistance building the document skeleton to
  match the official module guidance document's formatting and structural
  requirements.
