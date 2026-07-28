# AI at Work — Field Guide

**A practical, bilingual Q&A handbook on AI, LLMs, RAG, classic ML, data pipelines, MLOps, cost, automation tooling and the current model landscape.**

*Gyakorlati, kétnyelvű kérdés-válasz kézikönyv az AI-ról, LLM-ekről, RAG-ról, klasszikus ML-ről, adatpipeline-okról, MLOps-ról, költségekről, automatizálási eszközökről és a jelenlegi modell-tájképről.*

---

## What's here / Mi van itt

| File | Language | Contents |
|---|---|---|
| [`AI-at-Work-Field-Guide-EN.md`](AI-at-Work-Field-Guide-EN.md) | English | ~100 questions across 12 sections, with full explanations |
| [`AI-a-munkahelyen-Kisokos-HU.md`](AI-a-munkahelyen-Kisokos-HU.md) | Magyar | The same guide, written in Hungarian rather than translated |
| [`ai-llm-rag-faq.html`](ai-llm-rag-faq.html) | EN + HU | Self-contained interactive page: 60 cards, filterable by topic, with an EN/HU toggle. Open it directly in a browser — no build step, no dependencies. |

## Who it's for / Kinek szól

**EN** — Anyone who has to build with, decide about, or explain AI systems at work: engineers, analysts,
product and project people, team leads. No maths required. Every answer is structured as
*mechanism → trade-off → what to do about it in practice*.

**HU** — Mindenkinek, akinek AI-rendszerekkel kell dolgoznia, döntenie róluk, vagy elmagyaráznia őket:
fejlesztőknek, elemzőknek, termék- és projektes szerepeknek, csapatvezetőknek. Matematika nem kell hozzá.
Minden válasz szerkezete *mechanizmus → kompromisszum → mit tegyünk vele a gyakorlatban*.

## Sections / Szekciók

1. **Foundations** — how LLMs work, tokens, temperature, context windows, hallucination, grounding
2. **ML vs GenAI** — bias–variance, dimensionality reduction, clustering, deep learning, transfer learning, multi-armed bandits, embeddings
3. **RAG & context** — pipelines, chunking, the two failure modes, context engineering
4. **Tools, agents & MCP** — tool calling, agent vs workflow, skills, memory, MCP
5. **Data & pipelines** — ETL vs ELT, dbt, feature stores, batch vs streaming, data quality
6. **MLOps & LLMOps** — versioning, drift, monitoring, safe rollout, retraining, incident response
7. **Evaluation** — golden sets, LLM-as-judge, precision/recall, non-deterministic evaluation
8. **Orchestration, routing & cost** — model routing, cost estimation, the levers that matter
9. **Automation tooling** — n8n, Make, Zapier, Power Automate, Flowise, Airflow, Dagster, Temporal
10. **Models, CLIs & IDEs** — Claude, GPT, Gemini, Grok, Llama, Mistral, the Chinese open-weight families, Ollama, agentic CLIs, Antigravity, OpenClaw
11. **Security & governance** — prompt injection, AI usage policy, human-in-the-loop, redlines, rollout
12. **Cheat sheet** — the twelve sentences worth knowing verbatim

## A note on freshness / Megjegyzés az aktualitásról

**EN** — Section 10 (models, CLIs, IDEs) is a snapshot of **mid-2026** and is the fastest-moving part of
this guide. Names, versions, capabilities and especially pricing change monthly. Use it to understand the
*shape* of the landscape and the criteria for choosing; verify specifics before committing to a vendor.
Everything else — the mechanisms, trade-offs and practices — changes far more slowly.

**HU** — A 10. szekció (modellek, CLI-k, IDE-k) **2026 közepének** pillanatképe, és a kisokos leggyorsabban
változó része. A nevek, verziók, képességek és különösen az árazás havonta változnak. Használd a tájkép
*alakjának* és a választási kritériumoknak a megértésére; a konkrétumokat ellenőrizd, mielőtt szállító
mellett elköteleződsz. Minden más — a mechanizmusok, kompromisszumok és gyakorlatok — sokkal lassabban
változik.

## Licence

Text and code in this repository are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) —
use it, adapt it, share it, with attribution.
