# AI at Work — Field Guide

**100 questions on AI, LLMs, RAG, classic ML, data pipelines, fine-tuning, inference, evaluation, MLOps, cost, automation and governance — answered for people who have to make decisions, not pass an exam.**

> Interactive version: the **AI-LLM-RAG FAQ** page in this repository (`ai-llm-rag-faq.html`) — open it in
> any browser.
>
> **Who this is for.** Anyone who has to build with, decide about, budget for, or explain AI systems at
> work: engineers, analysts, data people, product and project roles, team leads. No mathematics required.
>
> **How every answer is structured.** *Mechanism* — what is actually happening. *Trade-off* — what you
> give up and get. *Practice* — what to do about it on Monday morning. Where a question has a common wrong
> answer, it is named explicitly, because knowing the failure mode is usually more valuable than knowing
> the definition.
>
> **On tool coverage.** Wherever a tool is named, credible alternatives are named alongside it, with the
> criteria that actually separate them. No tool in this guide is a recommendation by default; the point is
> to give you the decision axes.
>
> **On freshness.** Mechanisms, trade-offs and practices change slowly. Tool names, versions, capabilities
> and especially pricing change monthly. Anything version- or price-specific reflects **mid-2026** and
> carries a marker. Re-verify before you commit budget or sign a contract.

---

## Contents

| # | Section | Questions |
|---|---|---|
| 1 | [Foundations — how LLMs actually work](#1-foundations) | 1–8 |
| 2 | [Prompting and structured output](#2-prompting-and-structured-output) | 9–14 |
| 3 | [ML vs GenAI — and the classic ML toolbox](#3-ml-vs-genai) | 15–22 |
| 4 | [RAG and retrieval](#4-rag-and-retrieval) | 23–31 |
| 5 | [Context and memory](#5-context-and-memory) | 32–35 |
| 6 | [Agents, tools and protocols](#6-agents-tools-and-protocols) | 36–42 |
| 7 | [Orchestration, frameworks and gateways](#7-orchestration-frameworks-and-gateways) | 43–47 |
| 8 | [Data and pipelines](#8-data-and-pipelines) | 48–54 |
| 9 | [Fine-tuning and model customisation](#9-fine-tuning-and-model-customisation) | 55–59 |
| 10 | [Inference, serving and local models](#10-inference-serving-and-local-models) | 60–65 |
| 11 | [Evaluation](#11-evaluation) | 66–72 |
| 12 | [Observability, MLOps and LLMOps](#12-observability-mlops-and-llmops) | 73–78 |
| 13 | [Cost and FinOps for AI](#13-cost-and-finops-for-ai) | 79–83 |
| 14 | [Automation tooling](#14-automation-tooling) | 84–89 |
| 15 | [The model and tool landscape](#15-the-model-and-tool-landscape) | 90–94 |
| 16 | [Security and safety](#16-security-and-safety) | 95–98 |
| 17 | [Governance, legal and organisation](#17-governance-legal-and-organisation) | 99–100 |
| — | [Product and UX patterns](#18-product-and-ux-patterns) | appendix |
| — | [Cheat sheet](#cheat-sheet) | — |

---

## 1. Foundations

### Q1 — How does a large language model actually work?

**Mechanism.** A large language model is a neural network — in practice always a transformer decoder —
trained on a single task: given the text so far, predict the next **token**. Text enters, a tokeniser
splits it into sub-word chunks, each chunk becomes a vector (an *embedding*), and those vectors pass
through dozens to hundreds of stacked layers. Each layer does two things: **attention**, which lets every
position pull information from other positions, and a **feedforward network**, which transforms each
position independently and holds much of the model's stored knowledge. The final layer produces a
probability distribution over the entire vocabulary — typically 100,000 to 200,000 possible tokens. The
system samples one token from that distribution, appends it to the input, and runs the whole forward pass
again. That loop is called *autoregressive generation*, and it is the entire mechanism.

Nothing else is happening. There is no database lookup, no retrieval step, no verification pass, no
intention, no plan held in reserve. Everything the model "knows" is statistical structure compressed into
billions of floating-point weights during training — knowledge and capability stored as geometry rather
than as facts in rows.

**Trade-off.** This architecture buys extraordinary generality: the same model handles translation,
summarisation, code, classification and conversation, because all of them can be framed as *what text
comes next*. What you give up is any guarantee of correctness. The model is optimised for producing text
that is a *likely continuation*, and likelihood and truth are correlated but not the same thing. It is
also why the model cannot tell you how confident it is in any way you can use: the probability it assigns
to a token measures linguistic plausibility, not factual reliability.

**Practice.** Three consequences shape every system you build on top of one. Truth has to come from
outside the model — retrieval, tool calls, database queries — and be injected into the prompt or verified
after generation. Cost and latency scale with the number of tokens, because generation is a loop, which
makes brevity an engineering requirement rather than a style preference. And the correct mental model to
give non-technical colleagues is *a very well-read improviser*, not *a search engine with better manners*.
That single reframing prevents most unrealistic expectations before they form.

### Q2 — What is next-token prediction, and why does it produce something that looks like reasoning?

**Mechanism.** Next-token prediction is the training objective, not a description of the model's
ambitions. During training the model sees a fragment of real text, predicts a probability distribution
over what comes next, is shown the actual next token, and has its weights nudged — via backpropagation on
a cross-entropy loss — to make that actual token more likely. This repeats trillions of times over an
enormous corpus.

The reason this deceptively trivial game produces broad competence is that **you cannot predict text well
without modelling what the text is about**. To complete "the capital of Hungary is…" you need a fact. To
complete a mathematical proof you need the shape of valid argument. To complete a function body you need
the language's semantics and the surrounding code's intent. To complete a dialogue convincingly you need a
model of what the speakers want. Grammar, world knowledge, style, arithmetic and multi-step reasoning are
all absorbed as *instrumental* skills in service of the prediction task. Nobody taught the model to
reason; reasoning turned out to be useful for guessing well.

**Trade-off.** Because the objective rewards plausibility, the model produces its most plausible
continuation whether or not it is correct — which is exactly why it sounds equally confident when right
and when wrong. There is no separate internal signal for "this part I know, this part I'm improvising".
Later training stages (see Q7) partially teach the model to hedge, but they teach it to *say* uncertain
things in uncertain situations, which is a learned behaviour and not a calibrated measurement.

**Practice.** Two operational consequences. First, treat confident tone as carrying zero information
about correctness: your system's design must not rely on the model sounding unsure when it is unsure,
because it frequently doesn't. Second, output is produced one token at a time, sequentially, which is why
streaming exists (the first token can be shown long before the last) and why output length is the single
most controllable cost lever you have. A prompt that says "answer in at most three sentences" is a budget
decision.

### Q3 — What are tokens, and why do they appear on every invoice?

**Mechanism.** Tokens are the units the model actually reads and writes: sub-word fragments produced by a
tokeniser before the model sees anything. Common English words are usually one token; rarer words split
into pieces; whitespace and punctuation attach in slightly unintuitive ways. As a rule of thumb, one token
is about 0.75 English words, or roughly four characters.

That ratio is *not* universal, and this matters commercially. Text in languages under-represented in the
tokeniser's training data fragments into more pieces per word. Hungarian, with its agglutinative
morphology and accented characters, typically costs meaningfully more tokens per word than the English
equivalent — often 1.5× to 2× depending on the tokeniser. Code, JSON, long numbers, and base64 blobs are
similarly token-expensive. Different model families use different tokenisers, so token counts are not
portable between vendors, and even between generations of the same vendor.

**Trade-off.** Tokenisation is what makes models able to handle arbitrary text — including words they have
never seen — at a manageable vocabulary size. The cost is that the model has no direct access to
characters. This is the root cause of a whole family of otherwise baffling failures: counting letters,
reversing strings, spelling puzzles, rhyme schemes, and arithmetic on long numbers. The model sees
`straw` + `berry`, not eleven letters.

**Practice.** Tokens are simultaneously the unit of **price** (billed per input and output token), of
**latency** (which scales with count), of **context** (the window is measured in tokens), and of **rate
limits** (usually tokens per minute). So your unit economics are a token budget, and the biggest recurring
waste in most deployments is a bloated system prompt paid for on every single call. Measure real token
counts with the provider's tokeniser or counting endpoint rather than estimating from word counts,
especially for non-English text — a 60% estimation error compounds badly at scale. And when a task depends
on character-level structure, do it in code, not in the model.

### Q4 — What do temperature, top-p and top-k actually control?

**Mechanism.** Before choosing a token, the model has a raw score — a *logit* — for every token in its
vocabulary. Three knobs shape how that score set becomes a choice.

**Temperature** divides the logits before they are normalised into probabilities. Near zero, the
distribution sharpens until the highest-scoring token wins essentially always: near-deterministic, "greedy"
output. As temperature rises, the distribution flattens, and lower-probability tokens start getting picked
— which is where variety, surprise, drift and incoherence all come from, indistinguishably.

**Top-k** truncates: only the k highest-probability tokens are eligible. **Top-p** (nucleus sampling)
truncates adaptively: take tokens in descending probability until their cumulative mass reaches p, and
sample only from those. Top-p is generally preferred because it adapts to how peaked the distribution
already is — when the model is confident, the nucleus is tiny; when it is genuinely uncertain, the nucleus
widens.

**Trade-off.** The pervasive misunderstanding is treating temperature as an accuracy dial. It is not. Low
temperature does not make the model more correct; it makes it more **consistent**. If the model does not
know something, temperature 0 means it will produce the same wrong answer reliably every time — which can
actually be worse than variability, because consistency reads as confidence to users and to your own
monitoring.

**Practice.** Extraction, classification, routing, code, JSON, and anything you evaluate: temperature
0–0.3, and pin it, so that when a metric moves you know it was your change and not the sampler.
Brainstorming, copywriting variants, creative drafting: 0.7–1.0. Do not tune temperature and top-p
simultaneously without a reason; pick one and hold the other at its default, or you cannot attribute
effects. Note also that some recent reasoning-oriented models no longer accept sampling parameters at all,
having replaced them with an "effort" or "thinking depth" setting — if your code sends temperature to such
a model it may be rejected outright, which is a real migration hazard when switching model generations.

### Q5 — What is the context window, and what really happens when you exceed it?

**Mechanism.** The context window is the maximum number of tokens the model can attend to in a single
call. Critically, it covers *everything at once*: the system prompt, the full conversation so far, every
document you injected, every tool definition, every tool result, and the answer currently being generated.
It is a budget you spend, not storage you own.

Exceeding it does not degrade gracefully. Either the API rejects the request outright, or — far more
commonly and far more insidiously — your framework silently truncates or summarises to make it fit. The
default truncation strategy is almost always to drop the oldest messages, which is exactly where the
original instructions and earliest established facts live. Users experience this as an assistant that
mysteriously "forgets" the rules halfway through a long session, and it will not appear in your error
logs, because nothing errored.

**Trade-off.** Bigger windows are genuinely useful — a million tokens changes what is architecturally
possible — but "more context" and "better output" are not the same curve. Attention degrades over
distance: models reliably under-use material buried in the middle of a long context, a robust finding
usually called *lost in the middle*. Cost and latency rise with input size, and for attention specifically
the compute cost grows quadratically with sequence length. And a long context full of marginally relevant
material actively harms output by diluting the signal the model should be attending to.

**Practice.** Budget the window explicitly per slot: policy, tools, task state, retrieved evidence, recent
turns, compressed history. Retrieve passages rather than pasting whole documents. Keep a rolling summary
instead of the full transcript. Put the most important material at the beginning *and* restate the actual
instruction at the end, since the middle is the weak zone. Store durable facts in a real database rather
than hoping they survive in the transcript. And instrument it: log the token count per slot per request,
because "we're at 85% of the window on a normal request" is a fact you want to discover from a dashboard
rather than from a user complaint.

### Q6 — Explain transformers and attention without the mathematics

**Mechanism.** The transformer takes a sequence of token embeddings, adds positional information (modern
models typically use rotary position embeddings), and pushes the sequence through N identical blocks. Each
block contains attention plus a feedforward network, each wrapped in a residual connection and a
normalisation step — those wrappers are unglamorous but they are what makes stacking a hundred layers
trainable at all. At the end, a linear projection plus a softmax turns the final vectors into next-token
probabilities.

**Attention** itself is a matching operation with a useful metaphor: a library request. Each position
emits three vectors. A **query** ("what am I looking for?"), a **key** ("what do I have to offer?"), and a
**value** ("what I hand over if you pick me"). Every query is compared against every key to produce
relevance scores; the scores are normalised into weights; the weighted average of the values is what that
position takes away. This is how the model works out that "bank" means riverbank when "boat" appears
nearby — the boat token's key matched the bank token's query, so its value flowed in. **Multi-head**
attention means this happens in several independent subspaces simultaneously, each free to specialise:
one head tracking syntax, another coreference, another topic.

**Trade-off.** The transformer's decisive advantage over its predecessors (recurrent networks) is that the
whole sequence is processed in parallel rather than one step at a time, which is what made GPU-scale
training possible and therefore what made LLMs exist at all. The cost is that attention compares every
position to every other position, so compute grows **quadratically** with sequence length. That single
fact explains a surprising amount of the industry: why long context is expensive, why optimisations like
FlashAttention, grouped-query attention and sliding-window attention exist, why the KV cache exists (so
generation does not recompute earlier keys and values on every new token), and why sparse-attention
architectures are an active competitive frontier for long-context models.

**Practice.** You do not need this to use an API, but you need it for three conversations. When someone
asks why long context costs disproportionately more, the answer is quadratic attention plus KV-cache
memory. When someone asks why the model missed something in the middle of a long document, the answer is
attention dilution, and the fix is retrieval rather than a bigger window. And when a vendor advertises a
new architecture as "efficient long context", the claim to interrogate is specifically what they gave up
in the attention pattern to get it.

### Q7 — What is the difference between pre-training and post-training, and why does it matter to you?

**Mechanism.** **Pre-training** is the enormous, general phase: next-token prediction over trillions of
tokens of text and code. It costs millions of dollars, takes weeks to months, occupies thousands of
accelerators, and produces a model with broad knowledge and fluent language ability but no notion
whatsoever of being helpful. A purely pre-trained model, handed a question, is as likely to continue with
more questions as to answer, because that is what documents do.

**Post-training** turns that raw capability into a product. It typically layers supervised fine-tuning on
curated instruction/response pairs, then preference optimisation — RLHF, DPO, GRPO and relatives — where
the model learns from comparisons between candidate responses rather than from single correct answers.
This is where tone, refusal behaviour, formatting habits, tool-calling conventions, reasoning style and
"helpfulness" come from. It costs orders of magnitude less than pre-training, and it is where most of the
behaviour you actually notice originates.

**Trade-off.** The split creates a clean and very useful diagnostic boundary, but it also means the model
you use is shaped by value judgements made by the vendor during post-training — how cautious to be, how
verbose, when to refuse. Those judgements are not visible to you, they change between versions, and they
are the usual reason a model upgrade "feels different" in ways the benchmark scores do not capture.

**Practice.** Use the boundary to triage complaints, because it tells you who can fix what. Verbosity,
over-refusal, sycophancy, ignoring your JSON format, weak instruction-following: post-training behaviour,
therefore partly steerable by you through prompting and fully changeable only by the vendor. Missing
knowledge about your company, your customers, your product or last month's events: pre-training, and no
amount of prompt wording will conjure it — that is a retrieval or tool problem. This distinction saves
enormous amounts of wasted effort. Teams routinely spend weeks rewriting prompts to fix what is actually a
missing-information problem, and occasionally spend money on fine-tuning to fix what is actually a
formatting instruction.

### Q8 — What are model weights, when do they change, and why does everyone get this wrong?

**Mechanism.** Weights are the billions of learned numerical parameters that define the model's function —
the compressed residue of everything training exposed it to. They change **only during training**:
pre-training, fine-tuning, continued pre-training, or a preference-optimisation run. They do **not**
change when you prompt the model, attach retrieval, hand it tools, give it memory, or hold a
ten-thousand-turn conversation. Inference is a read-only forward pass.

**Trade-off.** This is the single most common misconception in workplace AI conversations, and it cuts both
ways. The reassuring side: a single bad interaction cannot corrupt the model, one user cannot poison it for
everyone, and your data does not leak into the weights by being used in a prompt (whether it is *logged*
or used for future training is a contractual question about the vendor, which is a completely different
risk and one you should check separately). The uncomfortable side: improvement is not automatic. "The model
will learn how we work" is false unless someone builds and owns a data flywheel — capturing examples,
curating them, and running an explicit training job.

**Practice.** Everything a model appears to "remember" at runtime is either still inside the context
window or stored in a database that you are responsible for maintaining, securing and deleting from.
Design accordingly: if a fact must persist, write it somewhere; if it must be forgettable, know where it
lives so you can delete it. When someone proposes that the system will get smarter with use, ask the
concrete question — who owns the loop, where do the labels come from, what triggers a training run, and
what gates it before deployment? If there is no answer, the system will be exactly as good in a year as it
is today, and that is a legitimate plan, just not the one being described.

---

## 2. Prompting and structured output

### Q9 — Which prompting techniques actually earn their keep, and which are cargo cult?

**Mechanism.** A prompt is the model's entire world at inference time, so prompting is really the practice
of controlling what the model conditions on. A handful of techniques have robust, repeatable effects.

**Clear task specification** — stating the role, the goal, the constraints and the output format
explicitly — is unglamorous and consistently the largest single win. **Few-shot examples** (showing two to
five worked cases) are extremely effective for format, tone and edge-case handling; they work by
demonstrating rather than describing. **Explicit reasoning** — asking the model to work through steps
before answering — helps genuinely multi-step problems, though on current reasoning-capable models it is
often built in and asking for it again can be redundant or even counterproductive. **Decomposition**,
splitting one hard prompt into two or three simpler calls with validated hand-offs, usually beats a single
elaborate prompt. **Structured output with a schema** (Q10) removes an entire class of parsing failure.
**Negative examples and explicit non-goals** — "do not include X" — work but less reliably than positive
demonstrations, so prefer showing what you want.

What earns much less than its reputation: elaborate persona framing ("you are a world-renowned expert
with 30 years of experience"), emotional pressure ("this is very important to my career"), threats or
bribes, ALL-CAPS emphasis stacked on every instruction, and long lists of prohibitions. These sometimes
moved the needle on older, weaker models. On current ones they mostly consume tokens, and stacked emphasis
actively backfires by making everything equally urgent — which means nothing is.

**Trade-off.** Every technique costs tokens and therefore money and latency on every call. A five-example
few-shot block might be worth it or might be replaceable by one clear sentence; the only way to know is to
measure. Prompting is also the *least durable* layer: a prompt tuned hard against one model version often
degrades on the next, because instruction-following behaviour shifts between generations.

**Practice.** Start with the simplest prompt that could work and add only what measurement justifies. Keep
prompts in version control as files, not as strings in a database, so changes are diffable and reviewable.
Attach a small eval set to every prompt you care about (Q66) and run it on every change — otherwise you are
guessing, and the guesses feel like knowledge. And when you inherit a prompt full of capitals and threats,
try deleting them before adding anything: on a modern model, the shorter version frequently scores better.

### Q10 — How do you get reliably structured output, and why isn't "return JSON" enough?

**Mechanism.** Asking a model in prose to "respond with JSON" gets you JSON *most* of the time, and
therefore breaks in production at a rate somewhere between 0.5% and 5% — trailing commentary, a markdown
code fence, a hallucinated field, a missing required key, a string where a number belongs. At scale, a 1%
malformed rate is a constant background incident.

There are three progressively stronger mechanisms. **Prompt-only** (describe the shape, show an example)
is the weakest. **Constrained decoding / structured output modes**, offered by most major providers, take
a JSON Schema and constrain the sampler at generation time so that malformed output is arithmetically
impossible — the model can only emit tokens that keep the output valid against the schema. **Tool/function
calling with a strict schema** achieves the same by framing the extraction as a call whose arguments must
validate. Open-source equivalents exist for self-hosted models, working by masking logits against a grammar
(the technique underlying libraries such as Outlines, Guidance, LM Format Enforcer and llama.cpp's GBNF
grammars).

**Trade-off.** Constrained decoding removes syntactic failure but not semantic failure: you are guaranteed
a valid `{"risk_score": 0.8}`, not a *correct* 0.8. There is also a real quality interaction — very rigid
schemas can degrade reasoning quality, because the model is forced to commit to a field before it has
"thought". The standard mitigation is to include a free-text reasoning field *before* the structured fields
in the schema, so the model can work before it decides. Finally, schema support varies: complex constructs
like recursion, conditionals and numeric bounds are often unsupported, and unsupported constraints are
sometimes silently dropped rather than rejected.

**Practice.** Use enums rather than free strings wherever the valid values are known — this single change
eliminates most invented values. Keep required fields minimal. Validate on receipt anyway, in your own
code, and treat validation failure as a first-class handled path with a bounded retry rather than an
exception that reaches the user. Put reasoning before conclusions in the schema. And for anything
high-volume, log the validation failure rate as a monitored metric: it is one of the cleanest early
indicators that a model version changed under you.

### Q11 — Does a longer, sterner prompt fix hallucination?

**Mechanism.** No, and understanding why is more useful than the answer. "Do not make anything up" adds no
*information* the model can act on. Hallucination happens when the required fact is unavailable — not in
the weights, not in the context — or when retrieval supplied the wrong material. An instruction cannot
conjure a missing fact. What it can do is change the model's *disposition*, making it more likely to hedge
or refuse — which sometimes looks like improvement while actually costing you correct answers on questions
the model could have answered.

Length has direct costs too: more tokens on every call, higher latency, more instructions available to
conflict with each other, and diluted attention across a longer prompt. A 2,000-token system prompt of
accumulated prohibitions is a common and expensive anti-pattern, usually built by successive people adding
one more rule after one more incident.

**Trade-off.** There *is* a narrow class of prompt change that genuinely reduces hallucination: not
sternness but **structure**. Giving the model an explicit escape hatch ("if the provided context does not
contain the answer, return `not_found`") measurably increases appropriate abstention, because it makes "I
don't know" a *legal move* with a defined output shape rather than a failure of helpfulness. Requiring
quotes or citations from supplied context has a similar effect: it is much harder to fabricate while also
producing a verbatim supporting quote.

**Practice.** Fix the information architecture, not the volume of instruction. Supply the fact via
retrieval or a tool. Constrain the output shape. Provide the abstention path and make sure your UI handles
it gracefully — an ugly "not found" screen guarantees someone will remove the abstention. Require
citations for factual claims and verify them programmatically where you can (does the cited document
exist, does the quoted string actually appear in it, does the number match the source). And when you
inherit a giant defensive prompt, the highest-value experiment is usually deletion: cut it to a third,
measure, and you will often find the score unchanged or better.

### Q12 — How do you version, test and maintain prompts like real code?

**Mechanism.** Prompts are executable artefacts: they determine system behaviour, they break, and they
need change control. Treating them as configuration strings in a database with no history is a recurring
source of "it worked last week and nobody changed anything" — because somebody did.

A working setup has five parts. Prompts live as **files in the repository**, so changes are diffable and
reviewable in a pull request. Each prompt carries a **version identifier that is logged with every
request**, so any production output can be attributed to an exact prompt version. Each prompt has an
attached **eval set** that runs in CI on change (Q66). Changes go through the same **staged rollout** as
code (Q77). And prompt content is **separated from prompt assembly** — the template is a file, the runtime
decision about what to put in it is code, and conflating them makes both untestable.

**Trade-off.** File-based prompts in git give you rigour but slow down iteration for non-engineers, and
prompt iteration is often best done by the domain expert rather than the developer. Dedicated prompt
management platforms exist to bridge this — Langfuse, PromptLayer, Agenta, Latitude and the prompt
features inside LangSmith and Braintrust all offer a UI for editing with versioning and rollback, decoupled
from deployment. The cost is a second source of truth and a runtime dependency in your critical path;
teams that adopt one should be deliberate about caching and about what happens when it is unreachable.

**Practice.** Whichever you choose, the non-negotiables are the same: every request logs which prompt
version produced it; every change is attributable to a person and a reason; every prompt that matters has
an eval set; and rollback is one action, not an archaeology project. Also keep a *frozen* copy of the exact
prompt text alongside evaluation results, because a metric that moved because the prompt changed and a
metric that moved because the eval set changed look identical in a dashboard and have completely different
fixes.

### Q13 — Why do two identical prompts give different answers?

**Mechanism.** Several independent causes, worth separating because they have different fixes.
**Sampling**: with temperature above zero the model is drawing from a distribution by design.
**Floating-point non-determinism**: GPU kernel scheduling, batching and the order of numerical reductions
make arithmetic non-associative, so even at temperature 0 outputs are not bit-identical run to run — and
notably, results can vary with *how many other requests were batched alongside yours*. **Serving-side
variation**: mixture-of-experts routing, speculative decoding, quantised replicas, and heterogeneous
hardware in the provider's fleet. **Silent version drift**: the provider updated the model behind an alias
you treated as stable. **Hidden context differences**: an injected timestamp, retrieved documents that
changed, a memory record, a cache hit versus a miss.

**Trade-off.** Full determinism is not achievable through a hosted API, and pursuing it is usually the
wrong goal. What you actually need is *reproducible enough to attribute change* — the ability to say "this
metric moved because of my change" — and that is achievable with pinning and averaging even without
bit-identity.

**Practice.** Pin explicit model versions rather than floating aliases in anything you measure or ship;
this is the highest-value single action, because silent vendor updates are the cause most likely to look
like your own regression. Fix temperature for evaluation runs. Never assert exact string equality in
automated tests — assert on structure, schema validity, and presence of required facts. Run multiple
samples per test case and report a pass rate with variance rather than a single verdict (Q70). Log enough
context to reconstruct any request, including the resolved model version the provider actually served. And
do not promise users reproducibility you cannot deliver: if a report needs to be regenerable identically,
store the generated artefact rather than regenerating it on demand.

### Q14 — What is grounding, and how do you measure it?

**Mechanism.** Grounding means every claim in the output is traceable to a source you supplied — a
retrieved passage, a tool result, a database row — rather than to the model's parametric memory. The
techniques are retrieval, mandatory citations, quote-then-answer patterns, structured output carrying a
source identifier per field, and refusal when the supplied context is insufficient.

Measurement has three distinct components, and collapsing them into one number is a common mistake.
**Groundedness (faithfulness)**: decompose the answer into atomic claims and check each against the
supplied context. Human-labelled on a golden set; model-judged at scale, with the judge calibrated against
the humans. **Citation precision and coverage**: are the cited sources actually the right ones, and what
share of claims carry a citation at all? **Abstention correctness**: did the system say "not found"
exactly when it should have — measured in both directions, since over-abstention is also a failure.

**Trade-off.** A subtlety worth internalising: *faithful is not the same as correct*. An answer can be
perfectly grounded in a document that is wrong, superseded, or three years out of date. Groundedness
measures fidelity to your corpus, and therefore inherits your corpus's quality. This is why source quality
and freshness are metrics in their own right, and why an internal assistant is only as trustworthy as the
worst document someone forgot to delete from the wiki.

**Practice.** Measure groundedness with a *fixed, known-good context* — ideally oracle documents — so that
retrieval failures cannot contaminate the score (Q72 covers why this separation matters). Report it
separately from retrieval quality. Add an unsupported-claim rate per response as your headline
hallucination metric, because it is the one that maps to user harm. Verify citations mechanically where you
can. And put freshness in the interface: "as of 14:32 today" and a visible source link prevent an entire
category of trust incident that no amount of model tuning will.

---

## 3. ML vs GenAI

### Q15 — What is the actual difference between machine learning and generative AI?

**Mechanism.** **Machine learning** is the broad discipline: algorithms that learn patterns from data
instead of being explicitly programmed. Most applied ML is *discriminative* — it maps inputs to a decision.
Will this customer churn? Is this transaction fraudulent? What will demand be next month? Which of five
categories does this ticket belong to? The output is a number, a class, or a probability.

**Generative AI** is the subset that models the data distribution well enough to draw new samples from it:
text, images, audio, video, code. Large language models are the text-and-code instance.

Four practical differences follow, and they matter more than the taxonomy. Discriminative ML produces
**calibrated** outputs — a 0.7 means something you can threshold against business cost — while generative
models produce fluent output with no reliable confidence attached. ML is evaluated with well-understood
metrics (AUC, precision/recall, RMSE, MAE) while generative output usually needs rubrics plus human or
model judgement. ML inference is cheap and fast — microseconds, effectively free per prediction — while
generation is orders of magnitude more expensive. And ML is auditable in the way a risk or compliance
function requires, whereas a generated paragraph is not.

**Trade-off.** The reason GenAI dominates attention despite these disadvantages is that it needs no
training data from you. A classic ML project starts with "we need 50,000 labelled examples"; a GenAI
project starts with an API key. That collapses time-to-first-result from months to hours, which is
genuinely transformative for exploration — and genuinely misleading about total cost, because the labelling
work you skipped reappears later as evaluation work you cannot skip.

**Practice.** The correct mental model is not that GenAI replaced ML. It is: **generative models where the
input or output is unstructured language or media; classic ML where the task is prediction over structured
data**. Most valuable production systems use both, wired together (Q22). When someone proposes an LLM for a
scoring or forecasting problem, the question to ask is whether they need a calibrated number that a risk
function will sign off on — if yes, that is an ML problem with possibly an LLM at the edges.

### Q16 — Supervised, unsupervised, self-supervised, reinforcement — what's the practical difference?

**Mechanism.** **Supervised learning** trains on labelled examples: input plus correct answer.
Classification and regression live here, and it is the workhorse of applied ML. Its bottleneck is almost
never the algorithm — it is labels: getting them, keeping them consistent between annotators, and keeping
them current as the world changes.

**Unsupervised learning** has no labels and finds structure: clustering, dimensionality reduction, anomaly
detection, topic discovery. It is exploratory by nature, which makes evaluation genuinely hard — there is
no ground truth to be right about, so you judge results by whether they are stable and useful.

**Self-supervised learning** manufactures labels from the data itself: hide a word and predict it, mask a
patch of image and reconstruct it, predict whether two crops came from the same photo. This is the trick
that made foundation models possible, because it turns unlabelled internet-scale data into a supervised
problem at zero labelling cost.

**Reinforcement learning** learns from a reward signal rather than correct answers: take an action, observe
the consequence, adjust. It powers control and sequential-decision problems, and in the LLM world it
appears as preference optimisation, where the "reward" is a human or model judgement of which of two
candidate responses is better.

**Trade-off.** The paradigms trade labelling cost against directness of supervision. Supervised learning
gives the cleanest signal at the highest data cost. Self-supervision gives an enormous amount of weak
signal for free but requires scale to be useful. RL needs no labels but needs a reward you can actually
compute, and specifying a reward that captures what you want without being gameable is famously hard — the
same problem appears in AI systems as Goodharting an evaluation metric.

**Practice.** Ask "what is the cheapest supervision that would solve this?" If you can frame the task
self-supervised, or use a pre-trained model that already did, you skip the bill that usually dominates a
supervised project. If you must label, invest in the annotation guideline before the annotators: inter-
annotator disagreement is the ceiling on your achievable model quality, and it is measurable before you
train anything. Tooling for this is mature and worth using — Argilla, Label Studio and Prodigy for
open/self-hosted setups, with managed labelling vendors for volume.

### Q17 — What is the bias–variance trade-off, and why does it explain most disappointing models?

**Mechanism.** Every model's error decomposes into three parts. **Bias** is error from the model being too
simple to represent the real pattern — it is wrong in a systematic direction. **Variance** is error from
the model being too sensitive to the particular sample it saw — change the data slightly and it learns
something different. **Irreducible noise** is the part nobody can remove, and knowing roughly how large it
is prevents chasing impossible targets.

High bias is *underfitting*: poor on training data and poor on new data alike. High variance is
*overfitting*: near-perfect on training data, disappointing in production. The trade-off is that the usual
moves — more capacity, more features, longer training — reduce bias while increasing variance, and
regularisation, simplification and more data push the other way.

**Trade-off.** The honest framing is that this is a *diagnostic* framework, not a dial you set. Modern
practice has complicated the textbook picture — very large over-parameterised models can achieve low bias
and low variance simultaneously in ways the classical curve did not predict — but the diagnostic remains
exactly as useful, because it tells you which direction to move.

**Practice.** Diagnose by comparing training error with validation error. Both high → a bias problem, so
add capacity, better features, or a more expressive model family. Training low and validation much higher →
a variance problem, so get more or more representative data, regularise, simplify, or ensemble.

This is the honest explanation for the most common disappointing outcome in applied ML: *"it looked great
in the notebook and then didn't work"*. That is nearly always variance, frequently made worse by
**leakage** — information in the training data that will not exist at prediction time, such as a feature
computed after the outcome, or a duplicate row appearing in both train and test. Before blaming the
algorithm, verify the split is honest: split by time when predicting the future, and split by entity
(customer, patient, document) when the same entity appears many times. A random row-level split on grouped
data manufactures optimism reliably.

### Q18 — What is dimensionality reduction, and when do you actually need it?

**Mechanism.** Real datasets often have far more columns than you can reason about, many of them
correlated or mostly empty. Dimensionality reduction compresses many features into fewer while preserving
as much useful structure as possible.

**PCA** (principal component analysis) is the workhorse: it finds the orthogonal directions along which
the data varies most and re-expresses points in those terms. Linear, fast, deterministic and invertible,
which makes it a solid default for compression and denoising. **t-SNE** and **UMAP** are non-linear and
built for *visualising* high-dimensional data in two dimensions; UMAP is generally preferred now for
preserving more global structure and running faster at scale. **Autoencoders** learn a compressed
representation with a neural network, useful when structure is genuinely non-linear and data is plentiful.
**Matrix factorisation** methods (SVD, NMF) remain standard in recommendation and topic modelling.

**Trade-off.** Two cautions that get ignored. First, components are combinations of original features, so
you trade interpretability for compactness — usually the wrong trade if a regulator or a customer will ask
why a decision was made. Second, and specific to t-SNE and UMAP: their axes are meaningless, and
inter-cluster distances are not faithful. A plot showing two clusters far apart tells you almost nothing
quantitative. Reading conclusions off a UMAP plot is one of the most common analytical errors in applied
data science, and it is very persuasive because the pictures are beautiful.

**Practice.** Reach for it when you diagnosed a specific problem: distance becoming uninformative in high
dimensions, training or inference too slow, storage cost, severe multicollinearity, or exploratory
visualisation. Do *not* apply PCA reflexively as preprocessing — with modern gradient-boosted trees, which
handle wide correlated tabular data natively, it frequently adds nothing and costs interpretability. In
text and embedding work, UMAP followed by density clustering is a genuinely strong pipeline (Q19), and
there the point is discovery rather than modelling, which is the use case it is honest for.

### Q19 — What is clustering, and how do you know a clustering is any good?

**Mechanism.** Clustering groups records so that members of a group resemble each other more than members
of other groups, with no labels to guide it. Typical business uses: customer segmentation, grouping support
tickets or product feedback into themes, near-duplicate detection, and anomaly detection by way of
"belongs to no cluster".

The main families behave differently in ways that matter. **k-means** is fast and scales well, but requires
choosing *k* in advance, assumes roughly spherical similarly-sized groups, and is sensitive to feature
scaling and outliers. **Hierarchical clustering** produces a dendrogram you can cut at any level — very
interpretable, expensive on large data. **DBSCAN / HDBSCAN** find arbitrarily-shaped clusters and
explicitly label outliers as noise, which is usually the better fit for messy real data where "most points
belong to nothing in particular" is the truth. **Gaussian mixture models** give soft probabilistic
membership, useful when items genuinely belong partly to several groups.

**Trade-off.** Evaluation is the hard part and where most clustering work quietly fails. Internal metrics —
silhouette score, Davies–Bouldin, Calinski–Harabasz — measure *geometric tidiness*, not usefulness. A
clustering can score beautifully and mean nothing. They also systematically favour the assumptions of
whichever algorithm you used, so comparing k-means to HDBSCAN by silhouette is close to meaningless.

**Practice.** Judge on three things instead. **Stability**: re-run on a resampled subset or with a
different seed and check whether substantially the same groups appear; unstable clusters are an artefact.
**Actionability**: do the segments differ on something you can act on *differently*? A segmentation that
does not change any decision is a chart, not a finding. **External validation**: do the clusters correlate
with an outcome you did not cluster on — retention, conversion, cost to serve?

For text specifically, the current standard recipe is: embed the documents, reduce with UMAP, cluster with
HDBSCAN, then have an LLM read a sample from each cluster and name the theme. BERTopic packages roughly
this. That last labelling step is what turns cluster IDs into something a human will actually use, and it is
also where an LLM adds real value to a classic ML pipeline rather than replacing it.

### Q20 — What is deep learning, and when is it genuinely the right tool?

**Mechanism.** Deep learning uses neural networks with many layers, where each layer learns progressively
more abstract features from the one below. Its defining advantage is **learned representation**: you do not
hand-engineer features, which is exactly what made it dominant on perceptual data where nobody knew how to
hand-engineer features anyway.

**Trade-off.** It is the right choice when the input is unstructured and high-dimensional — images, audio,
video, natural language — when you have substantial data (typically tens of thousands of examples, unless
you are fine-tuning something pre-trained, which changes the calculus entirely), when relationships are
genuinely complex and non-linear, and when a pre-trained model exists that you can adapt.

It is the wrong choice more often than enthusiasm suggests. On tabular data, gradient-boosted trees still
match or beat it while being faster, cheaper, and far easier to explain — this has been tested repeatedly
and remains true. On small datasets it overfits badly. Where an auditor needs a defensible decision path,
its opacity is a genuine cost, not a philosophical one. And it demands accelerators, tuning expertise and
MLOps maturity that a boosted tree simply does not.

**Practice.** The pragmatic division: **deep learning for perception and language, gradient-boosted trees
for tables, deterministic code for anything specified**. Choosing correctly between those three is worth
more than any amount of hyperparameter tuning inside the wrong one. And before training anything deep, check
whether a pre-trained model plus a small head (Q21) gets you 90% of the way for 2% of the effort — in
2026 it usually does, and building from scratch is a decision that needs justifying rather than a default.

### Q21 — What is transfer learning, and why is it the reason modern AI is affordable?

**Mechanism.** Transfer learning means taking a model trained on one large task and reusing what it learned
for a different, usually smaller task. Instead of learning "what images look like" or "how language works"
from nothing, you start from a model that already knows and specialise it.

There is a spectrum of how much you change, and picking the right point on it is the whole skill.
**Feature extraction** freezes the pre-trained model entirely and trains a small classifier on its outputs
— cheapest, works with very little data, and often astonishingly strong. **Full fine-tuning** continues
training all layers on your data at a low learning rate — most capable, most data-hungry, most likely to
catastrophically forget the original capability if done carelessly. **Parameter-efficient fine-tuning**
(LoRA and relatives) trains a small number of added parameters while freezing the base — most of the
benefit at a fraction of compute and storage, and crucially it lets you keep many task-specific adapters
over a single base model (Q56). And **in-context learning** — putting examples in the prompt — is transfer
learning with no training at all.

**Trade-off.** The benefit is a collapse in data and compute requirements by orders of magnitude: a task
that would need a million labelled examples from scratch might need a few hundred on a good pre-trained
base. The limitation is **domain distance**. Transfer works well when your data resembles the pre-training
distribution and degrades as it diverges. Specialised domains — industrial sensor imagery, medical
modalities, low-resource languages, niche legal registers — still require genuine domain data, and the
convenient assumption that "the foundation model has seen everything" is where a lot of pilots quietly fail.

**Practice.** Always start at the cheap end of the spectrum and move up only when measurement demands it:
in-context examples, then feature extraction, then PEFT, then full fine-tuning. Each step multiplies cost
and commitment. Test the domain-distance question early with a small held-out set from your *actual* data
rather than a public benchmark, because that is the number that decides whether this project is a week or a
quarter.

### Q22 — What does a good hybrid system look like, concretely?

**Mechanism.** Take insurance claim triage, because it exercises every component and is recognisable across
industries.

First, OCR plus a vision-capable model extract structured fields from documents and photos — the
messy-input-to-structure step, where generative models genuinely excel. Second, deterministic code
validates policy numbers, dates and coverage against the core system; this is lookup and arithmetic and it
must not be probabilistic. Third, a gradient-boosting model scores fraud risk and expected cost from
tabular features, producing a calibrated number. Fourth, rules apply hard policy: auto-approve below a
threshold at low risk, auto-decline on clear exclusions, route everything else to a human. Fifth, an LLM
drafts the customer-facing explanation letter, grounded strictly in the actual decision fields rather than
its own impression of the case. Sixth, a human approves anything above threshold. Seventh, every decision
is logged with its inputs, for audit and as future training data.

**Trade-off.** The pattern generalises: **generative models for unstructured→structured and
structured→language; ML for prediction and scoring; deterministic code for policy, arithmetic and
enforcement; humans for the tail.** The cost of this architecture is that it has seven components with six
interfaces, each of which can break, and it requires someone who understands all of them to reason about
end-to-end behaviour. A single-model design is genuinely simpler — it is just also unable to meet the
accuracy, auditability and cost requirements simultaneously.

**Practice.** Two operational rules make hybrids maintainable. **Type the interfaces**: every hand-off
between components is a validated schema, not a hopeful string. And **evaluate each component with its own
appropriate metric** — AUC for the scoring model, extraction accuracy for the document step, groundedness
for the letter, end-to-end task success for the whole — because a single blended score tells you nothing
about where to work.

This also explains the demo-to-production gap precisely. A demo is steps 1 and 5, and it can be built in a
day. Steps 2, 4, 6 and 7 are the work, they take a quarter, and they are what makes the thing legal,
affordable and trustworthy.

---

## 4. RAG and retrieval

### Q23 — What is RAG, and when is it the right approach?

**Mechanism.** RAG — retrieval-augmented generation — means: before generating, fetch relevant passages
from an external knowledge source, put them in the prompt, and instruct the model to answer *from them*.
The goals are fresh knowledge, private knowledge, reduced hallucination, and answers that cite where they
came from.

The single most useful reframing is this: **RAG is a search problem with a generation step attached.** Most
of the quality lives in chunking, retrieval and re-ranking, not in the model. Teams that treat it as an LLM
problem spend weeks tuning prompts and wonder why nothing improves; teams that treat it as an information
retrieval problem — with recall metrics, relevance judgements and a re-ranker — get results.

**Trade-off.** RAG solves freshness, privacy and attribution, and it buys them at the cost of an entire
subsystem: ingestion, parsing, chunking, embedding, an index to operate, permission synchronisation,
freshness handling, retrieval evaluation, and **two failure modes to debug instead of one** (Q29). It also
adds latency to every request. That is a real ongoing engineering commitment, not a library import.

**Practice.** Use it when knowledge changes often, is internal or user-specific, requires source
attribution, is too large for the context window, or when access control must be enforced per document.

Do *not* use it when the task is a *skill* rather than knowledge (style, tone, format — that is prompting or
fine-tuning), when the whole corpus comfortably fits in the prompt with caching (Q35), or when the question
is really a deterministic lookup. If the answer is "select balance from accounts where id = …", the right
technology is SQL through a tool call, not semantic search over a document about accounts. A surprising
number of "RAG projects" are actually text-to-SQL projects wearing a costume.

### Q24 — Walk through a production RAG pipeline end to end

**Mechanism — offline, indexing.** Ingest from sources. Parse and clean — PDFs, HTML, Office documents,
scanned images, tables, and all the encoding misery that implies. Chunk into retrievable units, attaching
metadata to each chunk: source, section path, timestamps, and **access-control information**. Embed each
chunk. Store vectors alongside metadata, and build a keyword index in parallel.

**Mechanism — online, at query time.** Rewrite the query: resolve pronouns from conversation history,
expand acronyms, and optionally generate several query variants to widen recall. Retrieve candidates with
hybrid search (Q26), filtered by the user's permissions and by recency. **Re-rank** the candidates with a
cross-encoder (Q27) — usually the single largest quality lever in the pipeline. Select the top few that fit
your token budget. Assemble the prompt with the passages, their sources, and explicit instructions to
answer only from them. Generate with citations. Post-process: validate the output shape, verify claims are
supported, refuse if they are not. Log the query, the retrieved document IDs and scores, the answer, and any
user feedback.

**Trade-off.** Each stage adds latency and cost, and the stages people cut to save both — query rewriting
and re-ranking — are precisely the ones carrying the most quality. The honest calculus is that a fast bad
answer is worth less than a slightly slower good one for almost every knowledge-work use case, but that is a
product decision to make explicitly rather than a default to inherit from whichever tutorial you started
from.

**Practice.** The stages teams skip and later regret are consistent: **permission filtering** (skip it and
your assistant becomes a mechanism for leaking documents between teams — this is the failure that gets
people fired), **query rewriting** (skip it and multi-turn conversations retrieve nonsense because "what
about the second one?" is not a searchable query), **re-ranking** (skip it and you are relying on embedding
similarity alone, which is the weakest signal in the stack), and **the logging loop** (skip it and you can
never explain or improve anything). Build the pipeline with all seven stages present but simple, rather than
four stages done elaborately.

### Q25 — RAG, fine-tuning, or prompting — how do you actually choose?

**Mechanism.** These solve different problems, and most confusion comes from treating them as competing
answers to one question.

**Prompting** is always first. Fastest and cheapest to iterate, no infrastructure. Use it for behaviour,
format, tone, and any knowledge small enough to include. Limits: per-call token cost, and it cannot hold a
large corpus.

**RAG** is for when the model needs *knowledge* it does not have: large, changing, private,
permission-scoped, or requiring citations. It solves freshness and attribution. Cost: a retrieval system to
build and operate, plus latency.

**Fine-tuning** is for when the model needs a *skill, format or style* you cannot prompt it into: a rigid
output structure, a domain dialect, a high-volume classification task, or when you need to move behaviour
out of a long prompt into a small fast model to cut cost and latency (Q55). Cost: labelled data, a training
and evaluation pipeline, and re-doing the work when base models improve.

**Trade-off.** The rule of thumb — **knowledge → RAG; behaviour → fine-tuning; everything → try prompting
first** — is right but incomplete, because they compose. A fine-tuned small model with RAG in front of it is
a very common and very effective production shape: the fine-tune handles the format and domain register at
low cost, retrieval handles the facts.

**Practice.** The most expensive mistake in this space is **fine-tuning to teach facts**. It is slow,
unreliable, and the facts are stale the moment training ends — and worse, the model will state them
confidently because they are now in the weights, with no citation and no way to correct them short of
retraining. If someone proposes fine-tuning so the model "knows our products", that is a RAG requirement
described in fine-tuning language. The reverse error also exists but is cheaper: using RAG to enforce an
output format, by stuffing examples into retrieved context, when a fine-tune or a schema would do it
deterministically.

### Q26 — What is hybrid search, and why is pure vector search a demo rather than a product?

**Mechanism.** Vector search finds passages whose embedding is geometrically near the query's embedding —
it matches *meaning*, so it handles paraphrase, synonymy and conceptual similarity. Keyword search (BM25 is
the standard algorithm) matches *terms*, weighting rare terms more heavily.

They fail in opposite directions, which is why combining them works so well. Vector search is weak on exact
identifiers: SKUs, error codes, part numbers, invoice numbers, version strings, surnames. `ERR_4471` has no
meaningful semantic neighbourhood — it is a token the embedding model has essentially no useful geometry
for, so a vector search for it returns passages about errors in general. Keyword search nails it instantly.
Conversely, keyword search misses "how do I stop the machine" when the manual says "emergency shutdown
procedure", which vector search handles trivially.

**Hybrid search** runs both and fuses the results. The fusion is usually either a weighted score
combination or **Reciprocal Rank Fusion** (RRF), which combines by rank position rather than by score and
is popular precisely because it needs no score normalisation between two incomparable scales.

**Trade-off.** Hybrid costs you a second index to maintain and populate, a fusion parameter to tune, and
slightly more latency. In exchange it removes the most embarrassing class of retrieval failure — the one
where a user searches for the exact error code printed on their screen and gets nothing.

**Practice.** Default to hybrid for any corpus containing identifiers, codes, product names, or proper
nouns, which is nearly every real corpus. Most vector stores now support it natively (Q28), so the
implementation cost is usually configuration rather than construction. Tune the weighting on your own
labelled query set — and build that query set from *real user queries*, especially the ones that returned
nothing, because those are where the gap between your assumptions and reality lives.

### Q27 — What is re-ranking, and why is it usually the highest-leverage improvement?

**Mechanism.** Retrieval and ranking are different jobs, and hybrid retrieval is optimised for the wrong
one. Embedding-based retrieval computes the query vector and each document vector *independently*, then
compares them — which is what makes it fast enough to search millions of chunks, because document vectors
are precomputed. The cost of that independence is precision: the model never sees the query and the document
together.

A **cross-encoder re-ranker** does exactly that. It takes the query and one candidate document *jointly*
and scores their relevance with full attention across both. This is dramatically more accurate and
dramatically more expensive — too expensive to run over a whole corpus, exactly right to run over the 30–100
candidates that first-stage retrieval returned.

The pattern is therefore two-stage: retrieve broadly and cheaply for **recall**, then re-rank narrowly and
expensively for **precision**, then pass only the top few to the generator.

**Trade-off.** Re-ranking adds latency — typically tens to low hundreds of milliseconds depending on
candidate count and whether the re-ranker is hosted or local — and it adds either an API dependency or a
GPU. In return it is repeatedly, across many teams and corpora, the single change with the largest quality
effect in a RAG pipeline. If you have budget for exactly one improvement, this is usually it.

**Practice.** Options span hosted and self-hosted, and being even-handed matters here because the
economics differ sharply. Hosted APIs (Cohere Rerank, Voyage, Jina, Mixedbread) are the fastest to adopt
and require no infrastructure. Open-weight cross-encoders you can run yourself (the BGE reranker family,
Jina's open models, mixedbread's, and the long-standing MS MARCO cross-encoders available through
sentence-transformers) cost GPU time instead of per-call fees and keep data in your boundary. A cheap
middle path is an **LLM-as-reranker**: ask a small fast model to score relevance, which needs no new
dependency but costs more per candidate than a purpose-built re-ranker.

Cap the candidate count — re-ranking 100 candidates is rarely twice as good as re-ranking 30 and is
reliably twice as slow. And measure the difference on your own labelled set before and after, because the
size of the win varies enormously by corpus.

### Q28 — Do you need a vector database, and how do the options actually differ?

**Mechanism.** A vector database stores high-dimensional embeddings and answers nearest-neighbour queries
quickly, using approximate indexes (HNSW and IVF being the common families) that trade a small amount of
recall for orders of magnitude of speed. Exact cosine similarity over ten million chunks per query is not
viable at interactive latency; that is the core need. Beyond raw search it gives you metadata filtering
(permissions, tenant, date, source type), upserts and deletes as documents change, hybrid search, and
horizontal scale.

**Trade-off.** The category is crowded and the options differ on axes that matter more than benchmark
throughput. The honest summary of the mid-2026 landscape:

- **pgvector** — a PostgreSQL extension. If you already run Postgres, this is the default worth beating:
  one database, one backup story, one security model, transactional consistency with your relational data,
  and SQL for filtering. Scales further than most people expect. Weaker at very large scale and at
  specialised hybrid-search features.
- **Qdrant** — Rust-based, open source, strong payload filtering, self-host or managed. A common choice when
  you have outgrown pgvector but want to keep control.
- **Weaviate** — open source with strong hybrid search and a module ecosystem; opinionated in ways that help
  if you agree with the opinions.
- **Milvus / Zilliz** — built for very large scale, more operational surface to match.
- **Chroma** — the friendliest developer experience for prototyping, now with a managed offering.
- **Pinecone** — the established managed option: least operational work, per-usage pricing, no self-host.
- **LanceDB** — embedded and columnar, good for multimodal and local-first workloads.
- **Turbopuffer** and similar object-storage-native engines — dramatically lower cost per gigabyte for large
  or write-heavy corpora, at the cost of different latency characteristics.
- **Elasticsearch / OpenSearch / Vespa** — if you already run a serious search platform, it probably does
  vectors now, and consolidating beats adding a system.

**Practice.** Start with what you already operate. The question is not "which vector database is best" but
"what is the smallest number of systems that meets the requirement" — and for a corpus in the hundreds of
thousands of chunks with a Postgres already in production, adding a specialised store is often pure
operational cost. Adopt a dedicated engine when you have a concrete driver: scale, hybrid-search quality,
multi-tenancy at volume, or a cost profile the incumbent cannot meet. And whichever you choose, verify three
unglamorous things before committing: metadata filtering performance *with* your real filters, delete
propagation behaviour, and what a full reindex costs — because you will reindex (Q31).

### Q29 — What are the two failure modes of RAG, and why must you tell them apart?

**Mechanism.** **Retrieval failure** — the right information never reached the model. It was not indexed;
chunking split it in half; the embedding did not match the query's phrasing; a permission or recency filter
excluded it; or top-k cut it off. Symptoms: confident but generic or incomplete answers, or a false "I
couldn't find that".

**Generation failure** — the right information *was* in the context and the answer still went wrong. The
model ignored it, contradicted it, blended it with parametric memory, missed a detail buried mid-context, or
attached the wrong citation.

**Trade-off.** They are measured with different metrics — retrieval recall and precision versus
groundedness and faithfulness — and fixed in entirely different parts of the system. Conflating them is the
most expensive diagnostic error in RAG work, because the natural instinct is to blame the model, and prompt
engineering cannot fix a chunking problem. Teams routinely spend weeks on the generation layer to solve a
retrieval problem, and the effort produces small random improvements that feel like progress.

**Practice.** Build the discriminating test into your workflow, because it is cheap: take a failing case and
manually paste the *correct* passage into the prompt. If the answer becomes correct, you have a retrieval
problem. If it stays wrong, you have a generation problem. That two-minute experiment redirects weeks of
work.

Then instrument it permanently: log retrieved document IDs with every request, and maintain a labelled query
set with known-relevant documents so retrieval recall is a monitored number rather than an impression. When
recall is healthy and answers are still wrong, *then* work on the generation layer.

### Q30 — How do you balance latency, relevance and cost in retrieval?

**Mechanism.** Three levers, pulling against each other.

**Latency levers**: smaller top-k, skip or shrink re-ranking, cache embeddings and frequent queries, run
retrieval in parallel with other work, stream the first token, use a faster generator, keep the index warm.

**Relevance levers**: hybrid search, cross-encoder re-ranking, query rewriting and expansion, better
chunking, larger candidate pools, a stronger generation model.

**Cost levers**: fewer and shorter chunks in context, prompt caching for the stable prefix, a cheaper
embedding model, tiered routing so only hard queries get the expensive path, an object-storage-native index
for large corpora.

**Trade-off.** Notice that nearly every relevance lever costs latency and money, and nearly every cost lever
costs relevance. There is no configuration that is simply better; there is only a configuration that is
right for a stated budget. The failure mode is optimising one number in isolation — a team improves recall
by 8% and ships it without noticing P95 latency doubled, and the feature gets abandoned by users who found
it too slow to be worth using.

**Practice.** Set the budget from the use case *before* tuning. An in-flow support assist needs sub-second
response and can accept top-5 with a shallow re-rank. A legal or clinical research tool can take ten seconds
and should retrieve fifty candidates and re-rank hard. A nightly batch enrichment job has no latency
constraint at all and should use the most accurate configuration you can afford.

Then report **quality and P95 latency and cost per query together on every change**, as a triple. This
single reporting discipline prevents the most common silent regression in retrieval systems, because it
makes the trade-off visible to whoever is approving the change rather than discovered later by users.

### Q31 — How do you handle information that changes constantly?

**Mechanism.** Split the question first, because *changing documents* and *live state* need completely
different answers, and conflating them is a common and expensive design error.

For **changing documents**: reindex incrementally and event-driven — on write, not in a nightly batch. Use
stable chunk IDs derived from content hashes so only genuinely changed chunks are re-embedded. Propagate
hard deletes to the index, because stale retrieval of deleted content is a compliance problem and not
merely a quality one. Put `valid_from` / `valid_to` timestamps on every chunk so retrieval can prefer
current content and the answer can state what date it reflects.

For genuinely **live state** — price, stock level, account balance, order status, ticket status — do not put
it in the index at all. Call the system of record through a tool at query time (Q36). RAG is for corpora;
APIs are for facts that must be correct right now.

**Trade-off.** Event-driven reindexing is more work than a nightly cron and introduces ordering and
idempotency concerns: the same document updated twice quickly, deletes racing with updates, and partial
failures leaving the index inconsistent with the source. A nightly rebuild is simpler and wrong for hours at
a time. Which you can tolerate depends entirely on the domain — a policy wiki can be a day stale, a pricing
document cannot.

**Practice.** Three things pay for themselves. Make the pipeline **idempotent and replayable**, so you can
reprocess a document or a whole source without duplicating or corrupting. Keep a **reconciliation job** that
periodically compares source and index and reports drift, because event-driven pipelines lose events and
you want to find out from a metric. And **surface freshness in the interface**: "as of 14:32 today" with a
source link prevents an entire category of trust incident that no amount of retrieval tuning will, and it
costs one line of UI.

---

## 5. Context and memory

### Q32 — What is context engineering, as distinct from prompt engineering?

**Mechanism.** **Prompt engineering** is crafting the instruction: wording, examples, output format, role,
reasoning scaffolding. It is static, human-authored, and optimises a single call.

**Context engineering** is designing the *system* that decides — dynamically, on every call — what
information occupies the context window: what to retrieve, what to summarise, which history to keep or drop,
which tool results to include, which reusable procedures to load, in what order, all inside a hard token
budget. It is an architecture and a runtime policy, not a piece of text.

The term emerged for a real reason. As models became better at following instructions, the bottleneck moved
from *how you ask* to *what the model can see*. In agentic systems that accumulate tool outputs over long
horizons, context assembly becomes the dominant design problem: a single verbose tool result can consume
more window than the entire conversation, and the decision about what to keep is made hundreds of times per
session.

**Trade-off.** Prompt engineering is cheap to change and cheap to get wrong. Context engineering is
architecture: it involves caching strategy, retrieval, summarisation, memory storage and eviction policy,
and it is expensive to retrofit. Teams that treat context as "whatever we happen to concatenate" hit a wall
where the system works on short interactions and degrades unpredictably on long ones, and there is no
prompt fix for that.

**Practice.** Make the budget explicit and instrumented: log tokens per slot per request. Decide eviction
policy deliberately rather than inheriting your framework's default (which is almost always "drop oldest",
the worst option, since the oldest content is the instructions). Prefer structured, deduplicated
representations over raw dumps — a table of five fields beats a page of prose carrying the same information.
Truncate tool outputs at the boundary with a summary plus a pointer, rather than letting a 50,000-token API
response into the window. And treat context assembly as testable code with its own unit tests, because it is.

### Q33 — What goes into the context window, and in what priority?

**Mechanism.** A workable priority order, highest first:

1. **System prompt and policy** — identity, rules, safety constraints, output contract. Small, always
   present, and the thing you least want silently evicted.
2. **Tool definitions** — only the tools relevant to this surface, because selection accuracy degrades as
   the tool list grows (Q39).
3. **Task-critical state** — the current request, plus structured state such as the record being edited or
   the file currently open.
4. **Retrieved evidence** — top re-ranked passages, with sources attached.
5. **Recent conversation turns** — verbatim.
6. **Compressed history** — a rolling summary of older turns.
7. **Long-term memory and preferences** — only what is relevant right now, not everything you know.
8. **Few-shot examples** — the first thing to cut once the model handles the task without them.

**Trade-off.** Three principles govern the layout, and they interact. Put the most important material at
the **beginning and the end**, because attention is weakest in the middle. Keep the **prefix stable** —
byte-identical — so prompt caching works (Q81), which directly conflicts with the instinct to put dynamic
context near the top. And **budget per slot**, so one oversized retrieval result cannot crowd out the user's
actual question.

The stable-prefix requirement is the one people discover late and expensively: a timestamp or a request ID
interpolated near the top of the system prompt silently destroys cache hit rates on every call, and the
symptom is a cost line that is several times higher than modelled with no obvious cause.

**Practice.** Order as: stable system prompt and tool definitions (cached) → stable shared context (cached)
→ dynamic retrieved evidence → conversation → the actual instruction restated last. Verify cache
effectiveness by checking the provider's cache-read token counts rather than assuming: if cached reads are
zero across repeated similar requests, something in your prefix is varying and it is worth an hour to find
out what.

### Q34 — How do you give a system memory, and what makes memory dangerous?

**Mechanism.** Memory is not one thing, and naming the layers prevents most bad designs.

**Working memory** is the context window itself — the current conversation, trimmed or summarised.
**Session state** is a scratchpad the system reads and writes while completing a task. **Long-term semantic
memory** holds facts and preferences in a database, retrieved when relevant — either by vector search or,
often better, as explicit key–value records ("prefers metric units", "works in the Prague office").
**Episodic memory** stores summaries of past sessions and their outcomes. **Procedural memory** is learned
workflows, which in practice means reusable skills (Q40).

**Trade-off.** The hard parts are not storage — they are policy, and each has a failure mode. **Write
policy**: what is worth remembering? Remember too much and retrieval becomes noise; too little and the
feature is pointless. **Retrieval policy**: when to surface a memory, given that irrelevant recall is worse
than none. **Contradiction and staleness**: when the new fact conflicts with the stored one, is it
last-write-wins, or do you ask? A memory that a customer moved cities, stored two years ago, may be wrong.
**User control**: how does someone inspect, edit and delete what the system believes about them — which is
simultaneously a trust requirement and, for personal data, a legal one under GDPR's access and erasure
rights.

**Practice.** Bad memory is worse than no memory. Silently wrong personalisation destroys trust and is
nearly impossible for a user to diagnose, because they cannot see what the system thinks it knows. So:
prefer explicit, inspectable, structured memory over opaque embedding stores where you can; show the user
what is remembered and let them delete it; timestamp every memory and treat age as a retrieval signal;
never store secrets or credentials in memory (they will be replayed into future contexts, which is exactly
the exposure you were avoiding); and start with session-scoped memory only, adding cross-session persistence
when you have a concrete use case rather than because it sounds impressive.

### Q35 — Long context or RAG? The window is a million tokens now

**Mechanism.** With very large context windows, a genuine alternative to retrieval exists: put the whole
corpus in the prompt and let attention do the retrieval. For a corpus that fits, this is simpler, often more
accurate (nothing is missing because nothing was filtered out), and requires no index, no embedding model,
no chunking strategy and no reindexing pipeline. Combined with prompt caching, the cost can be far lower
than intuition suggests, because the corpus is cached and only the question varies.

**Trade-off.** It stops working for four reasons, and knowing which one applies to you decides the
architecture. **Size**: a million tokens is roughly a few thousand pages — large for a handbook, trivially
small for a document management system. **Attention dilution**: retrieval quality inside a very long context
degrades, so "it fits" is not "it works" (test it rather than assuming). **Cost at scale**: cached input is
cheaper but not free, and paying for a million tokens on every one of a million daily requests is a
different business than paying for two thousand. **Permissions**: if different users may see different
documents, you cannot put the union of all documents in a shared cached prefix — this alone rules it out for
most internal deployments.

**Practice.** Use long context when the corpus is bounded, shared by all users, and changes rarely — product
documentation, a policy handbook, a codebase for a coding assistant, a contract under review. Use retrieval
when the corpus is large, per-user permissioned, or fast-changing. And note the strongest option is often
**both**: retrieve generously into a large window rather than agonising over top-3 versus top-5. Large
windows do not eliminate retrieval; they make retrieval much more forgiving, which is a real and
underappreciated improvement.

---

## 6. Agents, tools and protocols

### Q36 — An AI feature "calls a tool". What actually happens?

**Mechanism.** Step by step, because the details are where incidents come from:

1. Your application sends the user message, the system prompt, and the **tool schemas** to the model.
2. The model returns a structured tool-call request — a name plus JSON arguments — instead of prose.
3. Your orchestration layer parses and **validates** those arguments against the schema.
4. It checks authorisation and rate limits: is *this* user allowed to do *this*, to *this* record?
5. It executes the real function or API call.
6. The result — or a structured error — is appended to the conversation as a tool result.
7. The model is invoked again with that result, and either answers or requests another tool.
8. The whole trace is logged for debugging and evaluation.

**Trade-off.** Most people picture steps 1, 2 and 5. Steps 3, 4 and 8 are where production incidents are
prevented or caused. Skipping them makes the happy path work perfectly, which is exactly why they get
skipped — nothing fails in the demo. A tool-using system without argument validation, authorisation checks
and full tracing is not a system; it is a demo with database credentials.

**Practice.** Treat step 4 as the security boundary and design it in code rather than in the prompt: the
model should not be *able* to act outside the user's permissions, regardless of what it emits. Return errors
as structured, actionable messages, because the model reads them and can recover — "`order_id` not found;
call `search_orders` first" produces a correct retry, while "500 Internal Server Error" produces a confident
hallucination. And log every tool call with arguments, result and latency, because tool-level metrics
(error rate, latency, which tools get chosen) are the most diagnostic signal you have about an agentic
system's health.

### Q37 — Does the model execute the function? Who does?

**Mechanism.** No. The model only *emits an intent to call*. Your application code — the orchestrator, the
agent framework, the MCP client — executes it. The model has no network access, no filesystem, no
credentials; it produces text that your code chooses to interpret as a command.

**Trade-off.** This separation is the entire safety architecture of tool use, and it is a feature rather
than a limitation. It is precisely where you enforce authentication, authorisation, argument validation,
rate limits, idempotency, audit logging and human approval. The cost is that all of that is *your* job, and
none of it is provided by default.

**Practice.** Concretely: if the model emits `refund(order_id, 5000)`, nothing happens until your code
decides to run it. Therefore "the AI issued a wrong refund" is never a model failure — it is a missing check
in the execution layer. Every tool that changes state needs three decisions made explicitly and written
down before launch: who owns it, what permission scope it runs under, and whether it requires human
confirmation.

Those are business decisions dressed as engineering details, and the useful discriminator is
**reversibility**. A tool that reads data, or writes something easily undone, can run automatically. A tool
that moves money, deletes records, sends external communication or changes permissions should require
confirmation until you have measured enough to justify otherwise (Q98). Also make writes **idempotent** with
an idempotency key, because agents retry, and a retried refund is a real incident that has happened to real
companies.

### Q38 — Why is a vague tool description a genuine bug?

**Mechanism.** The tool description *is* the model's only documentation, and it is **runtime input**, not a
comment. The model chooses tools by matching the user's intent against those descriptions, and fills
arguments by reading the parameter descriptions. A description like "gets data" competes with every other
tool and produces wrong selections, missing arguments, or invented ones.

**Trade-off.** This surfaces to users as a feature that behaves unpredictably, and it is invisible in code
review because nothing is technically broken — the code is correct, the *copy* is wrong. It also degrades as
the system grows: two tools with similar descriptions are worse than one, and selection accuracy falls as
the tool count rises, so a description that worked with five tools fails with twenty.

**Practice.** A good description states what the tool does, **when to use it**, **when not to**, argument
formats with examples, and what it returns including the error shape. Explicitly distinguishing tools from
one another — "use `search_orders` for order lookups, not `search_docs`" — is frequently the single
highest-leverage fix in a misbehaving agent, and it takes ten minutes.

Treat tool descriptions as **versioned product copy written for a machine**, with their own test cases: a
small eval set of user requests mapped to the tool that should be chosen, run in CI. When a model reaches
for the wrong tool, look at the wording before you look at the model. And when a system has grown to thirty
tools, the fix is usually consolidation or dynamic tool loading rather than better descriptions for all
thirty.

### Q39 — Agent, workflow, or chatbot — and what is the least agency that solves the problem?

**Mechanism.** A **chatbot** maps input to output in a fixed turn: one call, maybe retrieval, text back. A
**workflow** runs a predetermined sequence of steps, some of which may involve a model — the control flow is
yours. An **agent** pursues a goal over multiple steps where the *sequence is not predetermined*: it decides
what to do next based on what it just observed, takes real actions through tools, carries state, and decides
when it is done.

The practical marker is a loop whose exit condition the model controls. If your flow diagram is a fixed
pipeline, you have a workflow with model steps in it.

**Trade-off.** The consequences of genuine agency are worth stating plainly, because they are usually
underestimated. **Errors compound**: five sequential steps at 95% reliability each give 77% end-to-end, and
ten give 60%. **Cost and latency become unbounded** unless you bound them explicitly. **Evaluation gets
much harder**, because you must judge the trajectory and not only the final answer (Q70). **The safety
surface grows** with every tool attached. And **debugging is qualitatively harder**, because the same input
can take different paths.

Against that, agency buys the ability to handle tasks you cannot enumerate in advance — which is a real and
sometimes decisive capability.

**Practice.** The useful question is never "should this be an agent?" but **"what is the least agency that
solves this?"** Most valuable production systems land on a workflow with one or two agentic steps inside it:
deterministic control flow with a model deciding a bounded sub-question. That gives you most of the
flexibility with most of the predictability.

When you do build agentic loops, bound them hard: maximum steps, maximum tokens, maximum tool calls,
maximum wall-clock, maximum spend per task. And add a terminal state, because an agent with no defined
"done" will find one you did not intend.

### Q40 — What is the difference between a prompt, a tool, and a skill?

**Mechanism.** A **prompt** is instructions and context passed at inference time. It shapes *how* the model
behaves. Cheapest to change; changes nothing about capability.

A **tool** is an external function the model can invoke to read or change the world. It extends *what* the
model can do beyond producing text. It needs a schema, permissions and error handling, and it is the trust
boundary.

A **skill** is a packaged, reusable procedure — instructions plus optional scripts, templates and reference
files — loaded when it becomes relevant. It encodes *how we do this task here*: a repeatable workflow,
neither a new capability nor merely a sentence.

**Trade-off.** The distinction matters architecturally because each scales differently. Prompts scale badly:
the failure mode is one enormous system prompt trying to cover every case, where instructions start
conflicting and nobody dares delete anything. Tools scale to a point and then degrade selection accuracy.
Skills scale well through **progressive disclosure** — the model sees a short index of what exists and loads
full instructions only when relevant — which is what lets an organisation accumulate hundreds of encoded
procedures while paying context cost for only the ones in use.

**Practice.** A quick classifier for where something belongs: "always format monthly reports like this" is a
skill. "Look up the customer record" is a tool. "Be concise" is a prompt. If you find yourself adding the
fourth paragraph of domain procedure to a system prompt, that content wants to be a skill.

Skills also have an underrated organisational property: they are readable, reviewable in a pull request, and
versionable in git, which means a domain expert can own one without touching application code. That makes
them the natural home for encoded know-how — the definition of "good" that otherwise lives only in
somebody's head.

### Q41 — What is MCP, what are the competing protocols, and does any of it matter yet?

**Mechanism.** **MCP** — the Model Context Protocol, introduced by Anthropic and since adopted broadly — is
an open protocol for how AI applications connect to external tools, data sources and prompts. The problem it
solves is combinatorial: wiring N clients to M systems with bespoke integrations is N×M work, and every
integration invents its own way to declare tools, handle authentication and report errors. MCP turns that
into N+M — whoever operates a system writes one server, whoever builds an assistant writes one client, and
any client works with any server. The useful analogies are *USB-C for AI tools* or *the Language Server
Protocol for AI integrations*: in both cases a many-to-many mess collapsed by agreeing on one interface.

MCP covers **application-to-tool** communication. Adjacent efforts address different layers.
**Agent-to-agent** protocols (Google's A2A, and the Linux Foundation-hosted AGNTCY work) address how
independent agents discover and delegate to one another. **`agents.md`** and similar conventions are a much
lighter-weight idea: a plain file in a repository telling coding agents how to work in it.

**Trade-off.** MCP has real adoption and is the safe bet for tool integration. Agent-to-agent
interoperability is genuinely early: the standards are young, the use case (agents from different vendors
negotiating with each other) is more speculative than most roadmaps admit, and betting a design on it in
2026 is a research decision rather than an engineering one. There is also a security dimension worth naming:
a protocol that makes it easy to attach third-party tools makes it easy to attach *untrusted* third-party
tools, and the supply-chain question (Q97) becomes real the moment you install a community server.

**Practice.** Choose MCP when you have many systems and/or many client surfaces, when you want third parties
to integrate with you, or when you want users to bring their own tools. Choose a bespoke integration when
there is exactly one critical path where latency and payload shape must be optimised, when a vendor's
semantics are too strange to model generically, or when you need a guarantee no protocol change can break.
They compose: MCP for breadth, hand-rolled for the two or three hot paths that are core to your product.
Treat every third-party MCP server as untrusted code with the permissions you grant it, and scope those
permissions accordingly.

### Q42 — How do you coordinate multiple agents without creating a debugging nightmare?

**Mechanism.** Multi-agent designs split work across specialised agents — a researcher, a writer, a
reviewer; or a coordinator delegating to workers. The genuine reasons to do it are: materially different
toolsets or permission scopes per role, context isolation so one agent's clutter does not pollute another's,
independent subtasks that can run in parallel, and different models suited to different roles (a cheap model
for extraction, an expensive one for synthesis).

**Trade-off.** The costs are consistently underestimated. Coordination overhead is real: each agent
re-establishes context, and the coordinator then reads the report, so a delegated subtask can cost several
times what doing it inline would. Errors compound across agents as well as within them. Debugging requires
correlating traces across processes. And the most common actual failure is **prose hand-off**: agent A
summarises its findings in natural language, agent B misreads them, and the error is invisible because both
outputs look reasonable in isolation.

**Practice.** Resist multi-agent until a single agent with good tools has demonstrably failed. When you do
build it: one **orchestrator** owns the plan and the state. Each sub-agent gets a narrow, single-
responsibility scope, its own minimal toolset, and a **structured input/output schema** — never free-form
prose hand-off, which is where these systems rot. Run independent branches in parallel and join them; make
dependent steps explicit edges. Add validation at the join, a step and token budget per agent, and a terminal
state.

For observability, one **trace ID across all agents** is non-negotiable, plus per-agent success metrics and
an evaluation that judges the trajectory. Without that, you will know the system got worse and have no way
to find out which agent did it.

Framework support for this varies meaningfully (Q43) — some are built around graphs with durable state,
others around role-based crews — and the choice constrains how much of the above you get for free versus
build yourself.

---

## 7. Orchestration, frameworks and gateways

### Q43 — Which agent framework should you use, and how do they actually differ?

**Mechanism.** Agent frameworks provide the loop, state handling, tool plumbing and often observability that
you would otherwise write yourself. The landscape as of mid-2026 has consolidated into recognisable
positions, and being even-handed here matters because the differences are real rather than cosmetic:

- **LangGraph** — models the application as a **state graph**: nodes are functions or model calls, edges can
  be conditional, and there is an explicit shared state object. That buys cycles, branching, durable
  checkpointing and resumption, human-in-the-loop interruption mid-run, and replayable debugging. The
  heaviest concept load, and the most common choice for regulated or long-running production workflows.
- **CrewAI** — role-based collaboration. You define specialists (researcher, writer, editor) and how they
  hand off. Fastest path to a working multi-agent prototype; less control over the fine mechanics.
- **Microsoft Agent Framework** — the merger of Semantic Kernel and AutoGen (unified in 2026). The natural
  choice inside a Microsoft/.NET estate, with enterprise integration as its centre of gravity.
- **LlamaIndex Workflows** — strongest heritage in data and RAG-heavy agents; the ecosystem around ingestion
  and retrieval is its differentiator.
- **Pydantic AI** — type-safe and validation-first for Python teams who want structure without heavy
  orchestration machinery.
- **Mastra** and the **Vercel AI SDK** — the leading TypeScript options, which matters if your product is a
  web application and you do not want a Python service in the path.
- **Agno**, **smolagents**, **DSPy** — respectively high-throughput agent swarms, deliberately minimal, and
  a different paradigm entirely (DSPy *optimises* prompts programmatically rather than having you write
  them).
- **Vendor SDKs** — the model providers' own agent SDKs, which track their models' capabilities most closely
  and lock you in most tightly.
- **No framework** — raw SDK plus a couple of hundred lines of your own loop, retry, logging and validation.

**Trade-off.** Frameworks buy speed to first working version and cost you debugging transparency, dependency
churn, and sometimes hidden prompt mutation you did not author. The heavier the framework, the more of your
architecture it decides. And the churn is real: this list looked different eighteen months ago and will look
different again, which argues for keeping framework-specific code contained.

**Practice.** Choose on three axes rather than popularity: does your workflow need **cycles and durable
state** (graph frameworks) or is it a **role-based fan-out** (crew frameworks) or **mostly linear** (no
framework needed)? What **language** is your production stack? And how much **abstraction can you debug**
under incident pressure at 2am? A defensible pattern is to prototype with a framework to learn the shape of
the problem, then keep the framework for orchestration while making sure your prompts, tool definitions and
context assembly remain plain, readable, framework-independent artefacts. If a framework hides your prompts
from you, you abstracted the wrong layer.

### Q44 — Chain or state graph — when do you need which?

**Mechanism.** A **chain** is a mostly linear, directed composition of steps: retrieve, then summarise, then
format. It is easy to read, easy to reason about, and correct when the sequence is known in advance.

A **state graph** models nodes and edges with an explicit shared state, where edges may be conditional and
cycles are permitted. That buys loops, retries with different strategies, branching on intermediate results,
checkpointing and resumption, human interruption mid-run, and the ability to inspect and replay state.

**Trade-off.** Graphs cost conceptual overhead and boilerplate that a three-step chain does not need. Chains
break the moment reality introduces a cycle — and reality almost always does, because every real business
process has a "send it back for correction" path.

**Practice.** The one-liner: chains are for pipelines, graphs are for state machines with a model in the
loop. Choosing a graph is really an admission that your workflow has cycles and interruptions.

There is a third option that gets overlooked and is often correct for anything long-running: a **durable
workflow engine** — Temporal, or Restate, or the durable-execution features in orchestrators like Dagster
and Prefect. These were built for exactly the hard part (a process that runs for hours, survives restarts,
retries individual steps, and can be inspected and resumed) and they are far more battle-tested at it than
AI-specific frameworks. If your agent needs to wait three days for a human approval and then continue, that
is a durable workflow problem that happens to call a model, not an AI problem that happens to need
durability.

### Q45 — When does orchestration earn its place, and when is it over-engineering?

**Mechanism.** Orchestration earns its place when the task genuinely cannot be done in one call and the
coordination has value: multiple dependent steps where later steps need earlier results; conditional paths
depending on intermediate output; parallel work to fan out and merge; long-running processes needing
checkpointing; retries with fallback strategies; human approval gates mid-flow; several models or tools that
must be sequenced.

**Trade-off.** It does *not* earn its place when a single well-designed call with tools does the job. The most
common over-engineering pattern in AI products is a five-node graph where one prompt would work better, be
faster, and be debuggable. Every step adds latency, cost, a serialisation boundary, and a new place to fail —
and a graph with six nodes has more failure modes than six because they interact.

**Practice.** Apply this test: can you draw the flow and point at each node's **measurable** contribution to
output quality? Not its conceptual tidiness — its measured contribution. If a node exists because the
architecture looked cleaner with it, delete it and measure; you will frequently find no difference.

A useful discipline is to build the single-call version first, even if you are confident you will need
orchestration. It gives you a baseline, it sometimes turns out to be sufficient, and when it is not, you know
exactly which capability was missing rather than guessing at the start.

### Q46 — What is an LLM gateway, and do you need one?

**Mechanism.** A gateway sits between your application and model providers, giving you one interface across
many models plus cross-cutting concerns: routing, retries and failover, caching, rate limiting, spend
tracking and budget enforcement, key management, logging, and sometimes guardrails. The landscape splits
along hosting and emphasis:

- **LiteLLM** — the widely used open-source, self-hosted proxy. One OpenAI-compatible interface across
  providers, and you run it, so data and keys stay in your boundary.
- **OpenRouter** — a hosted aggregator: one key and one bill across a very large model catalogue. The fastest
  way to evaluate many models, including ones you have no account with.
- **Portkey** — gateway plus observability and guardrails, with an open-source core and a managed offering.
- **Cloudflare AI Gateway** — an edge layer focused on caching, analytics and rate limiting, with essentially
  no infrastructure for you to run.
- **Kong AI Gateway**, **Apache APISIX** and similar — the AI features of established API gateways, which is
  the right answer if you already operate one and want one control plane.
- **Cloud-native options** — the AI gateway features in the major clouds' API management layers, attractive
  when your governance story is already there.

**Trade-off.** A gateway is a component in your critical path, which means it is also a new failure mode and
a new latency contribution (typically single-digit to low-tens of milliseconds). Hosted gateways see your
prompts, which is a data-governance question to answer explicitly rather than discover during an audit.
Self-hosted gateways are something you now operate, monitor and upgrade.

**Practice.** You probably do not need one for a single application calling a single provider — a thin
internal wrapper of your own gives you the same abstraction with no dependency. You probably do need one when
you have several applications, several providers, a real need for spend visibility per team, or a compliance
requirement to log and control model calls centrally. Whichever way you go, keep the *interface* in your own
code so that adopting or removing a gateway later is a configuration change and not a refactor.

### Q47 — What are the four layers of an AI product stack, and where should you invest?

**Mechanism.** A clean framing:

1. **Infrastructure and compute** — accelerators, cloud, serving and inference optimisation.
2. **Model layer** — foundation models, fine-tunes, embedding and re-ranking models.
3. **Orchestration layer** — retrieval, tools, agents, routing, memory, guardrails, evaluation,
   observability.
4. **Application and experience layer** — workflow, interface, trust affordances, feedback loops,
   distribution, proprietary data.

**Trade-off.** Layers 1 and 2 are where the money is *spent* and where you have the least differentiation,
because your competitor can buy the same model tomorrow and frequently will. Layers 3 and 4 are where
defensibility lives: proprietary data and feedback loops, workflow depth, evaluation know-how, and interface
design for handling uncertainty. The uncomfortable implication is that the layers that feel most technically
impressive are the ones that matter least strategically.

**Practice.** The architectural rule that follows: **stay deliberately portable at layers 1 and 2, and invest
at layers 3 and 4.** Concretely, portability at layer 2 means a model interface with no provider assumptions
leaking upward, pinned versions, and a regression evaluation that constitutes your swap procedure. Investment
at layers 3 and 4 means the eval harness, the retrieval quality, the feedback capture, and the interface for
showing uncertainty are owned, tested code that you would not be willing to throw away.

---

## 8. Data and pipelines

### Q48 — ETL or ELT, and why did the industry switch?

**Mechanism.** **ETL** — extract, transform, load — is the classic order: pull from source systems,
transform in a dedicated processing layer, load the clean result into the warehouse. It made sense when
storage and compute were expensive and coupled, so you only stored data you had already refined.

**ELT** inverts the last two steps: extract, load raw into the warehouse, then transform *inside* the
warehouse using SQL. The switch happened because cloud warehouses made storage cheap and separated it from
compute, so keeping raw data became affordable and scaling transformations became a matter of buying query
capacity.

**Trade-off.** ELT's advantages are substantial: raw data is preserved so you can re-derive everything when
requirements change or a bug is found; transformations live in version-controlled SQL that analysts can read
and contribute to; testing and lineage are far easier; and you are not operating a separate transformation
cluster. The costs are real too — you are now storing everything, including data you should not keep, which
turns retention and PII handling into an active discipline rather than a side effect of not collecting.

ETL retains its places: when you must mask or drop sensitive fields *before* they land anywhere, when the
source is a stream needing enrichment in flight, or when compliance forbids storing raw data at all.

**Practice.** For anyone working on AI features, this matters because your embeddings, features and retrieval
indexes are *downstream* of these pipelines. When an AI feature quietly degrades, the cause is very often
upstream in a pipeline nobody thought to check (Q53). Know which layer you depend on and who owns it: "the
model got worse" and "the upstream table stopped refreshing" look identical from the outside and have
completely different owners.

### Q49 — What does dbt actually solve, and what are the alternatives?

**Mechanism.** dbt is the T in ELT. You write transformations as SQL `SELECT` statements; dbt works out the
dependency graph, materialises each as a table or view in the correct order, and brings software engineering
practice to analytics code.

The mechanics worth knowing even if you never write any: each **model** is a `.sql` file containing one
select. Models reference each other with `{{ ref('other_model') }}`, and that reference is what lets dbt
build the DAG automatically — you never hand-maintain execution order. **Tests** are assertions declared in
YAML (not null, unique, accepted values, referential integrity) or written as custom SQL, and they run in CI.
**Sources** declare raw tables and support freshness assertions. **Snapshots** capture slowly-changing
dimensions over time. **Documentation** and a browsable lineage graph generate from the same YAML.
**Macros** (Jinja) factor out repeated SQL. **Incremental models** process only new rows instead of
rebuilding everything.

**Trade-off.** What dbt solves is not "running SQL" — anything does that. It is that analytics code
historically had no version control, no tests, no dependency management, no documentation and no lineage, so
nobody could safely change anything. dbt made transformation code reviewable, testable and deployable like
application code. The cost is Jinja-templated SQL, which is powerful and can become unreadable, and a build
step in a place where analysts previously just ran queries.

The alternatives are real and differ meaningfully: **SQLMesh** offers column-level lineage and virtual data
environments, with a strong story on avoiding expensive full rebuilds. **Google Dataform** is the
GCP-native option. **Coalesce** is GUI-first with dbt-like rigour underneath, which suits teams where the
transformation owners are analysts rather than engineers. And for Python-centric shops, orchestrators like
**Dagster** model data assets directly with similar guarantees.

**Practice.** Whatever you choose, the non-negotiables are the same, and they are what AI work depends on:
transformations in version control, tests running in CI, generated lineage, and declared freshness
expectations on sources. Your feature tables and the clean text you embed should come out of that. Otherwise
"why did retrieval get worse?" is unanswerable, and you will spend a week discovering it was a schema change
three systems upstream.

### Q50 — How do you get data in, and what are the ingestion options?

**Mechanism.** Ingestion moves data from source systems into your warehouse or lake. The options differ
mainly in who maintains the connectors:

- **Fivetran** — the established managed option: hundreds of maintained connectors, you configure rather than
  code, priced on data volume. Lowest effort, highest recurring cost, and the cost is volume-coupled in a way
  that surprises people.
- **Airbyte** — open source with a managed option, very large connector catalogue, self-hostable when data
  residency matters.
- **dlt** — a Python library rather than a platform: you write pipelines as code, which suits teams that want
  version control and testing over a UI.
- **Meltano** — the Singer-tap ecosystem, orchestrated; strong when you want open-source connectors under
  your own control.
- **Custom extractors** — for the sources nobody built a connector for, which in any real business is
  several.
- **Change Data Capture** — for databases, reading the transaction log (Debezium being the common open-source
  route) rather than polling. Lower load on the source and near-real-time.
- **Event streaming** — Kafka, Redpanda, Pulsar, or cloud-native equivalents, when the source is genuinely
  event-shaped.

**Trade-off.** The real decision is *buy the connector maintenance or own it*. Managed connectors break less
and cost more; every API you integrate yourself is a thing that will break when the vendor changes it,
usually silently, usually at the worst time. Volume-based pricing on managed platforms also interacts badly
with the instinct to load everything.

**Practice.** Buy connectors for the long tail of SaaS sources where the integration is commodity and
breakage is annoying rather than interesting. Own the extraction for your core, high-volume, high-value
sources, where you care about semantics and latency. Use CDC rather than polling for databases. And whatever
you use, monitor **freshness and row counts** as first-class alerts, because the characteristic ingestion
failure is not an error — it is a pipeline that keeps running and stops delivering rows.

### Q51 — Warehouse, lake or lakehouse — and does it matter for AI work?

**Mechanism.** A **data warehouse** (Snowflake, BigQuery, Redshift, and increasingly ClickHouse for
analytical workloads) stores structured, modelled data optimised for SQL analytics. A **data lake** stores
raw files of any shape in object storage, cheap and schema-on-read. A **lakehouse** puts table semantics —
transactions, schema evolution, time travel — on top of object storage via open table formats: **Apache
Iceberg**, **Delta Lake**, **Apache Hudi**. Databricks is the best-known platform in this space, but the
formats themselves are open, and the industry has converged notably on Iceberg as the interoperability
point.

**Trade-off.** Warehouses give the best SQL performance and the simplest operations; you pay for storage in
their format and are more coupled to the vendor. Lakehouses give cheap storage, one copy of data readable by
many engines, and freedom from a single vendor's compute — at the cost of more moving parts and a genuinely
higher operational bar.

**Practice.** For AI work specifically, three things matter more than the category label. **Can you read the
data with the tools your ML and embedding pipelines use** — which in practice means Python and Spark
frameworks, not just SQL clients? **Can you do point-in-time queries**, because reconstructing "what did we
know at the moment of this event" is required for honest training data (Q52)? And **what does it cost to
scan a lot of data repeatedly**, because feature engineering and embedding pipelines do exactly that.

If you are choosing today with no legacy constraint, an open table format in object storage keeps the most
options open. If you already run a warehouse that works, the AI use case is rarely sufficient reason to
migrate.

### Q52 — What is a feature store, and do you need one?

**Mechanism.** A feature store defines, computes, stores and serves the input features that models consume.
It usually has two halves: an **offline store** with historical values for training, and an **online store**
with current values at low latency for inference.

The specific problem it solves is **training/serving skew** — the same feature implemented twice, slightly
differently, by two people in two languages. That mismatch is one of the most common causes of a model that
performs well in evaluation and badly in production, and it is nearly invisible because nothing errors. A
feature store also gives **point-in-time correctness** (assembling feature values as they were *at the
moment* of each historical event, which prevents leakage from the future), reuse of definitions across teams
and models, and monitoring of feature distributions.

**Trade-off.** It is real infrastructure with real operational weight, and it solves a problem you may not
have yet. The options span **Feast** (open source, bring your own stores), **Tecton** (managed, opinionated,
strong on streaming features), **Hopsworks** (open source with a platform around it), and the native feature
stores inside Databricks, Vertex AI and SageMaker — which are the pragmatic answer if you already live in one
of those.

**Practice.** You do not need one for your first model. A single team with a handful of models is better
served by disciplined dbt models plus a clear convention that the *same* SQL computes the feature for
training and for scoring. The case strengthens sharply with: several teams sharing features, many models,
real-time inference requirements, or a compliance need to state exactly what a model saw. Until then, get the
*discipline* — one definition, point-in-time correctness, monitored distributions — without buying the
platform.

### Q53 — What does data quality actually mean, and how do you enforce it?

**Mechanism.** Six dimensions cover most of it: **completeness** (are values missing?), **accuracy** (do they
match reality?), **consistency** (do systems agree?), **timeliness** (is it current enough?), **validity**
(does it conform to expected format and range?), and **uniqueness** (are there duplicates?).

Enforcement works in layers. At the **contract** level, agree an explicit schema between producing and
consuming teams and version it, so a rename is a negotiated change rather than a surprise. At the
**pipeline** level, run declarative tests on every build — not-null, unique, accepted values, referential
integrity, row-count ranges, freshness — and fail loudly. At the **monitoring** level, watch distributions
and volumes for anomalies rather than only checking hard rules. At the **process** level, give every critical
dataset a named owner and treat a data incident like a service incident, with a postmortem.

Tooling ranges from tests inside your transformation layer (dbt tests, SQLMesh audits) through dedicated
validation libraries (**Great Expectations**, **Soda**, **Pandera** for dataframes) to observability
platforms (**Monte Carlo**, **Elementary**, **Metaplane**) that detect anomalies you did not think to assert.

**Trade-off.** Assertions catch what you predicted; anomaly detection catches what you did not, at the cost
of false positives and a tuning burden. Doing only the first leaves you blind to novel failures; doing only
the second produces alert fatigue and gets muted.

**Practice.** This belongs in an AI guide because **models are unusually sensitive to silent data problems**.
A column that starts arriving null raises no exception; it quietly degrades predictions. The overwhelming
majority of "the model broke" incidents are data incidents, and teams that learn this early stop wasting
time retraining models to fix pipeline bugs. Start with assertions on the handful of columns your models
actually depend on, plus freshness on every source — that combination catches most of what will hurt you,
for a day of work.

### Q54 — How do you build a pipeline that feeds a retrieval index?

**Mechanism.** Treat it as a data pipeline with an embedding step, not an AI project with some plumbing:

1. **Ingest** from systems of record, incrementally where possible, with CDC or timestamps so you know what
   changed.
2. **Parse and normalise** — extract text from PDFs, HTML and Office formats; fix encoding; strip
   boilerplate; keep tables intact. This stage is dull and is where most quality is won or lost.
3. **Enrich with metadata** that retrieval will filter on: source, author, section path, timestamps, and
   above all **access-control information**.
4. **Chunk** on structure, with stable IDs derived from content hashes.
5. **Embed** only changed chunks, in batch, recording the embedding model version.
6. **Upsert** into vector and keyword indexes; **propagate deletes**.
7. **Validate** — check retrieval recall on a fixed query set after every rebuild.

**Trade-off.** The build-versus-buy question is live here: managed RAG-as-a-service offerings and the
ingestion pipelines inside frameworks (LlamaIndex, Haystack) will get you to a working index faster, and give
you less control over exactly the stages that determine quality. Document parsing specifically is worth
buying or using a specialised library for — it is a genuinely hard problem and hand-rolled PDF extraction is
a well-known time sink.

**Practice.** Two things people forget and later regret. **Permissions must be enforced at retrieval time**,
from metadata kept in sync as source permissions change — otherwise the assistant becomes a mechanism for
leaking documents across teams, which is the failure with actual career consequences. And you need a **full
rebuild path**, because you will change your embedding model or chunking strategy, and when you do, every
chunk must be reprocessed. Design for that on day one; discovering it later means an outage or a migration
project.

---

## 9. Fine-tuning and model customisation

### Q55 — When is fine-tuning actually the right answer?

**Mechanism.** Fine-tuning continues training a pre-trained model on your examples, adjusting weights so
that the behaviour you demonstrated becomes the behaviour you get by default. It changes *how* the model
behaves, durably, without spending prompt tokens on instructions every call.

There are four situations where it genuinely wins. **A rigid output format or domain register** that
prompting reaches only unreliably — a specific report structure, a controlled vocabulary, a house style.
**A high-volume narrow task** where a fine-tuned small model matches a large model's quality at a fraction
of cost and latency: classification, extraction, routing, tagging. **Moving a long prompt into weights**,
which cuts per-call tokens substantially when that prompt is paid on millions of calls. And **teaching a
capability the base model lacks**, such as a specialised structured language or an unusual annotation scheme.

**Trade-off.** The costs are consistently underestimated. You need labelled data — typically hundreds to low
thousands of examples for behavioural fine-tuning, and their *consistency* matters more than their count,
because the model will faithfully learn your annotators' disagreements. You need a training and evaluation
pipeline, which is real engineering. You need to redo the work when base models improve, and they improve
every few months, which means a fine-tune has a shelf life. And there is a genuine risk that the next general
model beats your fine-tune for free — a real outcome that has happened repeatedly.

**Practice.** Decision order: prompt → few-shot examples → RAG → routing and caching → fine-tune a small
model → train from scratch (essentially never for language). Stop at the first step that clears the quality
bar inside the cost envelope. Before fine-tuning, verify you have exhausted prompting *with measurement*
rather than by impression, and confirm the task is stable enough to be worth encoding — fine-tuning a
requirement that changes quarterly is a treadmill.

And the recurring expensive mistake: **do not fine-tune to teach facts** (Q25). It is unreliable, the facts
go stale, and worse, the model then states them confidently with no citation and no way to correct short of
retraining.

### Q56 — What are LoRA, QLoRA and PEFT, and why did they change the economics?

**Mechanism.** Full fine-tuning updates every weight, which requires memory for the weights, their
gradients, and optimiser state — typically many times the model's own size, putting large models out of reach
of anything but a serious GPU cluster.

**Parameter-efficient fine-tuning (PEFT)** sidesteps this. **LoRA** (Low-Rank Adaptation) freezes the base
model entirely and inserts small trainable matrices alongside selected layers. Because the added matrices are
low-rank, they contain a tiny fraction of the parameters — often well under 1% — yet capture most of the
adaptation. **QLoRA** adds quantisation: hold the frozen base model in 4-bit precision while training the
LoRA adapters in higher precision, which cuts memory further and is what made single-consumer-GPU fine-tuning
of large models genuinely practical.

**Trade-off.** PEFT gives you most of full fine-tuning's benefit at a fraction of compute, memory and
storage. It also gives an underrated operational advantage: adapters are small files, so you can keep **many
task-specific adapters over one base model** and swap them per request instead of hosting several full
models. What you give up is a slice of quality on the hardest adaptations, and full fine-tuning remains
better when you are teaching a genuinely new capability rather than a behaviour.

**Practice.** Start with LoRA or QLoRA; reach for full fine-tuning only when measurement shows PEFT is the
binding constraint. Be aware that adapter-per-tenant architectures are attractive and add real serving
complexity — verify your inference stack supports dynamic adapter loading before designing around it. And
remember quantisation is a quality decision as well as a memory one: 4-bit is usually fine, but "usually" is
not a measurement, so check on your own eval set.

### Q57 — Which fine-tuning tools should you use?

**Mechanism.** The tooling landscape as of mid-2026 has clear positions, and picking wrongly costs weeks:

- **Unsloth** — optimised kernels for single or dual GPU LoRA/QLoRA work, and meaningfully faster than
  alternatives on supported architectures (roughly a two-fold speedup on comparable benchmarks). The
  constraint is that its gains come *from* architecture-specific kernels, so unsupported models and advanced
  parallelism fall outside its sweet spot.
- **Axolotl** — the multi-GPU production workhorse: configuration-driven, broad architecture support,
  composable parallelism. Slower per-GPU than Unsloth, better when you have a cluster and a pipeline.
- **TRL** (Hugging Face) — the canonical library for the alignment methods themselves: SFT, DPO, GRPO, PPO.
  Reach for it when you need control over the training objective rather than a fast path to a LoRA.
- **LLaMA-Factory** — broad model coverage with a friendly interface, popular for breadth of supported
  architectures.
- **torchtune** — PyTorch-native, for teams who want to stay close to the framework.
- **Managed fine-tuning** — the model providers' own APIs, plus platforms like Together, Fireworks, Modal or
  Predibase. You upload data and get a model, with no infrastructure and less control.

**Trade-off.** Self-hosted training gives control and keeps data in your boundary at the cost of GPU
operations, environment management and CUDA debugging. Managed fine-tuning removes all of that and gives you
less visibility into what happened, less ability to iterate on the objective, and a dependency on the
provider continuing to offer it for the model you chose.

**Practice.** For a first fine-tune, use managed if the data can leave your boundary — it collapses a
two-week setup into an afternoon and answers the question of whether fine-tuning helps at all. Move to
self-hosted when you are iterating frequently, when data residency requires it, or when the economics of
repeated training runs justify owning the pipeline. Whichever path, budget as much time for **building the
evaluation set** as for training: without it you cannot tell whether the fine-tune helped, and a fine-tune
you cannot measure is a liability rather than an asset.

### Q58 — What data do you need to fine-tune, and how do you get it?

**Mechanism.** For behavioural fine-tuning you need input/output pairs demonstrating the behaviour you want,
in the format the training framework expects. Volume requirements are lower than people assume — hundreds of
examples often suffice for format and style, low thousands for more substantive behaviour — but
**consistency matters more than volume**. A thousand examples where annotators disagreed teach the model to
be inconsistent, faithfully.

Sources, in rough order of quality: **production data with human corrections** (the best, because the edit
diff is exactly the signal you want — see Q72); **expert-written examples** (high quality, expensive,
slow); **existing artefacts** repurposed as examples (past reports, resolved tickets, historical
translations); and **synthetic data** generated by a stronger model, then filtered by humans or by rules.

**Trade-off.** Synthetic data is the pragmatic answer to the cold-start problem and carries two real risks.
It inherits the generating model's biases and errors, so unfiltered synthetic data can teach your small model
to imitate a big model's mistakes with high confidence. And it can be legally constrained — many providers'
terms restrict using outputs to train competing models, which is a contract question to check rather than
assume.

**Practice.** Write the annotation guideline before collecting anything, and measure inter-annotator
agreement on a small sample: that number is the ceiling on your achievable quality, and discovering it is
0.4 before you label ten thousand examples is worth a great deal. Hold out a genuine test set *before* you
start iterating, and never look at it until you are done. Version the dataset alongside the model. And
prefer harvesting corrections from a human-in-the-loop production flow over commissioning a labelling
project — it is cheaper, better distributed against real inputs, and it improves continuously.

### Q59 — What are RLHF, DPO and GRPO, and do you need to care?

**Mechanism.** These are **preference optimisation** methods: they train on comparisons ("response A is
better than response B") rather than on single correct answers, which is how you teach subjective qualities
like helpfulness, tone and appropriate caution.

**RLHF** (reinforcement learning from human feedback) trains a separate reward model on human preferences,
then uses reinforcement learning — classically PPO — to optimise the language model against that reward.
Effective and operationally heavy: two models, an unstable training loop, careful tuning.

**DPO** (direct preference optimisation) skips the reward model and optimises directly on preference pairs
with a simpler loss. Far easier to run, which is why it became the default for teams doing this themselves.

**GRPO** (group relative policy optimisation), popularised by DeepSeek's work, compares groups of sampled
responses without needing a separate critic model. It has become prominent particularly for reasoning
capability, and framework support has moved quickly.

**Trade-off.** These methods shape *behaviour and judgement* rather than knowledge or format, which is
precisely what supervised fine-tuning does poorly. They also require preference data — pairs with a
judgement — which is more expensive to collect than input/output examples, and they are more sensitive to
data quality: inconsistent preferences produce a model that is confidently mediocre in a new way.

**Practice.** Most teams do not need to run these at all. They matter to you in three cases: you are
building a model product rather than an application; you have a genuinely subjective quality dimension that
supervised examples cannot capture and you have the preference data to train on; or you are choosing between
models and want to understand *why* two models with similar benchmark scores feel different — which is
usually a post-training and preference-data difference rather than a capability one.

If you do have preference data lying around — thumbs-up/thumbs-down with the two candidate responses, or
accepted-versus-rejected drafts — that is a genuine asset. Most organisations throw it away by not logging
the rejected alternative, which is worth fixing today even if you never train on it.

---

## 10. Inference, serving and local models

### Q60 — When should you run models locally, and when is it a mistake?

**Mechanism.** Running open-weight models on your own hardware means downloading weights and serving them
yourself, rather than calling a hosted API.

**Trade-off.** The genuine reasons to do it: data cannot leave the machine or network for legal or
contractual reasons; you need offline operation; volume is high and complexity low, so per-token API cost
would dominate; you want zero-marginal-cost experimentation; or you need version stability so that nothing
changes underneath you — which, for a validated regulated workflow, can be a hard requirement rather than a
preference.

The reasons not to, which get less airtime: **capability gap** — the distance between a model that fits
comfortably on a laptop and a hosted frontier model remains substantial on hard reasoning, long context and
tool use, and benchmark parity does not translate to parity on your task. **Utilisation economics** — a GPU
you own costs the same whether it is busy or idle, and at low utilisation an API is almost always cheaper.
**Operational surface** — quantisation choices, context limits, throughput tuning, driver and CUDA
management, upgrades, and capacity planning all become yours. **Hidden total cost** — the engineer-months
spent operating inference infrastructure rarely appear in the comparison that justified it.

**Practice.** The pattern that works for most organisations is a **hybrid**: local models for bulk
classification, extraction, embedding and development iteration; a hosted frontier model for the hard tail.
That is routing (Q80) applied across the hosting boundary rather than just across model tiers, and it
captures most of the cost benefit without the capability cost.

Before committing, run the honest arithmetic: total cost of ownership including engineering time, at your
*actual* expected utilisation rather than peak, against API pricing at your actual volume. And test the
capability gap on your own evaluation set, not on a leaderboard.

### Q61 — Ollama, LM Studio, Jan, llama.cpp — what are the differences?

**Mechanism.** These are frequently compared as if they were competitors on one axis, but they occupy
different layers, and understanding that resolves most of the confusion. **llama.cpp** is an *engine* — the
inference runtime that much of this ecosystem is built on, with GGUF as its model format. **Ollama**, **LM
Studio** and **Jan** are *experience layers* over that engine.

- **Ollama** — CLI-first with an OpenAI-compatible HTTP API. Effectively the default for developer local
  inference: install, `ollama run <model>`, and you have a working endpoint in minutes. Assumes you are
  comfortable in a terminal or calling an API.
- **LM Studio** — GUI-first with a model browser and chat interface. The right choice when the user wants to
  click rather than type, including non-developers evaluating models.
- **Jan** — open-source, offline-first, GUI. The pick when open source and privacy posture matter and you
  want the LM Studio experience without a closed product.
- **llama.cpp directly** — for embedded, constrained or unusual hardware, or when you want control over
  build flags and quantisation.
- **Also worth knowing**: **LocalAI** as a self-hosted OpenAI-compatible drop-in with broad modality support,
  **llamafile** for single-file distributable models, and **MLX** as Apple's framework, which is the fastest
  path on Apple Silicon specifically.

**Trade-off.** Because Ollama, LM Studio and Jan sit on the same engine and format, any GGUF model that runs
in one runs in the others — so the choice is about interface, distribution and licence, not capability.
Crucially, none of these is a *serving* system: they are optimised for one user at a time and fall over as a
multi-user backend (Q62).

**Practice.** Pick by who the user is. Developer prototyping and scripting: Ollama. Non-technical evaluation
or a desktop assistant: LM Studio or Jan. Embedded or exotic hardware: llama.cpp. Apple Silicon
performance: MLX. And do not put any of them behind a production endpoint serving concurrent users — that is
a different tool (Q62), and the mistake is common precisely because Ollama makes it look easy.

### Q62 — How do you serve a model to many users at once?

**Mechanism.** Production serving is a genuinely different problem from local inference, and the difference
is **concurrency**. The techniques that matter are continuous (in-flight) batching, so new requests join a
batch without waiting for the current one to finish; **paged attention**, which manages KV cache memory the
way an operating system manages virtual memory and is the core reason throughput improved so dramatically;
prefix caching, so shared prompt prefixes are computed once; and tensor parallelism for models too large for
one device.

The options:

- **vLLM** — the widely adopted open-source default. High throughput via paged attention and continuous
  batching, broad model support, OpenAI-compatible API. Roughly an order of magnitude more concurrent
  throughput than a single-user runtime.
- **SGLang** — a credible competitor that can edge vLLM on shared-prefix workloads, which is exactly what RAG
  and agent systems produce.
- **TensorRT-LLM** — NVIDIA's stack: the best performance on NVIDIA hardware if you accept the build
  complexity and the coupling.
- **Hugging Face TGI** — mature, well-integrated with the HF ecosystem.
- **LMDeploy**, **MLC-LLM** — further options, the latter notable for cross-platform and edge deployment.
- **Managed inference** — Together, Fireworks, Baseten, Modal, RunPod, and the clouds' own endpoints: you get
  serving without operating it, priced per token or per GPU-hour.

**Trade-off.** Self-hosted serving is where the promised cost savings of open weights actually materialise —
and only at high, steady utilisation. The bar is real: you need capacity planning, autoscaling on GPUs (which
is slower and lumpier than autoscaling stateless web servers), model loading times, upgrade paths, and
someone on call who understands CUDA errors.

**Practice.** If you are serving open-weight models to real users, start with vLLM unless you have a specific
reason not to, and consider SGLang if your workload is prefix-heavy. Strongly consider **managed inference**
for the first production deployment: it de-risks the capability question before you commit to the
infrastructure question. Measure throughput and latency **at your actual concurrency**, not single-request
latency, because that is the number that decides how many GPUs you need — and it is the number that surprises
people.

### Q63 — What is quantisation, and what does it cost you?

**Mechanism.** Model weights are typically trained in 16-bit floating point. Quantisation stores them in
fewer bits — 8-bit, 4-bit, and increasingly formats like FP8 with hardware support — which shrinks memory
proportionally and usually speeds up inference, because inference at generation time is bound by memory
bandwidth more than by arithmetic.

Formats and methods differ in what they optimise. **GGUF** with its k-quant variants dominates the
llama.cpp ecosystem. **GPTQ** and **AWQ** are post-training quantisation methods common for GPU serving.
**FP8** is increasingly first-class on recent accelerators. There is also **quantisation-aware training**,
which bakes robustness to quantisation into training rather than applying it afterwards.

**Trade-off.** Quantisation is what makes local inference practical at all: it is the difference between a
model needing a data-centre GPU and running on a laptop. The cost is quality degradation, which is real,
non-uniform, and easy to under-measure. 8-bit is usually indistinguishable. 4-bit is usually acceptable and
sometimes not. Below 4-bit, degradation becomes noticeable. Crucially, degradation is **task-dependent**:
casual conversation barely changes while precise instruction-following, structured output, arithmetic and
long-context recall degrade first — which are exactly the capabilities production systems depend on. A
quantised model that chats well can fail at reliable JSON.

**Practice.** Never accept a quantisation level based on a general claim. Run *your* evaluation set at
several levels and pick from your own numbers — this takes an afternoon and prevents a category of
mysterious quality complaint. Test structured output and instruction-following specifically, not just
open-ended generation. And record the quantisation level as part of your model version, because "we deployed
the same model" is not true if one was 4-bit.

### Q64 — Hosted API or self-hosted — how does the arithmetic actually work?

**Mechanism.** With a **hosted API** you pay per token. Cost scales linearly with usage, there is no fixed
commitment, and the provider's GPU economics set your floor. It is an operating expense that grows with
success.

With **self-hosting** you pay for *capacity*, not usage. The mental model inverts: the metric that matters is
**utilisation**. Idle GPUs are pure loss. Levers become batching, KV-cache management, quantisation, model
size and traffic shaping.

**Trade-off.** Break-even against an API typically requires sustained high volume — and "sustained" is the
operative word, because bursty traffic means provisioning for peak and paying for the trough. Add the
engineering cost honestly: at least a partial full-time role for a serious deployment, plus on-call. Then
add the risk items that do not appear in a spreadsheet: GPU supply constraining your scaling plans, model
loading times affecting deploy speed, and the opportunity cost of the engineers doing this instead of product
work.

Against that, self-hosting buys data residency guarantees no contract fully replaces, version stability, no
rate limits imposed by someone else, and predictable cost at scale.

**Practice.** Model both curves at your actual expected volume with a realistic utilisation assumption
(50–60% is optimistic for non-batch workloads), including engineering cost. Then apply the decision rule
that usually holds: **start hosted, move specific high-volume workloads to self-hosted when the arithmetic
and a hard requirement both point that way.** Batch and offline workloads self-host most easily because you
can drive utilisation toward 100% by queueing. Interactive workloads with spiky traffic are the worst
candidates.

And whichever you choose, keep the model interface abstracted (Q47) so this decision stays reversible.

### Q65 — What about small models and edge deployment?

**Mechanism.** Small language models — from a few hundred million to a few billion parameters — have improved
substantially, largely through better data curation and distillation from larger models rather than
architectural revolution. They run on a laptop CPU, a phone, or a modest edge device, and for narrow tasks
they can match much larger models.

**Trade-off.** The honest capability picture: small models are strong at classification, extraction,
summarisation of short text, routing, and simple structured output, especially after fine-tuning on the
specific task. They are weak at open-ended reasoning, long context, complex multi-step tool use, and broad
world knowledge. The failure mode when pushed beyond their range is not graceful degradation — it is
confident nonsense, which is worse operationally than an error.

**Practice.** The pattern where small models win decisively is **high-volume narrow tasks**: classify a
million tickets, extract fields from a million documents, embed a corpus, filter a stream. Here a fine-tuned
small model can be dramatically cheaper and faster than an API call, and quality on the narrow task can be
*better* because it was trained on exactly that distribution.

Edge deployment adds genuine reasons beyond cost: latency (no round trip), offline operation, and privacy
(data never leaves the device — which is a stronger guarantee than any policy). Consider it seriously for
mobile features, on-device assistants, industrial equipment, and anything handling sensitive data where the
best privacy story is architectural rather than contractual. Frameworks for this are maturing —
llama.cpp, MLC-LLM, ONNX Runtime, Core ML and the vendors' mobile stacks — but expect the packaging and
update story to be more work than the inference itself.

---

## 11. Evaluation

### Q66 — How do you evaluate an AI feature before launch?

**Mechanism.** Four layers, each catching what the others miss.

**Golden set** — 100 to 300 real, representative cases sampled from actual traffic or user research, including
edge cases and deliberately adversarial inputs, each with a reference answer or explicit pass criteria. This
is the foundation and the part that takes real work; everything else is machinery around it.

**Automatic metrics** — deterministic wherever possible: schema validity, exact match on extraction fields,
retrieval recall@k, latency, cost per task. Model-as-judge with a written rubric where output is open-ended
(Q68).

**Human review** — a domain expert scores a stratified sample, calibrated against the automated judge so you
know how far to trust the automation.

**Adversarial and safety pass** — prompt injection, PII handling, jailbreaks, out-of-scope requests, and the
worst-behaved user you can imagine (Q96).

**Trade-off.** Building this costs one to three weeks for a first version, which always feels like time
stolen from shipping. The counter-argument is concrete: without it you cannot tell whether a change helped,
which means every subsequent improvement is a guess, and guesses that feel like knowledge are how teams ship
regressions confidently for months.

**Practice.** Set the launch bar **before** looking at results — for example ≥90% groundedness, zero safety
failures, P95 under three seconds, cost per task under a stated ceiling. Deciding thresholds after seeing
numbers is how every result becomes acceptable. Then dogfood internally, ship behind a flag to a small
percentage with online metrics and a rollback plan, and keep the golden set in CI as a regression gate.

Sample the golden set from **real traffic** wherever possible. Sets written from imagination systematically
miss the messy, ambiguous, multi-intent inputs that dominate production and cause most failures.

### Q67 — How do you build an evaluation framework from scratch?

**Mechanism.** Think of it as a pyramid, mirroring software testing.

**Base — unit level, deterministic, every commit.** Schema validation, required fields present, retrieval
recall@k on a fixed query set, citation validity, safety classifiers, format and language checks. Cheap,
fast, unambiguous.

**Middle — task level, every prompt or model change.** The golden set scored against a rubric on the
dimensions that matter for this feature: correctness, groundedness, completeness, tone, instruction
adherence. Reported per slice, compared against the previous version, with cost and latency budgets enforced
as tests that can fail.

**Top — end-to-end and human, pre-release and continuous.** Scenario and trajectory evaluation for agents,
expert review of a sample, adversarial suites, then staged rollout with online A/B and guardrail metrics.

**Trade-off.** Three cross-cutting elements separate a framework from a ritual, and each costs effort. An
**error taxonomy**, so failures are counted by cause (retrieval miss, hallucination, format break, refusal,
tool error) rather than blended into one score — a single number tells you that you got worse, not where to
work. **Per-slice reporting**, because averages hide exactly the failures that generate complaints: a system
at 92% overall can be at 55% in one language or for one customer segment. And a **written definition of
"good"** agreed with stakeholders before anything is built, because otherwise you will litigate it during a
launch review.

**Practice.** Then close the loop: production failures and thumbs-down cases flow back into the golden set. An
evaluation set that does not grow from real failures decays into theatre within a quarter, because it
increasingly tests a distribution your users left behind.

Also version the eval set and rubric explicitly, and record which version produced which score. A metric that
moved because the test set changed and a metric that moved because the system changed look identical in a
dashboard and have completely different responses.

### Q68 — What is LLM-as-judge, and when can you trust it?

**Mechanism.** Using a model to score outputs against a written rubric. It is the only practical way to
evaluate open-ended generation at scale, because human labelling is slow and expensive and you need to
evaluate on every change.

**Trade-off.** Trust it when the criterion is concrete and checkable ("is every claim supported by the
provided context?", "is this valid JSON matching this schema?", "did it answer in the user's language?");
when you have measured agreement with human labels on a calibration set and re-check periodically; when you
use **pairwise comparison** rather than absolute 1–10 scoring, which is markedly more stable; and when the
judge is a *different, strong* model from the one being evaluated.

Do not trust it for factual correctness against ground truth it does not have, for anything requiring domain
expertise it lacks, for subtle safety judgements, or for a final go/no-go decision with no human involved.

The biases are well documented and must be controlled: **position bias** (randomise order in pairwise
comparisons), **verbosity bias** (longer answers score higher regardless of quality), **self-preference**
(models favour outputs resembling their own style), and **rubric drift** as your rubric accumulates
clarifications.

**Practice.** The right mental model is a **smoke alarm, not a courtroom**: it tells you cheaply and
continuously where to look. Calibrate it — take 50 human-labelled examples, measure agreement, and only then
decide how much weight to put on it. Recalibrate when you change the judge model, which is a version change
like any other. And keep a small human-review loop permanently, both as calibration and because the cases
where judge and human disagree are disproportionately informative.

### Q69 — Explain precision, recall and F1 to a non-technical colleague

**Mechanism.** Use a concrete case. Suppose a system flags suspicious transactions.

**Precision** answers: of the ones we flagged, how many were genuinely suspicious? Low precision means we are
bothering honest customers and burning reviewer time — crying wolf.

**Recall** answers: of all the genuinely suspicious ones, how many did we catch? Low recall means fraud gets
through.

You cannot maximise both. Turn sensitivity up and you catch more fraud but flag more innocents; turn it down
and the reverse. **F1** is a single blended score balancing the two, useful mainly for comparing model
versions against each other.

**Trade-off.** F1 is convenient and slightly dishonest, because it implicitly assumes precision and recall
matter equally — which is almost never true. In fraud, a missed case usually costs more than a false alarm.
In an automatically sent customer email, a wrong send costs more than a missed opportunity. Weighted variants
exist for exactly this reason.

Two further cautions. On **imbalanced data** — and fraud, churn and defect detection are all extremely
imbalanced — accuracy is actively misleading: a model that flags nothing is 99.9% accurate and worthless.
And a single threshold hides the whole picture; the **precision-recall curve** shows you the trade-off space,
while AUC summarises ranking quality independent of any threshold.

**Practice.** The conversation that matters is *which error costs more*. Once someone gives you the relative
cost, setting the threshold is arithmetic — and it is a business decision, not a modelling one. Handing the
trade-off back with a number attached, rather than reporting an F1 score, is the single most useful thing you
can do with these metrics. Then revisit the threshold periodically, because the cost ratio changes with
business conditions while the model quietly keeps using the old one.

### Q70 — How do you evaluate something non-deterministic?

**Mechanism.** Stop expecting a single output and evaluate the *distribution* and the *properties*.

**Assert on properties, not strings**: schema validity, presence of required facts, absence of unsupported
claims, correct language, length bounds. Semantic similarity to a reference rather than exact match. **Run
several samples per case** (three to five) and report both pass rate and **variance**. **Fix what you can**:
temperature 0 and pinned model versions in evaluation runs, so you isolate the change you are testing. **Use
rubric scoring and pairwise comparison** for open-ended output. **Aggregate over enough cases** that your
confidence interval is smaller than the difference you are claiming.

**Trade-off.** Variance is itself a metric, and an under-used one: a feature with a good mean and high
variance feels broken to users, because they remember the bad instance. Reporting mean alone systematically
overstates how good a system feels.

The statistical point deserves emphasis because it is the most common evaluation error in the field:
**declaring a 2% win on 50 examples**. With 50 samples, a 2% difference is well inside noise. Either gather
enough cases that the difference is detectable, or report the result as inconclusive — which is a legitimate
and useful finding.

**Practice.** For agents, evaluate **trajectories** and not only outcomes: did it use sensible tools in a
sensible order, recover from errors, stay within budget? A correct answer reached by six wasteful tool calls
is a cost and latency problem hiding inside a passing test.

And use production as the arbiter where you can. **A/B testing** on real traffic is the only unbiased
evaluation you have; offline evaluation is a fast proxy that lets you avoid shipping obvious regressions, not
a substitute for measuring outcomes.

### Q71 — Which evaluation and observability tools should you use?

**Mechanism.** This tooling category matured quickly and the options differ on axes worth knowing:

- **Langfuse** — open source and genuinely self-hostable with no usage limits, which makes it the common pick
  when data residency is a requirement. Tracing, evaluation, prompt management, datasets.
- **LangSmith** — the path of least friction if you build on LangChain or LangGraph, with the deepest
  integration into that stack.
- **Braintrust** — strongest on the CI/CD story: evaluation-gated deployment, generous free tier, oriented
  around treating evals as a build step.
- **Arize Phoenix** — open source, OpenTelemetry-native, with a heritage in classical ML monitoring that
  shows in its built-in metrics and drift detection.
- **Weights & Biases Weave** — natural if you already use W&B for classical ML experiment tracking.
- **MLflow** — has grown genuine LLM tracing and evaluation features, which matters if it is already your
  model registry.
- **Evaluation libraries** rather than platforms: **Ragas** (RAG-specific metrics), **DeepEval**
  (pytest-style assertions), **promptfoo** (config-driven comparison, strong for prompt A/B), **OpenAI Evals**.
  These are code you run, with no service dependency.
- **Gateway-adjacent observability**: Helicone, Portkey, Traceloop/OpenLLMetry — useful when you want logging
  and cost tracking at the proxy layer rather than instrumenting the application.

**Trade-off.** Platforms give you tracing, datasets, evaluation and a UI in one place, at the cost of a
dependency in your critical path and, for hosted options, your prompts leaving your boundary. Libraries give
you full control and no dependency, and you build the UI, storage and workflow yourself. OpenTelemetry-based
options are the most portable bet if you expect to switch.

**Practice.** Decide on three questions. Can prompts and outputs leave your infrastructure? If not, that
narrows you to self-hostable options immediately. Do you want evaluation **in CI** (a library or a
CI-oriented platform) or **as a workflow** (a platform with a UI for non-engineers)? And do you already have
an observability standard — because emitting OpenTelemetry traces into the stack you already run often beats
adding a parallel system. Start with tracing before evaluation: you cannot evaluate what you cannot see, and
tracing pays for itself on the first incident.

### Q72 — Offline evaluations pass but users are unhappy. Now what?

**Mechanism.** This is the most common real situation, and the answer is a diagnosis order rather than a fix.

First, establish what "unhappy" means: which segment, which surface, since when, and what exactly do they
say? Then **read fifty real transcripts** before touching anything. That single step resolves most cases, and
it is skipped almost universally in favour of theorising.

Then work the usual suspects in likelihood order:

- **Evaluation-set mismatch** — your golden set does not represent real traffic: missing intents, missing
  languages, messier inputs, longer conversations. This is the most likely cause by a wide margin.
- **Wrong metric** — you measured correctness; users care about latency, tone, verbosity, effort saved, or
  trust. A technically correct answer that takes twelve seconds and needs editing is a failed answer.
- **Distribution shift** — real inputs are dirtier and broader than curated ones.
- **Integration reality** — the answer is right but arrives at the wrong point in the workflow, or takes more
  clicks than doing it manually.
- **Expectation mismatch** — the interface or announcement promised more than the system delivers. Sometimes
  the fix is copy, not the model.
- **Tail failures** — the mean is fine, the P95 is terrible, and people remember the P95.

**Trade-off.** The instinct is to improve the model. The evidence usually points at the eval set, the metric,
or the workflow — none of which are model problems, all of which are cheaper to fix, and all of which are less
satisfying to work on.

**Practice.** Fix the evaluation set from real failures **first**, so you can measure progress at all. Add the
missing metric. Ship a targeted improvement. Verify with an A/B rather than by vibes. Then build the standing
pipeline from production complaints into evaluation cases — because this situation is proof that it did not
exist, and without it you will be here again next quarter.

---

## 12. Observability, MLOps and LLMOps

### Q73 — How is MLOps different from DevOps, and what does LLMOps add?

**Mechanism.** **MLOps** is the practice of taking machine-learning systems to production and keeping them
healthy: versioning, automated training and deployment, monitoring, governance, and the feedback loop into
improvement.

It differs from DevOps in four structural ways. A software system has one thing that can change — code —
whereas an ML system has three: **code, data and model**, all of which must be versioned together for a
result to be reproducible. **Testing is statistical**: software passes or fails, a model is better or worse,
so your quality gate is a threshold on a metric rather than a green tick. **Deployment needs shadow modes and
canaries**, because you cannot fully predict behaviour on live traffic. And most distinctively, **ML systems
degrade without anyone changing them**, because the world moves and the training distribution stops matching
reality (Q75).

**LLMOps** adapts this for systems built on foundation models, where you usually are not training the model.
What it adds: **prompts as versioned artefacts** with change history and rollback (Q12); **evaluation
replacing accuracy metrics**; **a vendor as your core dependency**, making version pinning, deprecation
timelines and rate limits operational concerns; **per-request variable cost**, putting cost observability
alongside latency; **context assembly** as a first-class tested component; and **guardrails** in the runtime
path rather than as an afterthought.

**Trade-off.** What mostly disappears if you consume an API: GPU capacity planning, distributed training,
hyperparameter search. What emphatically does not: monitoring, incident response, rollback, lineage and
governance. Teams that assume "we're just calling an API, so we don't need MLOps" discover the second list
during their first incident.

**Practice.** The maturity ladder runs roughly: manual notebooks → scripted reproducible runs → automated
pipeline with a model registry → monitoring-triggered retraining → full governance with lineage and
approvals. Most organisations overestimate where they are by about two rungs. The honest self-assessment is
Q74's test.

### Q74 — What do you version, and how do you know if you're doing it properly?

**Mechanism.** Everything that can change the output, tied together so any result can be reproduced:

- **Code** — application, pipelines, prompts, tool definitions, schemas. In git, reviewed.
- **Data** — training and evaluation datasets, with a snapshot or content-addressed reference. "The customers
  table" is not a version; "the customers table as of this commit or timestamp" is. Tools like DVC, LakeFS or
  table-format time travel exist for this.
- **Model** — for your own models, a registry entry with training code commit, dataset version,
  hyperparameters, metrics and lineage (MLflow, W&B, or a cloud registry). For vendor models, the **pinned
  version string** — never a floating `-latest` alias in production.
- **Prompts** — as files, with a version identifier logged on every request.
- **Retrieval configuration** — embedding model version, chunking parameters, index build ID, top-k,
  re-ranker settings.
- **Evaluation sets and rubrics** — versioned, so a moved metric is attributable.

**Trade-off.** Full versioning discipline has real overhead and slows experimentation, which is why it gets
deferred. The overhead is front-loaded and the payoff is entirely in incidents and audits — which makes it
easy to skip right up until the moment it is expensive not to have.

**Practice.** Here is the test: **pick a request from production logs three weeks ago and ask whether you
could reproduce it exactly.** Same prompt version, same model version, same retrieved documents, same
configuration. If not, your incident investigations are guesswork and your regression analysis is opinion.

Start with the cheapest high-value items: pin model versions, put prompts in git with a logged version ID, and
log retrieval configuration and document IDs with every request. That is a day of work and covers most of
what you will actually need.

### Q75 — What is drift, and how do you detect it before users do?

**Mechanism.** Drift is divergence between the world the system was built for and the world it now operates
in. Three kinds:

**Data drift (covariate shift)** — the input distribution changes: new customer segments, a new product line,
a new locale, seasonality, an upstream schema change. **Concept drift** — the *relationship* between inputs
and correct outputs changes: the same input should now produce a different answer because behaviour, prices,
regulation or fraud tactics changed. **Prediction drift** — your output distribution shifts, often the first
observable symptom because it requires no labels.

In LLM systems the same idea appears as **prompt drift** (the inputs users send evolve away from what you
designed for — a genuinely under-monitored failure) and **silent model drift** (the vendor updated the model
behind your alias).

**Trade-off.** Detection is constrained by what you can observe. Input monitoring — per-feature distribution
tests, null rates, range violations, categorical novelty — needs no labels and catches a great deal. Output
monitoring tracks prediction or score distributions. **Performance monitoring is the real thing but needs
ground truth**, which often arrives late: you learn whether a customer churned three months later. Proxy
signals fill the gap: user corrections, retries, escalations, abandonment.

The tension is between sensitivity and alert fatigue. Distribution tests on many features will fire
constantly on normal variation, and a drift alerting system that cries wolf gets muted, at which point you
have the cost and none of the benefit.

**Practice.** Monitor a small number of features that actually matter to the model rather than all of them.
Alert on **rate of change** rather than absolute thresholds. Track proxy signals as first-class metrics
because they arrive immediately. For LLM systems specifically, cluster incoming requests periodically and
compare against your evaluation set distribution — that is how you catch prompt drift, and it is a monthly
job rather than a real-time alert.

### Q76 — What does good monitoring look like for an AI feature?

**Mechanism.** Four layers; skipping any one leaves a specific blind spot.

**Infrastructure** — latency percentiles (P50, P95, P99), error rates, timeouts, throughput, dependency
health. Standard, necessary, insufficient.

**Cost** — tokens and spend per request, per feature, per customer; cache hit rate; and **cost per successful
task**, which is the honest figure because it includes retries and failures.

**Quality proxies** — the layer teams forget. A wrong answer throws no exception, so you must instrument for
it deliberately: a sampled model-judged score on live traffic, output validation failure rate, refusal rate,
retrieval hit rate, citation validity, and behavioural signals (did the user edit the output, regenerate,
retry, escalate, or abandon?).

**Business outcome** — task completion, time saved, deflection rate, conversion.

Underneath all four, **traces**: for any single interaction you should be able to reconstruct inputs, prompt
version, retrieved document IDs and scores, every tool call with arguments and results, model version, token
counts, per-step latency, final output, and any guardrail trigger.

**Trade-off.** Full tracing has cost — storage, and a sampling decision for high-volume systems — and
sampling is where people get it wrong by sampling uniformly. Sample *errors and outliers* at 100% and normal
traffic at a low rate: the interesting traces are rare by definition.

**Practice.** The concrete test: **can you fully reconstruct one user's complaint from your logs?** If not,
you cannot debug the product, and you will resolve incidents by guessing.

Then decide in advance which failures must be **loud** (wrong data, leaked data, wrong action taken) and which
can be **quiet** (a mediocre suggestion). Treating them identically produces either alert fatigue or missed
severity, and both are worse than choosing.

### Q77 — How do you deploy a change to a production AI feature safely?

**Mechanism.** Treat every change — including a prompt edit and a model version bump — as a release.

**Before.** Run the golden set on old and new side by side, reported per slice and per error type rather than
as one blended average. Re-run adversarial and safety suites. Measure latency and cost per task.

**Rollout.** Shadow mode first: run the new configuration on real traffic without serving it, and compare
offline. Then a small canary (5% is typical) behind a flag, with automated guardrail metrics and an explicit
rollback trigger defined in advance. Then ramp in stages. Keep the flag and the ability to revert instantly.

**After.** Watch a fixed window of online metrics — thumbs-down rate, regeneration rate, escalation rate, task
completion, latency, cost — against the pre-change baseline rather than against expectations.

**Trade-off.** This is slower than editing a prompt and shipping, and the slowness is the point. The
counter-pressure is real though: a full release process for every prompt tweak will be circumvented, so
calibrate — trivial copy changes can go through a light path, anything touching behaviour, tools or model
version goes through the full one.

**Practice.** The point that specifically catches people out on model upgrades: **prompts are
model-specific**. "Same prompt, new model" is a different system, not the same system improved. Expect to
retune, and expect the change to be uneven — better reasoning alongside worse instruction-following,
different verbosity, changed JSON habits, shifted refusal boundaries. Budget time for it in the upgrade plan
rather than discovering it in production, and evaluate **per slice**, because a model upgrade that improves
the average while degrading one language or customer segment is a common and easily-missed outcome.

### Q78 — How do you handle an AI incident?

**Mechanism.** AI incidents differ from ordinary outages in one crucial way: **the system is usually up**. It
is answering, quickly and confidently — just wrongly. Detection is therefore the weak point, and the response
must begin by establishing what is actually happening.

A workable order:

1. **Confirm it's real.** Check volume and slices; a metric that moved because the traffic mix changed is not
   a quality regression.
2. **Establish what changed and when.** Correlate against deploys, prompt edits, model version changes
   (including silent vendor updates), index rebuilds, config changes and dependency updates. Most
   regressions have a change behind them.
3. **Localise in the pipeline.** Use traces: retrieval hit rate down? tool error rate up? latency and
   timeouts up? refusal rate up? output validation failing? Each points at a different layer.
4. **Localise by slice.** One language, one customer, one query type, one platform.
5. **Read the failures.** Twenty or thirty real bad traces beat an hour of dashboard staring.
6. **Mitigate before you explain.** Revert, route to the previous model, or fall back to a deterministic
   path. A working product beats a perfect root-cause narrative.
7. **Then root-cause, and close the loop.** Add the case to the evaluation set and add the missing alert.

**Trade-off.** Mitigating first means you sometimes lose the evidence needed to root-cause. Mitigate anyway,
but capture a sample of failing traces before you revert — five minutes of preservation saves a day of
speculation.

**Practice.** Decide in advance what your **fallback** is for each feature, and test it: previous model
version, cached response, deterministic non-AI path, or a graceful "unavailable" message. An untested
fallback is a plan, not a capability. And write the postmortem with the same seriousness as for an outage:
silent quality failures are easy to under-weight precisely because nothing paged anyone.

---

## 13. Cost and FinOps for AI

### Q79 — How do you estimate what an AI feature will cost?

**Mechanism.** Show the method, state assumptions, sanity-check against the business. A worked example:

1. **Active users:** 10M registered × 20% monthly active = 2M active.
2. **Usage:** 5 AI interactions per active user per month = 10M calls/month.
3. **Tokens per call:** system prompt + retrieved context + history ≈ 3,000 in; 400 out.
4. **Volume:** ~30 billion input and ~4 billion output tokens per month.
5. **Price:** at, say, $1/M input and $5/M output → $30k + $20k ≈ **$50k/month**.
6. **Adjust for reality:** +15% for retries and failed generations; −60% of input cost through prompt caching
   of the stable prefix; −70% on the easy 80% of traffic by routing to a cheaper model. Landing zone:
   **$15–25k/month**.
7. **Sanity-check:** ~$0.005–0.01 per call, ~$0.01 per active user per month. Against $10 ARPU that is ~0.1%
   and unremarkable. Against $0.30 ad-supported ARPU it is a business-model problem — and *that* is the
   finding worth escalating.

**Trade-off.** Every step has an error bar, and they compound. The dominant uncertainty is almost always
**interactions per user**, which is a product-behaviour question nobody can answer before launch. Estimates
are therefore for deciding whether the order of magnitude works, not for budgeting to two decimal places.

**Practice.** Name which assumption dominates and what you would measure first to tighten it. Track **cost
per successful task**, not per call — retries and failures are real spend and the per-call number flatters
you. Model the pessimistic case explicitly: what if usage is 3× the estimate and caching does not work as
well as hoped? And instrument cost per feature from day one, because retrofitting attribution across a shared
gateway later is genuinely painful.

### Q80 — What is model routing, and why does it decide affordability?

**Mechanism.** Routing sends each request to the cheapest model that can handle it, rather than everything to
the strongest. The router can be rules-based (task type, surface, user tier, input length), a small trained
classifier, embedding similarity to examples of known difficulty, or a **cascade** — try the small model,
escalate on low confidence or failed validation.

It matters because the difficulty distribution of real traffic is heavily skewed: typically 70–90% of
requests are easy — classification, extraction, short summarisation, formatting, reformulation — and paying
frontier prices for those is pure waste. Routing commonly cuts total inference cost several-fold *and*
improves median latency, at a small measurable quality cost.

**Trade-off.** The router must be far cheaper and faster than the models it routes to (single-digit
milliseconds, negligible cost, or it eats the savings), accurate enough that misroutes are rare and
recoverable, and deliberately **asymmetric**: sending a hard query to a small model produces a bad answer,
while sending an easy query to a big model merely wastes money. Bias toward escalation.

The hidden cost is complexity: a routed system has more paths, more failure modes, and a quality distribution
that varies by route. Cascades add latency on escalation, so the worst case is *slower* than always using the
big model.

**Practice.** Log every routing decision with its eventual outcome, so the router itself becomes trainable
from production data. Provide a manual override. Keep a static fallback for when the router is unavailable.
And **monitor quality per route** — degradation hides in the cheap path, where nobody is looking, and shows up
as a slow erosion of user trust rather than an alert.

### Q81 — What are the real levers on cost and latency?

**Mechanism.** In rough order of leverage per unit of effort:

**Prompt caching.** Cache the stable prefix — system prompt, tool definitions, long shared context. Often the
single largest win with no quality trade-off. It requires the prefix to be byte-identical, so remove
timestamps and per-request IDs from the front of prompts.

**Routing and cascading** (Q80). **Semantic caching** — many workloads, internal support especially, are
highly repetitive; serving near-duplicate queries from cache costs almost nothing. **Token discipline** —
trim the system prompt, retrieve fewer and shorter chunks, summarise history rather than replaying it, cap
output length. **Batch APIs** for anything non-interactive, typically around half price. **Streaming** —
does not reduce real latency but transforms *perceived* latency. **Moving work off the critical path** —
precompute, run asynchronously, generate on write rather than on read. **Smaller or fine-tuned models** for
narrow high-volume tasks.

**Trade-off.** Each lever has a catch. Caching requires prompt discipline that fights dynamic context.
Semantic caching risks serving a stale or subtly wrong answer to a similar-but-different question, so it
needs a similarity threshold you have actually validated. Batch trades latency for price. Token trimming can
quietly remove context the model needed.

**Practice.** Pick the binding constraint from the use case rather than optimising everything: interactive
assistance is latency-bound, high-volume enrichment is cost-bound, high-stakes low-volume analysis is
quality-bound. Then attack that one. The most under-used lever is the last one — **deleting the wait**. The
strongest latency optimisation is frequently not making the request faster but arranging for the user not to
be waiting for it.

### Q82 — How do you keep AI spend under control operationally?

**Mechanism.** Cost control needs four mechanisms, and most teams have only the first.

**Visibility** — cost attributed per feature, per team, per customer, ideally per request. Without this, spend
is a single number nobody owns. **Budgets and limits** — hard caps per API key, per feature, per tenant,
enforced at the gateway (Q46) rather than hoped for. **Anomaly alerting** — alert on rate of change, because
the characteristic AI cost incident is not a gradual rise but a loop that burns a month's budget in an
afternoon. **Unit economics tracking** — cost per successful task, trended, so efficiency work is visible.

**Trade-off.** Hard limits protect the budget and degrade the product when hit, so the failure behaviour needs
designing: reject, queue, downgrade to a cheaper model, or serve a cached answer. Choosing "reject" silently
is how a cost control becomes an outage.

**Practice.** Three specific guards worth having from the start. **Per-request ceilings** on tokens, tool
calls and steps, because agentic loops are the main way spend runs away. **Idempotency and retry caps**, since
retries multiply cost invisibly. And **a kill switch per feature**, so you can stop a runaway without
deploying.

Also review the **unnecessary spend** categories periodically: bloated system prompts paid on every call,
retrieval returning more context than the model needs, verbose output nobody reads, evaluation runs against
expensive models where a cheap one would do, and development traffic hitting production-tier models. These
are usually 20–40% of a mature deployment's bill and require no quality trade-off to fix.

### Q83 — Models improve constantly. What actually holds its value?

**Mechanism.** The model is the fastest-depreciating component in the stack. What holds value:

**The problem and the workflow** — what people are trying to get done changes on a scale of years.
**Proprietary data and feedback loops** — labelled examples, preference data, outcome data. This compounds and
cannot be bought. **Your evaluation harness** — golden sets, rubrics and the harness itself are what let you
adopt each new model in days instead of quarters; the most underrated durable asset in applied AI.
**Context and orchestration architecture** — retrieval quality, tool design, guardrails, observability.
**Distribution and trust** — being where the work already happens. **Domain expertise encoded as product** —
the taxonomy, the policy, the written definition of "good".

**Trade-off.** Investing in these while models improve underneath you feels like building on sand, and the
temptation is to wait for the landscape to settle. It will not settle, and the teams that treat each model
release as a rewrite spend their entire capacity on migration.

**Practice.** The architectural rule: **abstract the model, invest in everything around it.** Concretely — a
model interface with no provider assumptions leaking upward; pinned versions plus a regression evaluation as
the standard swap procedure; prompts as versioned assets; and a standing habit of re-benchmarking whenever a
new model ships.

Done properly, a new model release becomes a half-day exercise: run the eval set, compare per slice, retune
prompts where needed, canary, ramp. That turns model improvement into a **dividend** rather than a project —
and the difference between those two outcomes is almost entirely whether the eval harness exists.

---

## 14. Automation tooling

### Q84 — What is n8n, and what are the honest alternatives?

**Mechanism.** n8n is a workflow automation platform: flows are built as a visual graph of nodes, where each
node is a trigger (webhook, schedule, an event in an app) or an action (call an API, transform data, branch,
loop, write to a database). Two traits distinguish it in its category: it is **source-available and
self-hostable**, and you can drop into JavaScript or Python in a code node when the visual abstraction runs
out. For AI work it has nodes for the major model providers, vector stores, embeddings and agent
abstractions, so an LLM automation is buildable in an afternoon.

**Trade-off.** The alternatives are genuinely different rather than worse, and the landscape splits into four
groups:

**Hosted no-code integration platforms.** **Zapier** — by far the largest connector catalogue (thousands of
apps), easiest for non-technical users, priced per task, deliberately shallow on complex logic. **Make**
(formerly Integromat) — a more visual and more capable branching and iteration model at a lower price point,
steeper learning curve. **Microsoft Power Automate** — the default if you live in Microsoft 365, with deep
Office, Teams, SharePoint and Dataverse integration plus desktop RPA; strongest inside that ecosystem and
awkward outside it. **Workato** and **Tray** occupy the enterprise-integration end with governance and support
to match.

**Open-source / self-hostable.** **n8n** itself. **Activepieces** — MIT-licensed, AI-first, a genuine
open-source Zapier alternative. **Windmill** — code-first: steps are written in Python, TypeScript, Go, Bash
or SQL rather than assembled from nodes, which suits engineering teams who find node graphs limiting.
**Pipedream** — more code-first than n8n, developer-oriented, hosted with a generous free tier. **Kestra** —
declarative YAML workflows, positioned between automation and data orchestration. **Node-RED** — the
long-standing option, strong in IoT and event routing.

**AI-native builders.** **Flowise**, **Langflow**, **Dify** — visual builders specifically for LLM
applications (chains, agents, RAG) rather than general app-to-app integration.

**Data and workflow orchestrators.** **Airflow**, **Dagster**, **Prefect**, **Temporal** — built for
reliability, retries, backfills, scheduling and durable long-running execution, not for quick glue.

**Practice.** n8n's real strengths are self-hosting, execution-based rather than per-task pricing, and the
code escape hatch. Its real limits: complex logic becomes hard to read as a graph, testing and version control
are weaker than for ordinary code, debugging a long-running flow is fiddlier than reading a stack trace, and
self-hosting means you now operate a service with a queue, a database and upgrades. Those limits apply to
every visual tool in the list; the differences between them are about hosting model, pricing shape,
integration breadth and how far you can escape into code.

### Q85 — How do you choose among automation platforms?

**Mechanism.** Four questions settle it most of the time, in this order.

**Who maintains it?** If the answer is "a business team, not engineering", a visual no-code tool wins even if
it is less capable — an unmaintainable elegant solution is worse than a maintainable clumsy one.

**Where must the data live?** If it cannot leave your infrastructure, that eliminates hosted-only platforms
and points at self-hosted n8n, Activepieces, Windmill or your own code.

**How critical is it?** For something that must not silently fail — invoicing, compliance reporting,
customer commitments — you want the reliability guarantees of a real orchestrator (Temporal, Dagster,
Airflow) or ordinary tested code, not a visual flow whose failure mode is an email nobody reads.

**How complex is the logic?** Visual builders are excellent up to roughly a dozen nodes with light branching.
Beyond that the graph becomes less readable than equivalent code, and you lose diffs, review and unit tests.

**Trade-off.** There is a real tension between **iteration speed** and **rigour**, and both matter. The
mistake in one direction is engineering a bulletproof pipeline for something that should have been a
five-node flow owned by the operations team. The mistake in the other is running a revenue-critical process
in a visual tool with no tests, no staging environment and no owner.

**Practice.** A pattern that works: **prototype in a visual tool** to prove value and learn the real
requirements, then **rewrite as tested code** the two or three flows that became business-critical, and leave
the long tail in the visual tool where iteration speed matters more than rigour. Review annually and expect
some flows to graduate and others to be retired.

### Q86 — What should you automate first, and what should you leave alone?

**Mechanism.** Good first candidates share four properties: the task is **frequent** (savings compound),
**rule-shaped or judgement-light**, has a **tolerant failure mode** (a wrong draft is annoying, not
catastrophic), and has a **clear trigger and a clear finish**. Triage and routing, drafting first versions,
summarising and extracting from documents, enrichment and data entry, monitoring and alerting, and report
assembly all qualify.

**Trade-off.** Leave alone, at least initially: anything irreversible without confirmation; anything where
failure is silent and expensive; anything requiring judgement the organisation has not written down (if two
experts disagree on the right answer, automation encodes whichever one wrote the prompt); anything with a
regulatory explainability requirement; and anything so rare that maintaining the automation costs more than
doing it manually.

**Practice.** Two failure patterns cause most disappointment, and both are organisational rather than
technical.

**Automating a broken process.** Automation amplifies whatever the process already is, so a confusing
approval chain becomes a confusing approval chain that runs faster and is harder to inspect. Fix or at least
map the process first; the mapping frequently reveals that half the steps exist for reasons that no longer
apply.

**Automating away the review step** because the automation seems reliable. The review step is often what keeps
it reliable, and removing it should be a measured decision with a defined threshold (Q98), not a drift.

The most durable automations keep a human decision point at the end. People act on output they can check, and
quietly stop using output they cannot.

### Q87 — How do you make an automation reliable?

**Mechanism.** The difference between a demo automation and one people trust is almost entirely error
handling:

- **Idempotency.** Every write needs an idempotency key so a retry cannot double-charge, double-send or
  double-create. This is the single most common gap and the most expensive one.
- **Retries with backoff**, capped. Distinguish retryable failures (timeout, rate limit, 5xx) from permanent
  ones (validation error, permission denied) — retrying the latter wastes time and pollutes logs.
- **Explicit failure paths.** Decide what happens when a step fails: dead-letter queue, alert a human,
  degrade gracefully, or halt. A flow with no failure path fails silently, which is the worst option.
- **Validation between steps.** Validate the shape of data crossing every boundary rather than hoping.
- **Loop and cost protection.** Maximum steps, maximum tokens, maximum spend per run — automations that call
  models can burn a budget quickly.
- **Observability.** Run history, structured logs, and alerting on rate of change rather than absolute counts.
- **A tested rollback or manual override**, so a human can take over mid-process.

**Trade-off.** All of this roughly doubles the build time of a naive automation, which is exactly why it gets
skipped, and exactly why so many automations are quietly distrusted by the teams they were built for.

**Practice.** One process rule matters as much as the technical list: **give every automation a named owner
and a review date.** Unowned automations break silently when an upstream API changes, and nobody notices until
a customer does. A quarterly review that retires dead flows and re-verifies critical ones costs an hour and
prevents the "we have 200 automations and nobody knows which still work" state that mature deployments reach.

### Q88 — Where do AI agents fit into automation, versus deterministic flows?

**Mechanism.** A deterministic flow does the same steps every time. An agentic step decides what to do based
on what it sees. Most valuable automations are **mostly deterministic with a narrow model-powered decision**:
the flow is fixed, and the model classifies, extracts, drafts or routes inside it.

**Trade-off.** Putting a model in the control flow — letting it decide the sequence — buys flexibility and
costs predictability, cost control and debuggability (Q39). In an automation context this trade is usually
worse than in an interactive product, because automations run unattended: nobody is watching to notice that
the agent took a strange path, and the failure is discovered downstream.

**Practice.** The reliable pattern is **deterministic flow, model-powered steps**. Use the model where the
input is unstructured (classify this email, extract these fields, draft this reply, decide which of these
four queues) and keep the sequencing, the policy and the writes in ordinary flow logic with validation between
steps.

Where you do want agency in an unattended automation, bound it hard and add a **human checkpoint before any
external effect**. An agent that researches and drafts autonomously, then queues its output for one-click
human approval, captures most of the value with a fraction of the risk — and it is the shape that survives
contact with a compliance review.

### Q89 — What is RPA, and is it obsolete now?

**Mechanism.** Robotic process automation drives *user interfaces* rather than APIs: it clicks buttons, fills
forms and reads screens, mimicking a human operator. UiPath, Automation Anywhere, Blue Prism and Power
Automate Desktop are the established platforms.

It exists because a great deal of enterprise software has no usable API — legacy systems, vendor
applications with locked-down integration, mainframe terminals, desktop applications. If you cannot call it,
you drive it.

**Trade-off.** RPA is famously brittle: a UI change breaks the automation, and the failure is often silent or
produces subtly wrong data rather than an error. Maintenance cost is the well-documented reason many RPA
programmes disappointed. It is also slow compared to an API call, and it typically requires a licensed
desktop session per concurrent robot.

Against that: it is sometimes the only option, and "the only option" is a strong argument.

**Practice.** RPA is not obsolete but its scope has narrowed usefully, and vision-capable models have changed
the picture in two directions. They make **UI automation more robust**, because an agent that can look at a
screen and find the button is less brittle than a script keyed to pixel coordinates or DOM paths — this is the
basis of computer-use agents. And they make **the alternative more attractive**, because extracting structured
data from a screenshot or PDF no longer requires the rigid templates that pushed people toward RPA.

The decision order is unchanged and worth stating: **API if one exists; database access if that is
permitted; file exchange if that is offered; UI automation only when the first three are genuinely
unavailable.** And if you are choosing UI automation, budget for maintenance as an ongoing cost rather than a
one-off build.

---

## 15. The model and tool landscape

> **Snapshot marker.** This section reflects mid-2026 and is the fastest-moving part of the guide. Treat it as
> a map of positions and criteria, not a current price list.

### Q90 — What are the main model families, and how do they differ?

**Mechanism.** The Western frontier families and their centres of gravity:

- **Anthropic — Claude.** Strongest reputation for coding, long-horizon agentic work and long-context
  reliability, with a mature tool-use surface (and the origin of MCP). The current line spans a
  top-capability tier down to a fast inexpensive tier, which is exactly the spread routing exploits.
- **OpenAI — GPT / ChatGPT.** The broadest consumer reach and ecosystem, strong general capability, extensive
  multimodality, and the largest third-party integration surface. Codex is its coding-agent line.
- **Google — Gemini.** Deep Google Cloud and Workspace integration, strong multimodal and long-context work,
  with Antigravity as the developer surface (Q92).
- **xAI — Grok.** Positioned on real-time data access and a distinct conversational register, integrated with
  X.
- **Meta — Llama.** The most widely adopted Western open-weight family and the common starting point for
  self-hosting and fine-tuning.
- **Mistral.** European, open-weight-friendly, efficient, strong multilingual — frequently chosen in the EU
  for data-residency reasons.
- **Also relevant**: Cohere (enterprise retrieval and reranking focus), AI21, and the cloud providers'
  own model lines.

**Trade-off.** The practical point is not which vendor is best. It is that **tiers within a family differ
more than families differ from each other**, so the choice that actually affects your cost and latency is
which tier for which task, not which logo. Vendor choice is driven more by ecosystem fit, data-residency
options, enterprise contracting and support than by benchmark deltas that will invert next quarter.

**Practice.** Pick a primary for depth of integration and a secondary for failover and price comparison, and
keep the interface abstracted (Q47). Re-benchmark on your own eval set when a new tier ships rather than
reading launch posts. And treat "which vendor" as a procurement and architecture question, "which tier" as a
per-task engineering question.

### Q91 — What is the state of Chinese open-weight models, and why does it matter?

**Mechanism.** By mid-2026 several Chinese families ship frontier-class models, and most publish open weights
— which is what makes them strategically significant wherever you are:

- **DeepSeek** — built its reputation on strong reasoning and coding at dramatically low cost; a common
  default when you want a capable generalist cheaply.
- **Alibaba's Qwen** — the most-downloaded open model family globally and the most adaptable fine-tuning base,
  with unusually broad multilingual coverage and a very wide size range.
- **Moonshot's Kimi** — targets long context and long-horizon agentic work; its K-series releases have been
  among the largest open models published.
- **Zhipu's GLM** — frequently cited as the strongest open family for coding and agentic tool use.
- **MiniMax** — efficient long-context architectures using sparse attention, aimed at very large contexts at
  far lower compute cost than a standard transformer.
- **Xiaomi's MiMo** — smaller, efficiency-focused, targeting reasoning and coding at modest size.

**Trade-off.** Three reasons this matters practically. **Cost pressure**: open weights plus aggressive pricing
compressed the price of "good enough" capability across the whole market, including Western pricing.
**Self-hosting**: open weights let you run capable models inside your own boundary, resolving data-residency
questions no API contract fully resolves. **Due diligence**: open weights do not mean open training data,
licences vary meaningfully between families and versions, and if you use a *hosted* Chinese API rather than
self-hosted weights, the data-governance question is real and your legal and security functions must answer
it explicitly rather than by omission.

**Practice.** Evaluate them the way you evaluate anything else — on your eval set, in your language, on your
task. Read the actual licence for the actual version you intend to deploy, because "open" spans genuinely
permissive to restricted-use. And separate the two decisions cleanly: *using the weights on your own
infrastructure* and *sending data to a hosted endpoint* have completely different risk profiles and should
not be discussed as one thing.

### Q92 — Which AI coding tools exist, and how do they differ?

**Mechanism.** The category has diversified into four shapes, and the distinctions matter more than the
rankings:

**IDE plug-ins** — **GitHub Copilot** is the largest by adoption and the path of least resistance if you live
in VS Code or JetBrains and do not want to change editor. **Sourcegraph Cody** and **Continue** (open source,
bring-your-own-model) are alternatives with different strengths in codebase context and configurability.

**AI-first editors** — **Cursor**, a VS Code fork built around AI with a mature multi-file editing
experience; **Windsurf**, with its own take on agentic flows; **Zed**, performance-focused with collaborative
and AI features.

**Terminal / agentic CLIs** — **Claude Code**, **OpenAI Codex CLI**, **Google's Antigravity CLI** (Q93),
**Aider** (the long-standing open-source option, model-agnostic), **OpenHands** (formerly OpenDevin), and
**Cline** (open source, bring-your-own-key, which is the pragmatic choice when you need cost predictability or
must use local models).

**Autonomous / asynchronous agents** — tools that take an issue and produce a pull request, including the
background agent modes now offered by several of the above.

**Trade-off.** The axes that actually separate them: **where they run** (editor, terminal, cloud), **model
flexibility** (locked to one vendor versus bring-your-own-key, which decides whether you can use local models
or negotiate pricing), **permission model** (how much filesystem and shell access, and what confirmation
gates exist), **repo-scale context** (how well they find relevant code in a large codebase), and **team
features** (shared configuration, policy, audit).

**Practice.** Assistants accelerate your keystrokes; agentic CLIs take whole tasks. The observed pattern among
productive developers is not picking one but using two or three at different moments — an editor assistant for
flow-state work, an agentic CLI for well-specified multi-file tasks, and occasionally an async agent for
routine changes.

Two organisational cautions. **Filesystem and shell access** is what makes CLIs powerful and is a real
security decision — sandboxing, credential scoping and confirmation gates deserve deliberate configuration,
not defaults. And the teams that get value keep **review discipline**: the constraint has shifted from
writing code to reviewing it, and a team that merges agent output unread has automated the wrong half of the
job.

### Q93 — What happened to the Gemini CLI, and what is Antigravity?

**Mechanism.** **Antigravity** is Google's agent-first development platform. It launched in November 2025 as
an agent-oriented IDE (a VS Code fork) alongside Gemini 3, and at Google I/O in May 2026 became
**Antigravity 2.0**: a standalone desktop application plus an **Antigravity CLI**, an SDK, managed agent
execution in the Gemini API, and enterprise packaging.

Two things are notable beyond the branding. The architectural stance is that the primary abstraction is no
longer the editor but **orchestration of teams of agents working in parallel** — the editor is still present
and deliberately not the centre. And the **Antigravity CLI is written in Go**, replacing the Node.js-based
Gemini CLI, with faster startup and lower memory as the stated motivation.

**Trade-off.** The migration detail that matters operationally: Google retired consumer access to the
**Gemini CLI** and the Gemini Code Assist IDE extensions on **18 June 2026**, with Antigravity CLI as the
successor. If you have scripts or CI depending on the old CLI, that is a real migration rather than a rename.

More broadly, this is a good example of a general risk: developer tooling in this space is being replaced
faster than tooling normally is, and anything you wire into CI is exposed to it.

**Practice.** Treat AI developer tooling as a dependency with a shorter-than-usual expected lifetime. Keep
CI integrations thin and behind a wrapper, so replacing the underlying tool is a one-file change. Pin
versions. And when adopting a tool that a vendor has just repositioned, check what it replaced and what
happened to the users of the predecessor — the answer tells you something about what to expect next time.

*Sources for this answer:* [Google Developers Blog](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/),
[Antigravity at I/O 2026](https://antigravity.google/blog/google-io-2026).

### Q94 — How do you choose a model, and what actually differs in practice?

**Mechanism.** A repeatable process beats brand preference:

1. **Characterise the task** — routine or hard reasoning, long-context needs, multimodality, tool use, output
   format, language coverage.
2. **List non-functional constraints** — P95 latency, volume and cost ceiling, data residency and compliance,
   rate limits and capacity.
3. **Build the evaluation set first** — 100–300 real examples with explicit pass criteria, *before* choosing.
4. **Probe the ceiling** with the strongest model available: is the task solvable at all? That tells you
   whether you have a model problem or a system problem.
5. **Optimise downward** — smaller model, better prompt, retrieval, caching, possibly fine-tuning — while
   scores stay above threshold.
6. **Route** rather than picking one model for everything (Q80).
7. **Keep an exit** — an abstraction layer plus a regression evaluation.

**Trade-off.** Beyond headline capability, the differences that change your implementation are:
**instruction-following consistency** (some models follow long prompts very literally, others generalise —
and this is the most common source of regression when switching); **tool-use and structured-output
reliability**, which varies more than general capability and dominates agentic workloads; **verbosity and
tone defaults**, materially different in both feel and token cost; **refusal boundaries**, decisive for work
near sensitive domains; **long-context behaviour**, where two models with the same advertised window can
differ substantially; **language coverage**, where performance in Hungarian or any lower-resource language is
often much further behind English than English benchmarks suggest; and **version stability**.

**Practice.** The closing point that saves the most wasted effort: **public benchmarks do not transfer.** The
only number that matters is your evaluation on your traffic with your constraints attached. A model leading a
leaderboard can lose badly on your task, and you will only find out by measuring.

Test in the language you ship in. Test structured output and tool calling explicitly. And when you switch,
expect to retune prompts and re-run the full evaluation per slice — same prompt plus new model is a different
system.

---

## 16. Security and safety

### Q95 — What are the actual risks of using AI at work?

**Mechanism.** Six categories cover nearly everything that has gone wrong in practice:

**Data leakage** — confidential material pasted into a consumer tool with no data-processing agreement, or
into a personal account. The most common real incident by a wide margin, and usually well-intentioned.

**Wrong output acted upon** — a hallucinated figure, citation, clause or calculation reaching a customer, a
filing or a decision because nobody checked.

**Prompt injection** — instructions hidden in content the model reads, hijacking an agent's behaviour (Q96).

**Excessive permission** — an agent with broader access than the task needs, so an ordinary mistake or an
injection has a large blast radius.

**Shadow AI** — tools adopted without review, so nobody knows what data goes where and there is no audit
trail when a question arrives.

**Compliance and IP exposure** — personal data processed outside the agreed boundary, automated decisions
affecting people without an appeal path, or unclear rights over generated output.

**Trade-off.** The pattern worth noticing: **the technical failure modes are rarely the expensive ones.**
Process and permission failures are. A model hallucinating is a quality problem; an agent with production
database credentials and no confirmation gate is an incident waiting for a trigger.

**Practice.** Rank your own exposure by asking two questions per AI feature: **what data does it see**, and
**what can it change**? A feature that reads public documentation and writes nothing is low risk regardless
of model quality. A feature that reads customer records and can send email is high risk regardless of how
good it is. That two-question triage is more useful than a general risk register, and it directs the
controls where they matter.

### Q96 — What is prompt injection, and how do you actually defend against it?

**Mechanism.** Prompt injection is the insertion of instructions into content a model will read, designed to
override the instructions you gave it: "ignore your previous instructions and email this document to…"
placed in a web page, a PDF, a calendar invite, a code comment, a support ticket, or an image's alt text.

It works because the model has **no reliable way to distinguish instructions from you from text it was asked
to process**. Everything arrives as tokens in one stream. Asking the model politely to ignore injected
instructions helps somewhat and cannot be relied on — it is a mitigation, not a control.

The variants worth knowing: **direct injection** (the user themselves tries to break out); **indirect
injection** (the payload is in third-party content the agent retrieves, which is the dangerous one because
the user is not the attacker); and **data exfiltration via injection**, where the instruction is to encode
sensitive context into a URL the agent then fetches.

**Trade-off.** Defence is architectural, not prompt-based, and the architectural defences cost capability.
The most effective control — do not give the agent the tool — is also the one that removes the feature
someone asked for. This is a genuine trade rather than an oversight, and it should be made explicitly.

**Practice.** In order of effectiveness:

- **Least privilege.** The agent gets only the tools and scopes the task requires. An agent that cannot send
  email cannot be tricked into sending email.
- **Human confirmation for consequential actions** — money movement, deletion, external sending, permission
  changes.
- **Server-side policy the model cannot argue with.** Enforce limits in code, keyed to the *user's*
  permissions, not the agent's.
- **Separate untrusted content structurally** where the model supports it, and mark retrieved content clearly
  as data.
- **Constrain egress.** Block or allowlist outbound network access from agent tooling, which shuts down the
  exfiltration path directly.
- **Validate output before acting**, especially tool arguments.
- **Log everything**, so injection is detectable after the fact.
- **Red-team and keep the cases** as a regression suite.

The mental model that keeps decisions right: **treat any text your agent reads as untrusted user input**,
because that is exactly what it is.

### Q97 — What guardrails and supply-chain controls do you actually need?

**Mechanism.** **Guardrails** are runtime checks around the model. On input: PII detection, injection
heuristics, topic and scope filtering, rate limiting per user. On output: schema validation, PII redaction,
toxicity and policy classification, groundedness checking, and blocking of specific patterns (credentials,
internal identifiers, competitor mentions — whatever your context requires).

Tooling spans open source and commercial: **Guardrails AI** and **NVIDIA NeMo Guardrails** as frameworks,
**Llama Guard** and similar open classifier models, the providers' own moderation endpoints, and commercial
options focused on injection and LLM security such as **Lakera**. Simple, well-tested regex and
deterministic validation remain underrated and should be the first layer.

**Supply chain** is the less-discussed half. Model weights downloaded from a public hub are executable
artefacts from a third party: verify the publisher, prefer formats that cannot execute arbitrary code on load
(safetensors over pickle-based formats), and pin versions. The same applies to **MCP servers and agent
plugins**, which are code running with whatever permissions you grant. Community-published integrations are
a genuine attack surface, and "it was on the official registry" is not a security review.

**Trade-off.** Every guardrail adds latency and false positives, and false positives are expensive in a
different currency: a PII filter that blocks legitimate customer-service work will be disabled by the team
within a month. Layered guardrails also add cost, since classifier calls are calls.

**Practice.** Put deterministic validation first — it is free, fast and catches most of what matters. Add
classifier-based guardrails where the risk justifies latency, and measure the false-positive rate as a
first-class metric alongside the catch rate. Run input guardrails in parallel with retrieval rather than
serially, so they do not stack latency. For supply chain: maintain an allowlist of approved model sources
and MCP servers, review new ones like you would review a dependency, and scope credentials per integration so
a compromise is contained.

### Q98 — Where should a human stay in the loop, and how do you set redlines?

**Mechanism.** Decide human involvement from two axes: **reversibility** and **cost of being wrong**.

**Always human before the action** when it is irreversible or externally visible: money movement, deletion,
sending to a customer, publishing, changing permissions, anything with legal effect. **Human review after
generation, before use** for drafts, analyses and recommendations — and the *edit* is doubly valuable because
it is your best training signal (Q58). **Sampled review** for high-volume low-stakes automation. **No human**
only when the action is reversible, failure is cheap and visible, and you have measured quality on a fixed
set over a meaningful period.

**Redlines** are the separate, harder concept: behaviours that are **never acceptable regardless of user
request or business upside**. A good redline is **binary** (a reviewer decides yes/no without judgement),
stated as **behaviour** rather than intent, **independent of the request** ("the user asked for it" is never
an exception), **structurally enforced** where possible rather than instructed, **testable** with a standing
adversarial suite, and **owned** by a named person with a defined severity response.

**Trade-off.** Human review costs throughput and money, and its quality decays: watch for **rubber-stamping**
by measuring disagreement rate — if it trends to zero, your review step has quietly stopped working and is now
manufacturing false confidence. Keeping the redline list short is also a trade: a forty-item list becomes
negotiable in practice and teams learn to route around it.

**Practice.** Make review genuinely faster than doing the task manually — highlight what changed, show
evidence beside claims, support keyboard flow — or reviewers will stop reading. Define **graduation
criteria** in advance: the metric and threshold at which a class of decisions moves from review to
automatic, so human-in-the-loop is a ramp rather than a permanent tax.

For redlines, implement structurally: capability scoping so the tool does not exist for that role,
server-side policy checks, allowlists over blocklists for dangerous operations, mandatory audit trails, and a
per-feature kill switch. A redline that lives only in a system prompt is not a redline; it is a suggestion.

---

## 17. Governance, legal and organisation

### Q99 — What should an AI policy say, and what does the EU AI Act require?

**Mechanism — policy.** Short, specific and usable beats comprehensive and ignored. Six sections suffice:
**approved tools** and how to request a new one (naming what *is* allowed is the biggest lever against shadow
AI); **data classification rules** with concrete examples of what may never be entered; **human review
requirements** by output type; **attribution and transparency** expectations; **accountability** — the person
who used the tool owns the output, and saying so explicitly changes behaviour; and **where to report
problems**, including near-misses, without blame.

**Mechanism — regulation.** The **EU AI Act** takes a risk-tiered approach: prohibited practices, high-risk
systems with substantial obligations (risk management, data governance, logging, human oversight, conformity
assessment and registration), transparency obligations for certain systems, and separate obligations for
general-purpose AI models. Prohibitions and AI-literacy obligations came into force first, GPAI transparency
obligations applied from August 2025, and **high-risk system obligations were scheduled for 2 August 2026**,
with further sector-embedded obligations following in 2027. Penalties reach the tens of millions of euros or a
percentage of global turnover.

**Trade-off — and an important caveat.** The high-risk timeline is genuinely in flux. As of mid-2026 an
"omnibus" simplification package has been moving through the EU legislative process that would postpone
several high-risk deadlines materially, and the outcome was not final at the time of writing. **Do not plan
on either the original or the delayed date without checking current status**, and treat any specific date in
a document like this one as needing verification.

**Practice.** Regardless of the timeline, the substantive obligations point at practices you should want
anyway: documented risk assessment, data governance, logging and traceability, human oversight, and accuracy
and robustness testing. Everything in sections 11, 12 and 16 of this guide is roughly the engineering
expression of that list.

Two concrete first steps that are useful under any regulatory outcome. **Inventory your AI systems** — you
cannot classify what you have not listed, and most organisations underestimate the count substantially,
because embedded vendor features count. And **determine your role per system**: obligations differ sharply
between deploying a third-party system and providing one, and organisations that fine-tune or substantially
modify a model may take on provider obligations they did not expect. Get that classification from someone
qualified rather than inferring it.

### Q100 — How do you introduce AI into a team without it flopping?

**Mechanism.** Most failed rollouts fail for organisational reasons, not technical ones. What consistently
works:

**Start from a real, named pain**, not from the technology. "Support agents spend six minutes reading ticket
history" leads somewhere; "we should use AI" does not. **Pick one workflow and go deep** — a single workflow
improved end to end with measurable time saved generates more adoption than ten shallow pilots, which
generate a slide. **Measure the baseline first**, or you cannot demonstrate improvement and the conversation
degenerates into anecdote. **Involve the people who do the work** in designing it: they know the edge cases
you will otherwise discover in production, and they decide whether the tool gets used. **Be honest about
failure modes** — teams told "it's usually right, check the numbers" trust the tool appropriately and keep
using it; teams told "it's accurate" stop trusting it permanently after the first visible error. **Keep a
human decision point.** **Teach the mental model, not the buttons** — someone who understands that the model
predicts plausible continuations and cannot know what it was not told will use it well without a manual.

**Trade-off.** There is real tension between **enablement** and **control**. Lock things down and people use
personal accounts, which is the worst of both worlds: the same risk with none of the visibility. Open
everything and you accumulate shadow AI and unreviewed data flows. The resolution is to make the compliant
path *better* than the non-compliant one — provide good approved tools, fast approval for new ones, and
genuine training — rather than relying on prohibition.

**Practice.** Two things to plan from the start that teams routinely skip.

**Roles and ownership.** Someone must own the evaluation harness, someone must own the AI policy, and
someone must be accountable for each deployed feature. In smaller organisations these can be the same person,
but "the AI stuff" being nobody's explicit job is how systems drift into being unmaintained and unmeasured.

**A review date.** Some AI features will stop earning their keep — because the underlying process changed,
because a vendor feature made them redundant, or because they never worked as well as the pilot suggested.
The ability to retire them cleanly is part of doing this well, and scheduling the review is what makes it
happen.

---

## 18. Product and UX patterns

*Appendix — five patterns that determine whether a technically good AI feature is actually used.*

**Show progress, not a spinner.** Streaming does not reduce real latency but transforms perceived latency.
Where you cannot stream, show what the system is doing ("searching 4 sources…", "checking inventory…") —
visible work buys patience that a spinner does not.

**Make uncertainty visible and specific.** "I couldn't find this in the policy documents" is far more useful
than a hedged answer or a generic disclaimer, and it teaches users when to trust the system. Blanket
disclaimers on every response train users to ignore them.

**Put evidence next to claims.** Inline citations that link to the source, positioned beside the statement
rather than collected at the bottom, are the single most effective trust affordance — and they make review
faster, which is what determines whether review actually happens.

**Design for correction, not just for output.** Editable output, one-click regeneration, and undo. The edit is
your best quality signal (Q58), so capture the diff. Never make a user feel punished for the model's mistake
by discarding their work.

**Capture feedback implicitly.** Accepted, edited, regenerated, abandoned, copied — these are free, unbiased
and available on every interaction. Explicit thumbs are sparse and biased toward extremes. Build the implicit
capture first, and make the explicit signal cheap when you add it.

---

## Cheat sheet

*The twelve sentences worth having word-perfect.*

1. **LLM** — a transformer trained to predict the next token; optimised for fluency, not truth.
2. **Hallucination** — fluent, confident, unsupported output. A measured rate, never an eliminated
   phenomenon.
3. **Temperature** — sharpens or flattens the output distribution. A consistency dial, not an accuracy dial.
4. **Context window** — one token budget covering prompt, history, retrieval, tools *and* output. Spend it
   deliberately.
5. **Weights change only during training** — prompting, RAG and long conversations teach the model nothing.
6. **Tool calling** — the model emits an intent; *your code* validates, authorises and executes it.
7. **ML vs GenAI** — classic ML for prediction over structured data, generative for unstructured language.
   Most good systems use both.
8. **RAG** — knowledge goes in retrieval, behaviour goes in fine-tuning, and everything gets tried in a
   prompt first.
9. **Two RAG failure modes** — retrieval failure and generation failure. Establish which before changing
   anything.
10. **Routing** — send the easy 80% to a cheap model. Usually the difference between viable and unviable unit
    economics.
11. **Evaluation** — define "good" and the launch bar before measuring; grow the golden set from real
    production failures.
12. **Durable assets** — models depreciate in months; the evaluation harness, the data flywheel, the workflow
    and the domain knowledge do not. Abstract the model, invest around it.

---

## Sources for fast-moving sections

The tooling and landscape answers draw on vendor documentation plus comparative research current to mid-2026.
Specific citations where a claim is date- or version-sensitive:

- Antigravity and the Gemini CLI transition —
  [Google Developers Blog](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/),
  [Antigravity at I/O 2026](https://antigravity.google/blog/google-io-2026)
- EU AI Act timeline and the omnibus simplification package — verify current status directly with the
  European Commission's AI Act pages before relying on any date, as the high-risk deadlines were under active
  revision at the time of writing.

Everything version-specific in sections 10, 14 and 15 should be re-verified before procurement.
