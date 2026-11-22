# Blog style guide

General instructions + samples for writing (or impersonating) posts on this blog.
Distilled from all posts in `blog/src/posts/`. Confirmed choices for the QR-reader
post are marked [QR].

## Voice and persona

- First-person engineer talking to peer-experts. Conversational, never academic-formal.
- Dry, self-deprecating humor. No hype, no triumphalism. Honest about laziness,
  bad assumptions, and reinventing the wheel.
- [QR] Split pronouns by role: **I** for motivation, opinions, confessions, failures;
  **we / let's / you** for walking through technical steps together
  ("We can now…", "Let's dive into…", "if you are like me").
- Direct reader address is instructional, not intimate: "feel free to skip ahead",
  "grab your sunglasses", "Let's go Bayesian".

## Roughness: do not over-polish [QR]

- Fillers and small roughness are part of the voice. Keep them: "Turns out…",
  "Let's see…", "At the end of the day…", "Basically", "In the first place".
- Fix only typos that confuse meaning. Characteristic slips that may stay:
  "seperately", "smoothened", "in principal", "a indicator", dropped articles,
  doubled words ("the the"), `--` for em-dash.
- Signature verb: "smoothen / smoothened".

## Sentence rhythm and tics

- Short punchy openers ("Python is generally fast enough -- until it's not."),
  then long, winding, comma- and parenthesis-heavy explanations.
- Favorite transitions: "Recall that…", "Note that…", "In practice…",
  "As we shall see…", "Unfortunately…", "Interestingly…", "In particular…",
  "For example…", "In short:" (often followed by a blockquote rule).
- Parenthetical asides ("more of a transpiler really"), scare quotes
  ('realistic', 'fuzzy', 'good').
- Italics for new terms on first use and for stress (_a lot_, _could_);
  bold for punchline results; `>` for asides; backticks for tools (`pymc3`, `borg`).

## Structure

- Cold open: personal anecdote / provenance or a bold practical question, sometimes
  with the research question in italics. Then an explicit roadmap
  ("In part 1 we saw…", "In the next post…").
- Middle: `##`-headed pipeline (data → method → results), often with `###`
  sub-steps; rhetorical questions as transitions; toy problem → harder variants →
  code experiment → honest failure/nuance.
- [QR] Full failure arc per section: what I tried, why it broke, what replaced it.
  Never a separate failure gallery; failures enter where they explain a choice.
- Close: modest `## Conclusion` / "What's next?" / "Final Thoughts" that undercuts
  utility, states limits plainly, and invites extension. Never triumphant.

## Technical exposition

- Math: inline `$…$` for variables and values (`$p$-value`, `$17+4V$`); display
  `$$…$$` only for real models/derivations. Assume a math-literate reader; little
  hand-holding on linear algebra or Python.
- Code: runnable `python` blocks with imports and seeded RNG (`np.random.seed(…)`),
  followed by pasted output (`<Output>`). Hide long code in `<Details summary=…>`.
  Show JAX/Rust only when speed matters. Always define jargon on first use, in italics.
- Figures: referenced inline as evidence ("Below we see…", "top-right shows…",
  "looks like this:"), no captions; assume the reader inspects the plot.
  Prefer real project output over generic conceptual diagrams.

## Verbatim samples (steal this tone)

- "This is purely done out of laziness"
- "I don't have a satisfying explanation for this."
- "I'm sure there's a better way to do this, but it seems to work fine."
- "Not bad, if you ask me."
- "So there you have it, Russian judges tend to…"
- "It's not the best website you'll ever find, but it's mine and I like it."
- "I should publish a paper in NeurIPS and get a job at Google!"
- "swept under the rug", "smooth as butter", "Automatization isn't always worth it."
