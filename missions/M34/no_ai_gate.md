# No-AI gate — ground an answer from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Use `datasets/M34/transfer.json` only. Do not reuse notebook traces.

## Part A: classify four traces

For `t-retrieval`, `t-context`, `t-generation`, and `t-citation`, name
the **primary** failure layer: retrieval, context, generation, or
citation. Write one sentence that names the discriminating field
(retrieved ids, packed/dropped ids, answer text, or citation vs span).

## Part B: assemble a pack

Using the listed chunk order and the declared budget (`max_chars=70`,
`max_chunks=2`):

1. pack in order without reordering or splitting;
2. write packed ids, dropped ids, and the character count.

## Part C: write a supported answer

From the packed chunks in Part B, write an answer to
`t-library-close` using only a claim the cited span supports. Quote
the citation chunk id.

## Part D: reject an unsupported fluent answer

`The library closes at 22:00 on weekdays.` cites `lib::c0`. State
whether support validation must pass or fail, and which token is
missing from the span.

## Part E: RAG is not truth

In one or two sentences, explain why a retrieved, well-cited pipeline
still does not guarantee a true answer.

Pass requires independent classification, a hand-assembled pack, a
supported claim, a rejected fluent error, and an oral defense. Leave all learner responses unfilled in the repository.
