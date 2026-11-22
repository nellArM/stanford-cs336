// Hand-written commentary for problem prompting_baselines; numbers come from
// experiments/baselines/ (see generated/baselines.typ and generated/baseline_misparse.typ).

Measured accuracy (category 1) is 0.08% for `question_only`, 0.08% for zero-shot `r1_zero` and 19.3% for 3-shot
`r1_zero`. One response is sampled per question at temperature 1.0, so each number carries sampling noise of roughly
#sym.plus.minus 1 percentage point for the 3-shot prompt.

*Are categories 2 and 3 hiding correct answers?* To look beyond a handful of examples, every response in categories 2
and 3 was screened with a simple heuristic --- does the last number in the response (after `<answer>` if present)
equal the ground truth? --- and the hits, plus ten random non-hits per category, were read by hand.

#include "generated/baseline_misparse.typ"

- *`r1_zero`, category 2 (654).* The dominant pattern is that the model writes a full sentence, or the last step of
  the calculation, inside the answer tags (for example `<answer>... the total cost of 6 erasers and 8 pencils is
  12 + 24 = $36.</answer>` for ground truth 36). The grader compares the whole tag content with the number and marks
  it wrong. All ten of the heuristic hits that were read are genuinely correct solutions, and none of the ten random
  non-hits is, so about 57 of 654 (9%) of this category is "correct but not parsed".
- *`r1_zero`, category 3 (664).* These fail the strict `</think> <answer>` pattern: the model never closes
  `</think>`, writes `</thinking>`, puts text between the two tags, or drifts into another document. 47 end in the
  right number and five of the six that were read are correct solutions with broken tags, so roughly 40 (6%) of this
  category is correct. Altogether the zero-shot `r1_zero` model solves an estimated 7% of the problems, not
  0.08%; almost all of the gap to the measured number is formatting.
- *`r1_zero_three_shot`, category 2 (997).* Here the answer tag almost always holds a bare number, and the number
  is simply wrong (arithmetic slips, misread quantities). Only 16 responses end in the correct number; of the twelve
  read, eleven are correct but carry a unit, a currency symbol with a thousands separator, an equation or a sentence
  (`<answer> 120 ml </answer>`, `<answer> 60*30 = 1800 pounds of chocolates in total. </answer>`). So about 1.5% of
  this category is mis-parsed. Category 3 (68) is mostly malformed closing tags (`</ think>`, `</thought>`) or
  responses that run to the token limit; 3 of them end in the right number.
- *`question_only`.* Category 2 (308) exists mainly because the model echoes the instruction "put your final answer
  within `\boxed{}`", which the parser accepts as a (wrong or empty) boxed answer; nothing read in this category was
  a correct solution apart from one response that stated the right number in prose and boxed a different one.
  Category 3 (1,010) contains a few correct answers given without a box ("correct answer: 50"); of the 26 heuristic
  hits in both categories most are coincidental numbers inside unrelated generated text.

#include "generated/baseline_examples.typ"

*(b) Behaviour under each prompt.* With `question_only` the base model does not treat the text as a question to be
answered at all. It continues the "document": it invents further exam questions, imitates forum pages, tutoring
dialogues and web boilerplate ("Share on your favorite social platforms."), repeats the boxed-answer instruction, and
frequently runs until the 512-token limit (434 of 1,319 responses; mean length 259 tokens). Simply providing the
question is therefore not enough for a base model.

The zero-shot `r1_zero` prompt changes the genre to a User/Assistant exchange and, because it ends in an opened
`<think>` tag, makes the model reason first: responses are short (109 tokens on average) and mostly on topic. But
the model has only a description of the format, not an instance of it. Half of the responses do not reproduce the
exact tag sequence, and when they do, the content of `<answer>` is free text rather than a number, which is why
measured accuracy is near zero although the reasoning is often right.

The three in-context examples fix both problems at once: 95% of responses follow the format exactly, the reasoning
copies the terse "so the answer is ..." style of the demonstrations, and the answer tag contains a bare number.
Accuracy rises to 19.3%, and the remaining errors are real reasoning or arithmetic errors rather than formatting.
For the RL experiments this means the zero-shot `r1_zero` policy starts with almost no reward signal and must first
discover the answer format, while the 3-shot prompt starts from a much better policy.
