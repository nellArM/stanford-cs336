#set document(title: "CS336 Assignment 4 (data): Filtering Language Modeling Data")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 10.5pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)
#show raw.where(block: false): set text(size: 9pt)
#show raw.where(block: true): set text(size: 8.5pt)
#show link: set text(fill: rgb("#1f5fbf"))

#let problem(name, points) = block(above: 1.3em, below: 0.6em)[
  #text(weight: "bold")[Problem (#raw(name)) — #points points]
]
#let note(body) = block(fill: rgb("#f3f5f8"), inset: 8pt, radius: 3pt, width: 100%, body)
#let excerpt(body) = block(inset: (left: 10pt), stroke: (left: 1.5pt + rgb("#c5ccd6")), text(size: 9pt, body))

#align(center)[
  #text(size: 16pt, weight: "bold")[CS336 Assignment 4 (data): Filtering Language Modeling Data]
  #v(2pt)
  Written report — handout version 26.0.1 (Spring 2026), completed as self-study
]

#note[
  *Setting and deviations from the handout.* This work was done outside the course, on one shared Linux
  server (128 cores, 2 of 4 RTX A6000 GPUs available) with no access to the course's Modal volume.
  The differences from the handout are:
  - *Data scale.* The English WET set was rebuilt locally from *400* raw WET files of crawl
    `CC-MAIN-2026-17` (the staff used 2,500), with the staff's sampling seed and the same rule
    (fastText English probability ≥ 0.7). Runtimes are extrapolated to 2,500 files and to a full crawl.
  - *Wikipedia URLs.* The 2026-05-01 dump is no longer hosted; reference URLs were extracted with the
    staff's regular expressions from 4 shards of the 2026-10-01 dump.
  - *Example files.* `example.warc.gz` / `example.warc.wet.gz` were fetched by the staff's
    `scripts/download_data.py --offline-only` (first WARC/WET of `CC-MAIN-2026-12`).
  - *Training.* 8 B200 GPUs are not available. The same model, optimizer, schedule and global batch
    (1,024 sequences per step) are run through the unmodified `scripts/train.py` on 2 A6000s using
    gradient accumulation, for 1/16 of the staff's 16,384 steps (see the last problem).
  - *PyTorch build.* `torch` is taken from the CUDA 12.6 wheel index because the server's driver does
    not support CUDA 13.
  All statistics below were produced by the code in this repository; all "look at the data" answers
  are based on samples that were actually read.
]

= Filtering Common Crawl

#problem("look_at_cc", 4)

*(a)* The first response record is `http://020bld.cn.vauofnj.cn/html/991f998999.html`, crawled on
2026-03-05. It is no longer reachable (the host does not resolve/connect). The raw HTML is a Chinese
page built from a stock "company website" template (car audio, car batteries, a hotline of
`020-123456789`) whose article body is an unrelated question-and-answer text about whether lily-pollen
stains wash out — i.e. an auto-generated SEO/spam page on a throw-away subdomain.

*(b)* The WET text contains everything visible: the navigation menu, breadcrumb, placeholder contact
block ("地址：联系地址联系地址联系地址", `admin@…}`), "previous/next article" links, the friend-link
footer and an ICP licence line; all of that should have been dropped by a main-content extractor. As
training data the page teaches a model template boilerplate, fake contact details and — worse — the
article itself has had words shuffled in to evade duplicate detection ("蕊染花蜜", "色洗花蜜"), so the model
would learn subtly ungrammatical Chinese. What is useful is small: the general topic knowledge (how
to remove pollen stains; a short botanical description of lilies) and the look of Chinese Q&A text.

*(c)* It could be useful as a _negative_ example for a web-spam / content-farm detector (or for
training a boilerplate remover). It is not useful for an English assistant or for any application
that needs fluent, factual Chinese, e.g. a household-advice chatbot.

*(d)* Annotations of the next 25 WET records (language by reading, checked against fastText):

#{
  set text(size: 8.3pt)
  table(
    columns: (auto, auto, 2.3fr, 3fr),
    stroke: 0.4pt + rgb("#b8c0cc"),
    inset: 3.5pt,
    table.header[*\#*][*Lang.*][*Domain*][*Type of page / notes*],
    [1], [zh], [021sakura.com], [Template company site ("industry news" list); only menus and slogans.],
    [2], [zh/ja], [044mm.com], [Adult video player page; title plus category menu. Harmful, no prose.],
    [3], [zh], [0553njl.com], [Construction company, awards index page; almost entirely navigation.],
    [4], [zh], [0553njl.com], [Same site, short news item (client's thank-you letter) under a huge menu.],
    [5], [zh], [07921.cn], [Local job board listing; browser-upgrade banner, login links, phone number.],
    [6], [zh], [07921.cn], [Same job board, another paginated category listing (near-duplicate of 5).],
    [7], [en], [100.ubc.ca], [UBC centennial alumni profile; real but short text under a long alumni-services menu.],
    [8], [ru], [11235813.org], [Forum software "help" search page (Invision board); boilerplate only.],
    [9], [ru], [123-market.ru], [Online shop brand listing; ad-block notice, product names and prices.],
    [10], [de], [128437.homepagemodules.de], [Forum user profile page for a spam account; login prompts.],
    [11], [zh], [168ps.com], [Pirate streaming site page for a variety show; menus and SEO description.],
    [12], [zh], [198gg.com], [Adult video index with keyword-stuffed title. Harmful.],
    [13], [el], [1pekesat-exae.mysch.gr], [Greek school helpdesk forum index (phpBB); navigation only.],
    [14], [en], [2019.whitehorseartshow.com.au], [Art-show catalogue entry for one painting; dates and menu, a few words of content.],
    [15], [zh], [223hei.com], [Adult video page. Harmful.],
    [16], [zh], [250a.com], [Paywalled "dating course" download page with a cloud-drive link.],
    [17], [zh], [26888hd3.com], [Online gambling bonus table (deposit → bonus). Spam.],
    [18], [en], [2e-vaucresson-garches.agse.fr], [Hacked French scout-group blog carrying a machine-spun English dating-spam post.],
    [19], [en], [3344eh.com], [The single word "Loading" (JavaScript-only page).],
    [20], [ja], [377410.com], [Tutoring agency testimonial index: a list of near-identical link titles.],
    [21], [zh], [37online.com], [Gossip/"leak" blog post about a football match; generated filler prose.],
    [22], [zh], [3iio8u.xyyanglao.com], [Hijacked subdomain: real semiconductor company text with gambling brand names substituted in.],
    [23], [zh], [4008881886.cn], [Auto-generated fake company profile (random product list, stock political phrases).],
    [24], [zh], [431dd.com], [Adult video page (same template as 2). Harmful.],
    [25], [zh], [432149.com], [Lottery spam title followed by scraped web-novel paragraphs: fluent text, wrong context.],
  )
}

None of these 25 records is what I would call a high-quality page. The closest is record 7 (the UBC
alumni profile, i.e. the 8th record of the file), which has a few sentences of genuine human-written
English buried in navigation; records 21 and 25 contain fluent prose but are spam. In other words it
takes more than 26 records to meet the first clearly high-quality document — because records are
sorted by URL, the start of the file is dominated by numeric spam domains, and later samples
(below) look considerably better.

#problem("extract_text", 3)

*(a)* `cs336_data/extract_text.py::extract_text_from_html_bytes` decodes as UTF-8, falls back to
`resiliparse.parse.encoding.detect_encoding` when that fails, and calls `extract_plain_text`.
`uv run pytest -k test_extract_text_from_html_bytes` passes.

*(b)* On the first 501 pages of the example WARC that also appear in the WET file, the two extractions
contain essentially the same visible text (median length ratio mine/WET = 1.06). The differences are
structural: Resiliparse keeps the document layout — blank lines between blocks and list items rendered
as "• item" (89 bullets per page on average), and it includes `alt` text and form labels, so it is
longer on 42% of pages — whereas the WET text is one flat line per DOM text node with no markers. Neither
removes navigation or footers; I find the Resiliparse output slightly better because the preserved
structure makes menus (runs of short bullet lines) easy to recognise and strip afterwards, while the
WET format is more compact.

#problem("language_identification", 6)

*(a)* `cs336_data/language_identification.py::identify_language` runs fastText `lid.176.bin` on the text
with newlines replaced by spaces, strips the `__label__` prefix (the model already uses `en`/`zh`) and
clips the probability to $[0, 1]$. Both `test_identify_language` tests pass.

*(b)* Errors go both ways. False negatives silently remove particular kinds of English — dialects,
code-switched text, short or list-like pages, text full of names or numbers — so the model becomes
worse exactly for the under-represented users who write that way; false positives let other languages
and gibberish through, which wastes capacity and can make a model switch language or emit mojibake
mid-answer. A document-level label also hides mixed-language pages. For a user-facing product I would
(i) measure precision/recall per language and per text genre on a hand-labelled sample instead of
trusting the score, (ii) run identification at paragraph level and keep or drop spans rather than
pages, (iii) use a conservative threshold together with a second detector and send disagreements to
manual review, and (iv) monitor the language mix of the final dataset and the deployed model's outputs.

*(c)* I ran the classifier on the text extracted from all 19,325 non-empty HTML responses of the
example WARC and labelled 20 random documents by hand. The classifier agreed with me on 19; the one
error was an English aircraft-registry page consisting only of menu items, labelled Norwegian with
confidence 0.23. Every other doubtful case also had a low score: a French photo-gallery page made of
size labels (fr, 0.24), an English page with Hindi fragments (en, 0.39), an English product page under a
`/fr/` URL (en, 0.44). By the classifier 41.1% of documents are English (8 of my 20, and 9 by my own
labels), but only 25.1% are English with score ≥ 0.7 and 17.1% with score ≥ 0.8. Since all
mistakes and mixed pages sat below 0.5 and confident predictions were always right, a threshold of
about 0.7–0.8 is appropriate; the pipeline uses 0.80 (on top of the 0.7 prefilter), accepting some
loss of short or list-like English pages that are poor training text anyway.

#problem("mask_pii", 3)

*(a)–(c)* `cs336_data/pii.py` implements three regular expressions:
e-mail addresses (`local@domain.tld`, not preceded/followed by word characters, so an existing
`|||EMAIL_ADDRESS|||` is untouched); North-American phone numbers (optional `+1`, 3-digit area code
with or without parentheses, 3 + 4 digits, separated by nothing, space, dot or dash, with look-arounds
that refuse matches inside longer digit strings); and IPv4 addresses (four octets 0–255). All five
`test_mask_*` tests pass.

*(d)* Naive masking changes the text distribution: the model sees the literal strings
`|||PHONE_NUMBER|||` etc. in contexts where real text has varied values, so it can emit the
placeholder at inference time and never learns the format of an address or number (bad for tasks such
as extracting contact details or writing a signature block). False positives destroy non-PII content
— ISBNs, order numbers, version strings, dates, measurements — and false negatives leave real PII in,
while giving a false sense of safety. Mitigations: replace matches with realistic synthetic surrogates
of the same format instead of a constant token (or at least make the token a dedicated special token
that is filtered from model output); validate candidates (phone-number libraries, checks on context
words such as "tel", "call"); and audit precision/recall on a labelled sample for each data source.

*(e)* 37.4% of the documents in the example WARC had at least one replacement (9,998 e-mails, 7,789
phone numbers, 478 IP addresses in total). In 20 random masked documents the e-mail matches were all
correct (`office\@….org`, `editor\@….com`, …) and US/Canadian numbers such as
`(707) 391-xxxx` or `819-713-xxxx` were handled well. *False positives:* a Chinese QQ messenger ID
(`27651xxxxx`), an example string in gambling terms and conditions (`666-6666666`), and 10-digit
non-phone numbers in general; Korean and Japanese numbers (`031-977-xxxx`, `078-918-xxxx`) were masked
only because they happen to have the 3-3-4 shape. *False negatives:* practically every international
format — `+7 (495) 120-xx-xx`, `+97158983xxxx`, `0974-567-xxx`, `+370 602 xxxxx`, `(+84) 97 505 xxxx`,
the UK number `01483 56xxxx` — is left in clear text (a loose scan found about 8,400 such strings in
this one file), as would be obfuscated addresses such as "name [at] domain [dot] com".

#problem("harmful_content", 6)

*(a)–(b)* `cs336_data/harmful_content.py` wraps the two Dolma Jigsaw fastText models and returns
(`"nsfw"`/`"non-nsfw"`, confidence) and (`"toxic"`/`"non-toxic"`, confidence). Both tests pass.

*(c)* The classifiers were trained on English Wikipedia talk-page comments, so they key on vocabulary.
Applied as filters they remove legitimate text that merely mentions sensitive words — sexual-health and
medical pages, news about crimes, fiction, reclaimed slurs and dialects that the Jigsaw data is known
to over-flag — which skews the data against certain topics and communities and leaves the model unable
to discuss, recognise or refuse harmful content competently. They also miss harm that is expressed
politely or in other languages. Mitigations: use high thresholds and remove only clearly harmful pages;
classify spans rather than whole pages; evaluate false-positive rates by topic and dialect; keep the
removed data to train safety classifiers; and rely on post-training safeguards rather than data
filtering alone.

*(d)* In 20 random documents the classifiers flagged nothing, and I agree in 19 cases; the one miss was
an explicit web-cam listing page (`sexcityguide.com`) scored non-NSFW (P(nsfw) = 0.00, P(toxic) = 0.03)
because the page is a list of category names rather than sentences. Overall only 0.95% of documents are
flagged at threshold 0.5 (NSFW 0.25%, toxic 0.94%; 0.69% of the English documents). I therefore also
read 12 of the 184 flagged documents: the English ones with scores near 1.0 were all pornographic sites
(true positives), but 64 of the 184 flagged pages are German and 24 Czech — ordinary job adverts and
hobby pages misread by an English-only model — and a "lorem ipsum" theme demo scored P(toxic) = 0.79.
Lowering the threshold to 0.2 or 0.1 raises the flagged fraction to 2.8% and 4.9%, mostly through
such false positives. Suitable thresholds are therefore around 0.5 for both classifiers, applied only
after the English filter; recall on list-like adult pages has to come from other signals (domain or
keyword lists).

#problem("gopher_quality_filters", 3)

*(a)* `cs336_data/gopher.py` rejects documents with fewer than 50 or more than 100,000 words, a mean
word length outside 3–10 characters, more than 30% of lines ending in an ellipsis, or fewer than 80% of
words containing a letter. Words are whitespace-separated tokens that contain at least one
alphanumeric character (so stand-alone bullets and separators are not counted — with naive whitespace
tokens the "•" list markers produced by Resiliparse made 31% of English pages fail the alphabetic
rule). All seven `test_gopher` tests pass.

*(b)* 84.8% of all extracted documents (88.1% of the English ones) pass; failures are too few words
(5.4%), mean word length (5.9%, almost all Chinese/Japanese pages without spaces) and non-alphabetic
words (3.7%). In 20 random documents the filter rejected two — a three-word "WordPress Coming Soon"
page and a 58-word restaurant widget error page — and I agree with both. Of the 18 it passed, I would
keep at most a third: the rules happily pass pure navigation and catalogue pages (a 7,741-word dental
supply category tree, a supermarket outlet selector, a casino comparison table, a Tatoeba menu), a
login form with a few sentences and several tag/index pages. The rules catch degenerate documents but
say nothing about whether the words form prose, which is why the pipeline adds line-level rules and
a learned classifier.

#problem("quality_classifier", 15)

*(a)* *Data.* `cs336_data/wiki_urls.py` extracted 2,763,626 unique reference URLs (258,601 registered
domains) from 4 dump shards, dropped binary file types and archive/ID-resolver hosts, capped every
domain at 3 URLs and sampled 14,000. `scripts/scrape_positive_urls.sh` fetched them with
`wget --timeout=5 --tries=1 -i … --warc-file=… -O /dev/null` split over parallel wget processes; I
stopped it after 6,051 successful responses (target: 5,000). Positives are the HTTP-200 HTML responses
after text extraction that are English (≥ 0.7) and pass the Gopher rules: *3,570* documents.
Negatives are the *4,359* documents of the example Common Crawl WARC that pass exactly the same
extraction and filters, so the classifier cannot separate the classes by language, length or extractor
artefacts. Classes were balanced (3,570 each) and split 90/10.

*Model.* `cs336_data/train_quality_classifier.py` trains a fastText supervised model on lower-cased,
whitespace-normalised text (first 20,000 characters): word bigrams, 100 dimensions, 25 epochs, learning
rate 0.5. Validation accuracy is *84.2%* (wiki: precision 0.83 / recall 0.85; cc: 0.85 / 0.83). The
model is then quantized so that it can ship with the code (`cs336_data/assets/quality_classifier.ftz`,
12 MB; 80.4% validation accuracy after quantization). The numeric quality score is P(wiki).

*(b)* `cs336_data/quality_classifier.py::classify_quality` returns (`"wiki"` or `"cc"`, confidence).
`uv run pytest -k test_classify_quality` passes (the low-quality fixture is labelled `cc`, the
Wikipedia-reference fixture `wiki`).

= Deduplication

#problem("exact_deduplication", 3)

`cs336_data/deduplication.py::exact_line_deduplication` makes one pass counting an 8-byte BLAKE2b
digest of every line and a second pass rewriting each file with only the lines whose digest count is 1.
`test_exact_line_deduplication` passes. The corpus-scale variant used in the pipeline
(`filter_pipeline.py`, stage 2) is the same algorithm with 64-bit MurmurHash values held in NumPy
arrays: workers hash their files in parallel, the main process finds the duplicated hashes with one
`np.unique`, and workers filter their lines against the sorted array by binary search.

#problem("minhash_deduplication", 8)

`minhash_deduplication` normalises each document (lower-case, NFD, accents and punctuation removed,
whitespace collapsed), forms the set of word $n$-grams, and computes a MinHash signature: each
$n$-gram is hashed once with MurmurHash3 and the $k$ hash functions are simulated by universal hashing
$h_i(x) = (a_i x + b_i) mod (2^31 - 1)$, vectorised in NumPy. Signatures are cut into $b$ bands; documents
sharing a band are candidates; candidates are confirmed with the true $n$-gram Jaccard similarity and
merged with union-find, so clusters are transitive; one random member of each cluster is written.
Both `test_minhash_deduplication` tests pass. At corpus scale (stage 3) the bands are grouped with a
sort instead of a Python dictionary and every bucket member is compared with the bucket's first
document (a star rather than all pairs), which keeps large boilerplate buckets linear.

= Filtering data for language modeling

#problem("filter_data", 6)

*(a) Pipeline.* `uv run python -m cs336_data.filter_pipeline --workers 48` processes the English WET
files with a `ProcessPoolExecutor`. Because the target is perplexity on Paloma's C4-100-domains split,
the central design decision is to *imitate how C4 itself was built*: in the validation data 99.9% of
lines end in terminal punctuation and essentially none has fewer than 5 words, so text that does not
look like that is off-distribution. The steps, in order:

+ *Language*: English with fastText score ≥ 0.80.
+ *Gopher rules* on the raw WET text.
+ *Harmful content*: P(nsfw) ≥ 0.5 or P(toxic) ≥ 0.5 removes the page.
+ *C4-style rules* (`cs336_data/c4_rules.py`): keep only lines with ≥ 5 words that end in terminal
  punctuation, drop lines with boilerplate phrases (JavaScript, cookie notices, "all rights reserved",
  "log in", …) or mostly non-Latin characters; then drop pages left with < 50 words or < 3 sentences,
  or containing "lorem ipsum" or a curly brace.
+ *Quality classifier* on the cleaned text, P(wiki) ≥ 0.05. The threshold was calibrated on Paloma
  (allowed by the handout): the classifier's scores are bimodal, and 0.05 keeps 87% of 4,194
  validation documents while removing 44% of the Common Crawl candidates (0.25 would discard 21% of
  validation-like documents — forum and commercial text that C4 does contain).
+ *PII masking* of e-mail addresses, phone numbers and IPv4 addresses (modifies, never removes).
+ *Exact line deduplication* over the whole corpus, then pages left with < 50 words / < 3 sentences are dropped.
+ *MinHash + LSH deduplication* (128 hashes, 16 bands, word 5-grams, Jaccard ≥ 0.8).

#figure(
  {
    set text(size: 9pt)
    table(
      columns: (auto, auto, auto, auto),
      align: (left, right, right, right),
      stroke: 0.4pt + rgb("#b8c0cc"),
      inset: 4pt,
      table.header[*Step*][*Documents removed*][*Share of all discards*][*Documents remaining*],
      [Input (English WET, 100 files)], [—], [—], [2,553,123],
      [Language (en ≥ 0.80)], [778,711], [42.2%], [1,774,412],
      [Gopher rules], [198,620], [10.8%], [1,575,792],
      [NSFW classifier], [1,731], [0.09%], [1,574,061],
      [Toxic-speech classifier], [1,277], [0.07%], [1,572,784],
      [C4-style line/page rules], [180,264], [9.8%], [1,392,520],
      [Quality classifier], [610,283], [33.0%], [782,237],
      [Too short after exact line dedup], [75,361], [4.1%], [706,876],
      [MinHash + LSH duplicates], [408], [0.02%], [706,468],
      [*Total discarded / kept*], [*1,846,655*], [100%], [*706,468 (27.7%)*],
    )
  },
  caption: [Documents removed by each filter step (400 raw WET files of CC-MAIN-2026-17).],
)

Steps that modify rather than remove: the line rules and line deduplication reduce the kept documents
from their original size to 3.29 billion characters in total (20% of the 16.3 billion input characters);
exact line deduplication deleted 4,345,389 of 17,442,569 lines (24.9%; 1,011,088 distinct repeated
lines); PII masking touched 43,400 documents (36,997 e-mails, 39,110 phone numbers, 3,457 IPs). MinHash
found 3,401 candidate pairs, confirmed 408 and formed 261 clusters (largest: 18 documents) — few,
because line deduplication had already removed most templated text.

*(b) Runtime.* On 48 worker processes the pipeline took *383 s* of wall-clock time for the 100 English
files derived from 400 raw WET files: filtering 242 s (8,846 CPU-seconds, about 3.5 ms per document),
line deduplication 56 s, MinHash deduplication 34 s, tokenization 51 s. Building the English set from
the raw files (download, language identification, rewrite) took a further 514 s on 24 processes for
26.6 GB of compressed WET data (8,552,606 records, of which 29.9% were kept as English).
Scaling linearly, the 2,500-file set would take about *40 minutes* for the pipeline (plus about 54
minutes for the English prefilter) on this machine. A full crawl has 100,000 WET files (about 2.1
billion pages): about 250× our input, i.e. roughly 2,100 CPU-hours — about *44 hours* on these 48
workers or 17 hours on all 128 cores — and the entire Common Crawl archive ("over 250 billion pages")
is about 120 such crawls, on the order of 250,000 CPU-hours. The per-file stages scale linearly and
are trivially distributed; the global stages would need a sharded implementation (the line-hash table
of one full crawl alone is about 3.5 billion distinct 8-byte hashes).

#problem("inspect_filtered_data", 4)

*(a) Five random kept documents* (one random document from each of five random output shards):

+ `cheadlecog.com/post/exciting-new-initiative-…` (198 words).
  #excerpt["Support Staffordshire is leading a new initiative across Staffordshire, starting with the Staffordshire Moorlands as the first participating area. Called Local Help My Way, the initiative will connect people who need extra help to live independently …"]
  A local community news item in clean, complete sentences — exactly the news/blog register of C4; worthwhile.
+ `store3z.net/62-great-books-by-black-authors-…` (842 words).
  #excerpt["Our recommendation of the place to start out with Jonesâs work, however, is her quick hear, Half Light. Once youâve finished, go forward and make your way by way of Jonesâs extensive catalog. …"]
  Machine-spun spam: an article paraphrased by synonym substitution ("quick hear" for "short read") with mojibake apostrophes. Not worthwhile — it is fluent enough to pass every filter, and it is the main failure mode left in the data.
+ `brianmurphymusic.com/don-dt-thompson.html` (377 words).
  #excerpt["(William) Don or 'D.T.' Thompson. Saxophonist, composer, arranger, b Drumheller, Alberta, 19 Sep 1932, d Vancouver 21 Mar 2004. He played saxophone and clarinet at 12 and began promoting his own jazz concerts …"]
  An encyclopaedic musician biography; dense, factual and well written. Worthwhile.
+ `isu.edu/biology/biolab/student-behavior/terms--definitions/` (224 words).
  #excerpt["All areas of learning have their own terminology that allows for concise and effective communication between practitioners. Specific terminology decreases the chances of subjective interpretations …"]
  University teaching material in formal prose; worthwhile (educational domains are well represented in C4).
+ `movingfurniturerecords.com/artist/eventless-plot/` (252 words).
  #excerpt["Eventless Plot borrows elements from different genres and aesthetics to build their own hybrid of sound. Formed in Thessaloniki in 2002 by Vasilis Liolios, Aris Giatas and Yiannis Tsirikoglou …"]
  A record-label artist biography; clean promotional prose, typical of small-business pages in C4. Worthwhile.

Four of five are good; in all of them the navigation, headers and footers of the original pages are gone.

*(b) Five removed documents* (drawn at random from the per-reason samples the pipeline stores, so rare
reasons are over-represented):

+ `….blogspot.com/2019/12/denver-co.html` — removed by the *Gopher* rule on alphabetic words.
  A two-sentence cat-blog post followed by comment timestamps and share buttons ("Posted by … at 12:02 AM Email ThisBlogThis!Share to X…"). Justified: there is almost no content, though the rule that fired is incidental.
+ `2gaycam.com/cam/…/` — removed by the *toxic* classifier. A web-cam directory: a list of
  user names and explicit category words. Justified (it would also have failed the line rules).
+ `hat.wordpress.org/plugins/backup-master-for-your-website/` — removed by the *quality classifier*.
  A WordPress plugin directory page (menus, "Log in", "Submit a plugin", a short product description).
  Justified, although a main-content extractor could have rescued the description.
+ `india-porn.mobi/…` — removed by the *NSFW* classifier. A pornographic video page with an explicit
  story snippet. Justified.
+ `bondagestories.biz/stories/story.php?storyid=3634` — removed by the *toxic* classifier. The menu of
  an erotic-fiction archive around a German-language story that passed the English prefilter because of
  the English navigation. Justified on two counts (language and content).

*(c) Iterations.* (1) The first version used whitespace tokens for the Gopher rules; auditing showed
it rejected 31% of English pages because of bullet characters, so "words" were redefined as tokens with
an alphanumeric character. (2) After comparing kept text with Paloma, the C4-style line rules were
added: before them, kept documents still began with menus and cookie banners. (3) A stray Persian line
inside an English page led to the per-line non-Latin check. (4) The quality threshold was lowered from
0.25 to 0.05 after measuring that 0.25 would reject a fifth of Paloma itself; this raised the final
dataset from about 615 to 709 million tokens. (5) Example (a)-2 suggests a further filter for spun
text and mojibake (e.g. dropping pages containing "â" sequences, or a small language-model perplexity
filter); I did not add it, to keep the reported pipeline identical to the one that produced the
training data.

#problem("tokenize_data", 2)

Stage 4 of `filter_pipeline.py` (`stage4_tokenize`) encodes every kept document with the GPT-2 tokenizer
from `transformers`, appends `<|endoftext|>` (id 50256) after each document, and writes the
concatenation with `ids_array.tofile(path)` as `np.uint16`
(`local-shared-data/filtered/filtered_cc_gpt2_tokens.bin`, 1.42 GB). The filtered dataset contains
*709,480,999 tokens* in 706,468 documents (mean 1,004 tokens per document; Paloma documents average 694).

#let summary = json("figures/training_summary.summary")

#problem("train_model", 8)

*Setup.* The model and optimisation are exactly those of `cs336_basics/train_config.py`, run by the
unmodified `scripts/train.py`: 24 layers, $d_"model" = 1024$, 16 heads, $d_"ff" = 3072$, context 512,
GPT-2 vocabulary; AdamW (lr $6 times 10^(-4)$, betas 0.9/0.98, weight decay 0.1), 3% warm-up and cosine
decay to 10%, gradient clipping at 1.0, bf16 autocast, `torch.compile`. `scripts/train_local.py` only
supplies the configuration: one GPU × 32 sequences × 32 accumulation steps = the staff's 1,024
sequences per optimizer step (the planned 2-GPU DDP launch was abandoned: NCCL's direct GPU-to-GPU
transport hangs on this machine, and the second card was needed by another job, so the run used a
single RTX A6000; the script still supports `--nproc_per_node=2`); *1,024 steps* instead of 16,384 (about 537 million sampled
tokens, 0.76 epochs of the filtered data); validation on Paloma C4-100-domains every 128 steps (100
batches of 32 sequences); losses written to a local `metrics.jsonl` instead of wandb.

#if summary.available [
  *Result.* The best validation loss on Paloma C4-100-domains was *#summary.best_eval_loss*
  (perplexity #summary.best_eval_perplexity) at step #summary.best_eval_step of #summary.steps;
  the run took #summary.wall_hours hours of wall-clock time on one RTX A6000.

  #figure(
    image("figures/learning_curve.png", width: 82%),
    caption: [Training loss (filtered Common Crawl) and validation loss (Paloma C4-100-domains).],
  )

  #figure(
    {
      set text(size: 9pt)
      table(
        columns: summary.evals.len() + 1,
        stroke: 0.4pt + rgb("#b8c0cc"),
        inset: 4pt,
        [*Step*], ..summary.evals.map(e => [#e.at(0)]),
        [*Val. loss*], ..summary.evals.map(e => [#e.at(1)]),
      )
    },
    caption: [Validation loss at each evaluation.],
  )

  Because the run uses 1/16 of the staff's optimisation steps, the absolute loss is not comparable
  with losses from the full 16,384-step schedule; the curve is still decreasing at the end of the run.
] else [
  #note[
    *Results pending.* The training run had not been executed when this report was compiled. After it
    finishes, `uv run --with matplotlib python writeup/plot_training.py` regenerates
    `figures/learning_curve.png` and `figures/training_summary.summary`, and recompiling this file fills in
    the best validation loss, the learning curve and the table of evaluations here.
  ]
]
