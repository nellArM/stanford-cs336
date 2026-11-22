"""Extract external reference URLs from a few English-Wikipedia dump shards and subsample them.

The staff job (scripts/download_data.py) processes every shard of the 2026-05-01 dump;
that dump is no longer hosted and we only need a few thousand positives, so this uses a
handful of shards of a current dump with the same extraction regex.

    uv run python -m cs336_data.wiki_urls --dump-date 20261001 --num-shards 4 --num-urls 14000
"""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import gzip
import random
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

from tldextract import TLDExtract

from cs336_data.paths import WIKI_DIR

# Same patterns as scripts/download_data.py::extract_wiki_urls.
URL_RE = re.compile(
    r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
)
REF_RE = re.compile("&lt;ref&gt(.*)&lt;/ref&gt;")
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".gz", ".mp3", ".mp4", ".doc", ".docx", ".xls",
    ".xlsx", ".ppt", ".pptx", ".djvu", ".ogg", ".webm", ".tif", ".tiff",
)
# Archive mirrors and DOI/ID resolvers: huge hosts that would dominate the sample.
SKIP_DOMAINS = {"archive", "doi", "worldcat", "wikipedia", "wikimedia", "google", "youtube", "handle", "jstor"}


USER_AGENT = "cs336-data-pipeline/1.0 (course assignment; python-urllib)"


def _open(url: str):
    # dumps.wikimedia.org rejects the default Python user agent.
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}))


def extract_shard(dump_date: str, shard: str, tmp_dir: str) -> list[str]:
    dump = Path(tmp_dir) / shard
    with _open(f"https://dumps.wikimedia.org/enwiki/{dump_date}/{shard}") as response, open(dump, "wb") as f:
        shutil.copyfileobj(response, f)
    urls: list[str] = []
    with bz2.open(dump, "rt", errors="ignore") as f:
        for line in f:
            if refs := REF_RE.search(line):
                urls.extend(URL_RE.findall(refs.group(0)))
    dump.unlink(missing_ok=True)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-date", default="20261001")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--num-urls", type=int, default=14000)
    parser.add_argument("--max-per-domain", type=int, default=3)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    all_urls_path = WIKI_DIR / f"enwiki-{args.dump_date}-extracted_urls.txt.gz"
    if not all_urls_path.exists():
        index = _open(f"https://dumps.wikimedia.org/enwiki/{args.dump_date}/").read().decode()
        shards = sorted(
            set(re.findall(rf"enwiki-{args.dump_date}-pages-articles-multistream[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2", index))
        )
        chosen = rng.sample(shards, args.num_shards)
        print(f"{len(shards)} shards available; using {chosen}")
        tmp_dir = WIKI_DIR / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        # Wikimedia asks for at most a couple of concurrent downloads per client.
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            results = executor.map(extract_shard, [args.dump_date] * len(chosen), chosen, [str(tmp_dir)] * len(chosen))
            with gzip.open(all_urls_path, "wt") as f:
                for urls in results:
                    for url in urls:
                        f.write(url + "\n")

    with gzip.open(all_urls_path, "rt") as f:
        urls = sorted({line.strip() for line in f if line.startswith("http")})
    print(f"{len(urls)} unique http(s) URLs")

    extractor = TLDExtract(suffix_list_urls=())
    by_domain: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        if url.lower().split("?")[0].endswith(SKIP_EXTENSIONS):
            continue
        parts = extractor(url)
        if not parts.domain or not parts.suffix or parts.domain in SKIP_DOMAINS:
            continue
        by_domain[f"{parts.domain}.{parts.suffix}"].append(url)

    candidates: list[str] = []
    for domain_urls in by_domain.values():
        rng.shuffle(domain_urls)
        candidates.extend(domain_urls[: args.max_per_domain])
    rng.shuffle(candidates)
    sample = candidates[: args.num_urls]
    out = WIKI_DIR / "subsampled_positive_urls.txt"
    out.write_text("\n".join(sample) + "\n")
    print(f"{len(by_domain)} domains; wrote {len(sample)} URLs to {out}")


if __name__ == "__main__":
    main()
