"""Hamkke (함께, "together"): can Korea's LLMs speak with everyone who works in Korea?

Translation benchmark: Korean <-> languages of Korea's E-9 (Employment Permit System)
sending countries, on FLORES-200 devtest, scored with chrF++.

  python bench.py run --model exaone3.5:2.4b-instruct-q4_K_M --n 50
  python bench.py run --model upstage/solar-pro --base-url https://openrouter.ai/api/v1 --n 100
  python bench.py report
"""
import argparse, csv, json, os, re, sys, tarfile, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "flores200_dataset" / "devtest"
RUNS = ROOT / "runs"
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# E-9 sending countries -> FLORES code. Timor-Leste (Tetum) is not in FLORES-200.
LANGS = {
    "vie_Latn": "Vietnamese", "npi_Deva": "Nepali", "khm_Khmr": "Khmer",
    "ind_Latn": "Indonesian", "uzn_Latn": "Uzbek", "tgl_Latn": "Filipino (Tagalog)",
    "tha_Thai": "Thai", "mya_Mymr": "Burmese", "sin_Sinh": "Sinhala",
    "ben_Beng": "Bengali", "kir_Cyrl": "Kyrgyz", "khk_Cyrl": "Mongolian",
    "urd_Arab": "Urdu", "zho_Hans": "Simplified Chinese", "lao_Laoo": "Lao",
    "tgk_Cyrl": "Tajik",
    # not E-9 but large communities in Korea (Central Asian / Koryo-saram); English = control
    "rus_Cyrl": "Russian", "kaz_Cyrl": "Kazakh", "eng_Latn": "English",
}
KO = "kor_Hang"


def load(code):
    if not DATA.exists():
        print("downloading FLORES-200 (25 MB)...", file=sys.stderr)
        tgz = ROOT / "data" / "flores200.tar.gz"
        tgz.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(FLORES_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(ROOT / "data", filter="data")
    return (DATA / f"{code}.devtest").read_text(encoding="utf-8").splitlines()


def name(code):
    return "Korean" if code == KO else LANGS[code]


def prompt(src, tgt, text):
    return (f"Translate the following {name(src)} text into {name(tgt)}. "
            f"Output only the {name(tgt)} translation, nothing else.\n\n{text}")


def clean(out):
    """Models chat. Keep the translation: drop <think> blocks, preambles, labels, quotes; first line only."""
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if len(lines) > 1 and lines[0][-1] in ":：":  # "Here is the translation:" + translation
        lines = lines[1:]
    line = re.sub(r"^(translation|번역)\s*[:：]\s*", "", lines[0] if lines else "", flags=re.I)
    if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'“”«»":
        line = line[1:-1]
    return line.strip()


def chat(args, prompt):
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": args.model, "temperature": 0, "max_tokens": 512,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {os.environ[args.key_env]}"} if os.environ.get(args.key_env) else {})})
    for attempt in range(6):  # hosted APIs rate-limit (429) and hiccup (5xx): back off 1..32 s
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if attempt == 5 or (e.code != 429 and e.code < 500):
                raise
            time.sleep(int(e.headers.get("Retry-After") or 2 ** attempt))


def run(args):
    langs = args.langs.split(",") if args.langs else list(LANGS)
    ko = load(KO)
    out_dir = RUNS / re.sub(r"[^\w.-]", "_", args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        other = load(lang)
        pairs = [(KO, lang, ko), (lang, KO, other)]
        if args.direction != "both":
            pairs = [p for p in pairs if (p[0] == KO) == (args.direction == "from_ko")]
        for src, tgt, s in pairs:
            f = out_dir / f"{src}-{tgt}.jsonl"
            done = {json.loads(l)["i"] for l in f.open(encoding="utf-8")} if f.exists() else set()
            todo = [i for i in range(args.n) if i not in done]
            # ponytail: sequential requests; add a thread pool when benchmarking remote APIs at scale
            for i in todo:
                hyp = chat(args, prompt(src, tgt, s[i]))  # raw; cleaned at scoring
                with f.open("a", encoding="utf-8") as fh:  # append per line = resumable
                    fh.write(json.dumps({"i": i, "hyp": hyp}, ensure_ascii=False) + "\n")
                print(f"{args.model} {src}->{tgt} {i + 1}/{args.n}", end="\r", file=sys.stderr)
            print(f"{args.model} {src}->{tgt} done      ", file=sys.stderr)


def score(n=None):
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)  # chrF++, the FLORES/NLLB standard
    rows = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.jsonl")):
            src, tgt = f.stem.split("-")
            hyps = {r["i"]: clean(r["hyp"]) for r in map(json.loads, f.open(encoding="utf-8")) if n is None or r["i"] < n}
            ids = sorted(hyps)
            ref = load(tgt)
            s = chrf.corpus_score([hyps[i] for i in ids], [[ref[i] for i in ids]]).score
            rows.append({"model": d.name, "src": src, "tgt": tgt, "n": len(ids), "chrf++": round(s, 1)})
    return rows


def report(args):
    rows = score(args.n)
    if not rows:
        sys.exit("no runs yet: python bench.py run --model ...")
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)

    models = sorted({r["model"] for r in rows})
    langs = [l for l in LANGS if any(l in (r["src"], r["tgt"]) for r in rows)]
    get = {(r["model"], r["src"], r["tgt"]): r for r in rows}
    md = []
    for title, key in [("Korean → X", lambda l: (KO, l)), ("X → Korean", lambda l: (l, KO))]:
        md += [f"### {title} (chrF++)", "",
               "| Model | " + " | ".join(LANGS[l] for l in langs) + " | Avg (excl. English) |",
               "|---" * (len(langs) + 2) + "|"]
        for m in models:
            cells = [get.get((m, *key(l))) for l in langs]
            vals = [c["chrf++"] for c, l in zip(cells, langs) if c and l != "eng_Latn"]
            avg = f"**{sum(vals) / len(vals):.1f}**" if vals else "–"
            md.append(f"| {m} | " + " | ".join(f"{c['chrf++']}" if c else "–" for c in cells) + f" | {avg} |")
        md.append("")
    md.append(f"n = sentences scored per cell (FLORES-200 devtest, first n): "
              f"{sorted({r['n'] for r in rows})}")
    (ROOT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    heatmap(models, langs, get)


def heatmap(models, langs, get):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # one-hue sequential ramp (light = low score)
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(2, 1, figsize=(2.5 + 0.55 * len(langs), 2 * (1.4 + 0.35 * len(models))), constrained_layout=True)
    for ax, (title, key) in zip(axes, [("Korean → X", lambda l: (KO, l)), ("X → Korean", lambda l: (l, KO))]):
        grid = [[(get.get((m, *key(l))) or {}).get("chrf++", float("nan")) for l in langs] for m in models]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=70, aspect="auto")
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                if v == v:  # not NaN
                    ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 38 else "#1a1a1a")
        ax.set_xticks(range(len(langs)), [{"zho_Hans": "Chinese"}.get(l, LANGS[l].split(" ")[0]) for l in langs], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)), models, fontsize=8)
        ax.set_title(f"{title}  ·  chrF++ (higher is better)", loc="left", fontsize=10, color="#1a1a1a")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.savefig(ROOT / "results.png", dpi=160, facecolor="white")
    print("wrote results.md, results.csv, results.png", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", default="http://localhost:11434/v1", help="any OpenAI-compatible endpoint (default: Ollama)")
    r.add_argument("--key-env", default="OPENAI_API_KEY", help="env var holding the API key, if any")
    r.add_argument("--n", type=int, default=100, help="sentences per direction (FLORES devtest has 1012)")
    r.add_argument("--langs", help="comma-separated FLORES codes (default: all)")
    r.add_argument("--direction", choices=["both", "from_ko", "to_ko"], default="both")
    sub.add_parser("report").add_argument("--n", type=int, help="score only the first n sentences (same subset for every model)")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
