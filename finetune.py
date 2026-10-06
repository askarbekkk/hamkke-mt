"""LoRA fine-tune a small open LLM for Korean <-> low-resource community languages.

Runs on a free Colab / Kaggle T4. Evaluates the base model, trains, evaluates again, and
writes both into runs/ in the same format as bench.py, so `python bench.py report` compares them.

  pip install -q peft sacrebleu sentence-transformers
  python finetune.py --langs khm_Khmr,npi_Deva,mya_Mymr
"""
import argparse, io, json, random, re, urllib.request, zipfile
from pathlib import Path

from bench import KO, LANGS, ROOT, RUNS, load, prompt

OPUS = {"vie_Latn": "vi", "npi_Deva": "ne", "khm_Khmr": "km", "ind_Latn": "id", "uzn_Latn": "uz",
        "tgl_Latn": "tl", "tha_Thai": "th", "mya_Mymr": "my", "sin_Sinh": "si", "ben_Beng": "bn",
        "kir_Cyrl": "ky", "khk_Cyrl": "mn", "urd_Arab": "ur", "lao_Laoo": "lo", "tgk_Cyrl": "tg",
        "kaz_Cyrl": "kk", "rus_Cyrl": "ru"}
# cleanest first. Skipped: software strings (GNOME/KDE4/Ubuntu), entity lists (XLEnt), archaic bible.
CORPORA = ["TED2020", "NeuLab-TedTalks", "QED", "GlobalVoices", "MultiParaCrawl", "OpenSubtitles", "MultiCCAligned"]
HANGUL = re.compile(r"[가-힣]")
JUNK = re.compile(r"https?://|www\.|@|\d{4}|[|<>{}]")  # web-crawl boilerplate: urls, phones, markup
latin = lambda s: len(re.findall(r"[A-Za-z]", s)) / len(s)  # high in product listings / untranslated spam


def labse_filter(pairs, keep, threshold=0.7):
    """Drop misaligned pairs: LaBSE cosine(ko, other) >= threshold, order (= corpus priority) kept."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/LaBSE")
    a, b = (m.encode([p[k] for p in pairs], batch_size=256, normalize_embeddings=True, convert_to_tensor=True)
            for k in (0, 1))
    sims = (a * b).sum(-1).tolist()
    good = [p for p, s in zip(pairs, sims) if s >= threshold]
    print(f"  LaBSE kept {len(good)}/{len(pairs)} pairs (>= {threshold})")
    return good[:keep]


def opus_pairs(lang, quota, banned):
    """Up to `quota` filtered (korean, other) pairs from OPUS for one language."""
    code = OPUS[lang]
    api = f"https://opus.nlpl.eu/opusapi/?source=ko&target={code}&preprocessing=moses&version=latest"
    found = {c["corpus"]: c["url"] for c in json.load(urllib.request.urlopen(api))["corpora"]}
    out, seen = [], set()
    for corpus in [c for c in CORPORA if c in found]:
        zpath = ROOT / "data" / "opus" / found[corpus].split("/")[-1].replace(".txt.zip", f".{corpus}.zip")
        if not zpath.exists():
            zpath.parent.mkdir(parents=True, exist_ok=True)
            print(f"  downloading {corpus} ko-{code}")
            urllib.request.urlretrieve(found[corpus], zpath)
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            read = lambda ext: io.TextIOWrapper(z.open(next(n for n in names if n.endswith(f".{ext}"))), encoding="utf-8")
            cand = []
            for ko, ot in zip(read("ko"), read(code)):
                ko, ot = ko.strip(), ot.strip()
                if (10 <= len(ko) <= 250 and 10 <= len(ot) <= 250 and 1 / 3 <= len(ko) / len(ot) <= 3
                        and HANGUL.search(ko) and not HANGUL.search(ot)
                        and not JUNK.search(ko + ot) and latin(ko) < 0.2
                        and (lang.endswith("_Latn") or latin(ot) < 0.2) and ko not in seen and ko not in banned and ot not in banned):
                    seen.add(ko)
                    cand.append((ko, ot))
        random.shuffle(cand)
        out += cand[:quota - len(out)]
        print(f"  {corpus}: {len(cand)} clean pairs, have {len(out)}/{quota}")
        if len(out) >= quota:
            break
    return out


def translate(model, tok, lang, n, out_dir, batch=16):
    """Score-ready FLORES devtest translations (first n), both directions, bench.py format."""
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for src, tgt in [(KO, lang), (lang, KO)]:
        texts = load(src)[:n]
        with (out_dir / f"{src}-{tgt}.jsonl").open("w", encoding="utf-8") as fh:
            for b in range(0, n, batch):
                chats = [tok.apply_chat_template([{"role": "user", "content": prompt(src, tgt, t)}],
                                                 tokenize=False, add_generation_prompt=True) for t in texts[b:b + batch]]
                enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
                for k, g in enumerate(tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                    fh.write(json.dumps({"i": b + k, "hyp": g}, ensure_ascii=False) + "\n")
        print(f"  {out_dir.name} {src}->{tgt} done")


def build_rows(langs, a):
    # never train on test data: ban every FLORES dev+devtest sentence
    banned = set(load(KO)) | {s for l in langs for s in load(l)}
    flores_dev = ROOT / "data" / "flores200_dataset" / "dev"
    for code in [KO, *langs]:
        banned |= set((flores_dev / f"{code}.dev").read_text(encoding="utf-8").splitlines())
    print("== building training data")
    rows = []
    for lang in langs:
        pairs = opus_pairs(lang, a.pairs * 3 if a.labse else a.pairs, banned)
        for ko, ot in (labse_filter(pairs, a.pairs, a.labse) if a.labse else pairs):
            rows += [(prompt(KO, lang, ko), ot), (prompt(lang, KO, ot), ko)]
    random.shuffle(rows)
    return rows


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", required=True, help="comma-separated FLORES codes, or 'all'")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--pairs", type=int, default=10000, help="training pairs per language (x2 directions)")
    ap.add_argument("--n", type=int, default=100, help="FLORES devtest sentences to evaluate")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch", type=int, default=8, help="per-device batch (x2 grad accumulation); 8 fits a T4")
    ap.add_argument("--labse", type=float, default=0.7, help="LaBSE alignment threshold, 0 = off")
    ap.add_argument("--eval-only", action="store_true", help="just benchmark --base (any HF model), no training")
    ap.add_argument("--push", help="optional HF Hub repo id for the adapter, e.g. you/hamkke-lora")
    a = ap.parse_args()
    random.seed(0)
    langs = list(LANGS) if a.langs == "all" else a.langs.split(",")
    short = a.base.split("/")[-1]

    rows = [] if a.eval_only else build_rows(langs, a)
    torch.cuda.empty_cache()  # LaBSE's GPU memory, before the LLM loads

    tok = AutoTokenizer.from_pretrained(a.base, padding_side="left", trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16  # T4 / P100
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32  # fp32 = CPU smoke test
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")

    print("== evaluating base model")
    for lang in langs:
        translate(model, tok, lang, a.n, RUNS / short)
    if a.eval_only:
        return

    def encode(p, target):
        head = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        h = tok(head, add_special_tokens=False).input_ids
        t = tok(target + tok.eos_token, add_special_tokens=False).input_ids
        ids = (h + t)[:256]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ([-100] * len(h) + t)[:256]}

    data = [encode(p, t) for p, t in rows]
    print(f"  {len(data)} training examples")

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in model.parameters():  # fp16 AMP needs fp32 trainable weights
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    tok.padding_side = "right"
    Trainer(model=model, train_dataset=data,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True),
            args=TrainingArguments(output_dir=str(ROOT / "ckpt"), per_device_train_batch_size=a.batch,
                                   gradient_accumulation_steps=2, learning_rate=2e-4, num_train_epochs=a.epochs,
                                   warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=50,
                                   fp16=fp16, bf16=bf16, save_strategy="no", report_to="none")).train()
    tok.padding_side = "left"

    name = f"{short}-hamkke-lora"
    model.save_pretrained(ROOT / "adapters" / name)
    if a.push:
        model.push_to_hub(a.push)

    print("== evaluating fine-tuned model")
    for lang in langs:
        translate(model, tok, lang, a.n, RUNS / name)
    print("done. download runs/ and run: python bench.py report")


if __name__ == "__main__":
    main()
