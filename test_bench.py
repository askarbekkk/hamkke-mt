from bench import LANGS, KO, clean, load

# output cleaning: chatty model replies -> bare translation
assert clean("Xin chào") == "Xin chào"
assert clean("<think>hmm, Vietnamese...</think>\n\nXin chào") == "Xin chào"
assert clean("Translation: Xin chào\n\nNote: formal register") == "Xin chào"
assert clean('"Xin chào"') == "Xin chào"
assert clean("번역: 안녕하세요") == "안녕하세요"
assert clean("The translation of the given Lao text into Korean is:\n\n안녕하세요") == "안녕하세요"
assert clean("") == ""

# every benchmark language exists in FLORES and is sentence-aligned with Korean
n = len(load(KO))
assert n == 1012, n
for code in LANGS:
    assert len(load(code)) == n, code
print("ok")
