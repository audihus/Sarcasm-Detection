"""Demo sementara: trace incongruity_label pada beberapa kalimat contoh."""
from framework.incongruity import (
    load_inset, incongruity_label, _score_units, STRONG, WINDOW, NEG_WINDOW,
)

inset, maxlen = load_inset("../real_data/twitter/positive.tsv",
                           "../real_data/twitter/negative.tsv")
print(f"InSet: {len(inset)} entri | STRONG={STRONG} WINDOW={WINDOW} NEG_WINDOW={NEG_WINDOW}\n")

# token sudah dalam bentuk baku (seperti keluaran to_lexicon_tokens)
examples = {
    "1. ramah TAPI kotor      ": ["pelayanannya", "ramah", "tapi", "kamarnya", "kotor"],
    "2. semua positif         ": ["tempatnya", "nyaman", "makanannya", "enak", "pelayanannya", "cepat"],
    "3. TIDAK bagus           ": ["tidak", "bagus", "sama", "sekali"],
    "4. TIDAK jelek tapi buruk": ["tidak", "jelek", "tapi", "pelayanannya", "buruk"],
    "5a. TIDAK enak (dekat)   ": ["makanannya", "tidak", "enak"],
    "5b. ...enak jauh dr tidak": ["tidak", "ada", "satu", "pun", "makanan", "enak"],
}

for name, toks in examples.items():
    units = _score_units(toks, inset, maxlen)          # (posisi, skor SETELAH negasi)
    clash, strength = incongruity_label(toks, inset, maxlen)
    raw = [(w, inset.get(w, "·")) for w in toks]        # bobot mentah InSet (sebelum negasi)
    print(f"{name} -> CLASH={clash} (strength={strength})")
    print(f"    token+bobotMentah : {raw}")
    print(f"    unit setelah negasi: {units}")
    pos = [(i, s) for i, s in units if s >= STRONG]
    neg = [(i, s) for i, s in units if s <= -STRONG]
    print(f"    POS-kuat={pos}  NEG-kuat={neg}\n")
