"""
augment_pipeline.py
===================
Text augmentation pipeline untuk deteksi sarkasme Bahasa Indonesia.

Strategi: enrich input text dengan marker eksplisit untuk signal sarkasme,
lalu biarkan BERT yang belajar dari signal tersebut. Tidak mengubah arsitektur,
100% kompatibel dengan run_classification.py existing.

Pipeline (Hybrid Level 1+2+3):
  1. expand_emoji_to_text    : emoji -> token Indonesia
  2. detect_polarity_clash   : cek ada token positif DAN negatif di teks
  3. detect_emoji_sentiment  : cek emoji berkonflik dengan sentimen teks
  4. augment_text            : prepend [CLASH] dan/atau [EMO_CONFLICT] + expand emoji
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, FrozenSet

try:
    import emoji as emoji_lib
    EMOJI_LIB_AVAILABLE = True
except ImportError:
    EMOJI_LIB_AVAILABLE = False
    print("[WARNING] Library 'emoji' tidak tersedia. Fallback ke manual dict saja.")
    print("          Install dengan: pip install emoji")


# ===========================================================================
# SECTION 1: EMOJI EXPANSION
# ===========================================================================

# Top-30 emoji paling sering muncul di social media Indonesia
# Value: token yang akan menggantikan emoji di dalam teks.
# Format: emoji_<deskripsi> agar BERT bisa belajar dari token ini.
EMOJI_ID_DICT: dict[str, str] = {
    "😂": "emoji_haha",
    "🤣": "emoji_sangat_lucu",
    "😭": "emoji_nangis",
    "😅": "emoji_gugup",
    "🙏": "emoji_terima_kasih",
    "😍": "emoji_suka_banget",
    "❤️": "emoji_cinta",
    "👍": "emoji_bagus",
    "😊": "emoji_senang",
    "🥺": "emoji_memelas",
    "😒": "emoji_skeptis",       # signal sarkasme kuat
    "😔": "emoji_sedih",
    "😏": "emoji_sinis",         # signal sarkasme
    "🤔": "emoji_bingung",
    "😤": "emoji_kesal",
    "💀": "emoji_haha_mati",     # slang: lucu banget (death by laughter)
    "😢": "emoji_menangis",
    "🫠": "emoji_lemas",
    "🥲": "emoji_terharu",
    "😑": "emoji_datar",         # signal sarkasme
    "🙄": "emoji_malas",         # signal sarkasme
    "😩": "emoji_lelah",
    "💔": "emoji_patah_hati",
    "😡": "emoji_marah",
    "🤦": "emoji_facepalm",
    "👏": "emoji_tepuk_tangan",
    "✨": "emoji_keren",
    "🔥": "emoji_keren_sekali",
    "💯": "emoji_setuju",
    "😌": "emoji_tenang",
}

# Emoji yang bersentimen negatif/skeptis (untuk deteksi konflik)
NEGATIVE_EMOJIS: FrozenSet[str] = frozenset({
    "😒", "😑", "🙄", "😤", "😡", "😔", "😢", "💔", "😭", "😩",
    "🤦", "😏", "😾", "💢", "🤬", "😞", "😟", "🥺", "😰", "😨",
})

# Emoji yang bersentimen positif
POSITIVE_EMOJIS: FrozenSet[str] = frozenset({
    "😂", "🤣", "😍", "❤️", "👍", "😊", "💯", "✨", "🔥", "👏",
    "😁", "😆", "🥰", "😄", "😃", "🎉", "💕", "💖", "😀", "🤩",
})


def _demojize_fallback(char: str) -> str:
    """
    Fallback jika library emoji tidak tersedia.
    Pakai unicodedata.name untuk dapat nama emoji.
    Output: emoji_<nama_lowercase_underscored>
    """
    try:
        name = unicodedata.name(char, "").lower().replace(" ", "_").replace("-", "_")
        if name:
            return f"emoji_{name}"
    except Exception:
        pass
    return "emoji_unknown"


def _clean_demojized(text: str) -> str:
    """
    Ubah output emoji.demojize (format :nama_emoji:) jadi token bersih.
    Contoh: ':smirking_face:' -> 'emoji_smirking_face'
    """
    # Ganti :kata_kata: jadi emoji_kata_kata
    cleaned = re.sub(r":([a-z0-9_]+):", r"emoji_\1", text)
    return cleaned


def expand_emoji_to_text(text: str) -> str:
    """
    Ganti semua emoji dalam teks dengan token Indonesia.

    Prioritas:
      1. Jika ada di EMOJI_ID_DICT -> pakai token manual Indonesia
      2. Jika tidak ada -> pakai emoji.demojize (English) lalu bersihkan format-nya
      3. Fallback terakhir -> pakai unicodedata.name

    Original text DIPERTAHANKAN, hanya emoji yang di-replace.
    Spasi ditambahkan di sekitar token agar tidak menempel ke kata lain.

    Args:
        text: teks asli yang mungkin mengandung emoji

    Returns:
        teks dengan semua emoji sudah di-expand
    """
    if not text:
        return text

    result = []
    i = 0
    text_chars = list(text)
    n = len(text_chars)

    while i < n:
        char = text_chars[i]

        # Cek multi-char emoji (misal: ❤️ adalah ❤ + variation selector)
        # Coba match 2 karakter dulu
        two_char = "".join(text_chars[i:i+2]) if i + 1 < n else ""
        if two_char and two_char in EMOJI_ID_DICT:
            result.append(f" {EMOJI_ID_DICT[two_char]} ")
            i += 2
            continue

        # Cek single emoji di manual dict
        if char in EMOJI_ID_DICT:
            result.append(f" {EMOJI_ID_DICT[char]} ")
            i += 1
            continue

        # Cek apakah karakter ini adalah emoji (diluar manual dict)
        if EMOJI_LIB_AVAILABLE and emoji_lib.is_emoji(char):
            # Coba demojize 2-char window dulu
            two_char_str = "".join(text_chars[i:i+2]) if i + 1 < n else ""
            if two_char_str and emoji_lib.is_emoji(two_char_str):
                demojized = emoji_lib.demojize(two_char_str)
                token = _clean_demojized(demojized)
                result.append(f" {token} ")
                i += 2
                continue
            else:
                demojized = emoji_lib.demojize(char)
                token = _clean_demojized(demojized)
                result.append(f" {token} ")
                i += 1
                continue

        # Bukan emoji, tambahkan as-is
        result.append(char)
        i += 1

    expanded = "".join(result)
    # Bersihkan spasi ganda yang mungkin muncul
    expanded = re.sub(r" {2,}", " ", expanded).strip()
    return expanded


def extract_emojis(text: str) -> list[str]:
    """
    Ekstrak semua karakter emoji dari teks.
    Returns list of emoji characters.
    """
    if not text:
        return []

    emojis = []
    text_chars = list(text)
    n = len(text_chars)
    i = 0

    while i < n:
        char = text_chars[i]

        # Cek 2-char emoji dulu
        two_char = "".join(text_chars[i:i+2]) if i + 1 < n else ""
        if two_char and (two_char in EMOJI_ID_DICT or two_char in NEGATIVE_EMOJIS or two_char in POSITIVE_EMOJIS):
            emojis.append(two_char)
            i += 2
            continue

        if char in EMOJI_ID_DICT or char in NEGATIVE_EMOJIS or char in POSITIVE_EMOJIS:
            emojis.append(char)
            i += 1
            continue

        if EMOJI_LIB_AVAILABLE and emoji_lib.is_emoji(char):
            emojis.append(char)

        i += 1

    return emojis


# ===========================================================================
# SECTION 2: POLARITY CLASH DETECTION
# ===========================================================================

def load_inset_lexicon(
    positive_path: str,
    negative_path: str,
) -> Tuple[frozenset, frozenset]:
    """
    Load InSet lexicon dari file TSV dengan filter skor dan stopword PySastrawi.

    Positif: skor >= 3; Negatif: skor <= -3.
    Header baris pertama dilewati; ValueError tiap baris di-skip.
    Tokenizer: re.findall(r'[a-zA-Z]+', kata.lower()) — multi-word entries di-split.
    """
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "PySastrawi"], check=True)
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    factory = StopWordRemoverFactory()
    STOPWORDS = set(factory.get_stop_words())
    print(f"Stopwords PySastrawi: {len(STOPWORDS)} kata")

    def _load_words(path: str, *, min_score: Optional[float], max_score: Optional[float]) -> set:
        words: set = set()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"InSet lexicon tidak ditemukan: {path}")
        with open(p, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            entry = parts[0].lower().strip()
            try:
                score = float(parts[1])
            except ValueError:
                continue
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue
            for token in re.findall(r'[a-zA-Z]+', entry):
                if token and token not in STOPWORDS:
                    words.add(token)
        return words

    pos_fixed = _load_words(positive_path, min_score=3.0, max_score=None)
    neg_fixed = _load_words(negative_path, min_score=None, max_score=-3.0)

    print(f"Kata positif setelah filter: {len(pos_fixed)}")
    print(f"Kata negatif setelah filter: {len(neg_fixed)}")

    return frozenset(pos_fixed), frozenset(neg_fixed)


def _simple_tokenize(text: str) -> list[str]:
    """Tokenizer untuk InSet lookup: ekstrak token alphabetik lowercase."""
    return re.findall(r'[a-zA-Z]+', text.lower())


def detect_polarity_clash(
    text: str,
    positive_words: frozenset,
    negative_words: frozenset
) -> Tuple[bool, dict]:
    """
    Deteksi apakah teks mengandung clash sentimen (ada token positif DAN negatif).

    Clash = indikasi potensial sarkasme karena ada kontradiksi leksikal.
    Contoh: "Bagus banget hari ini" (bagus=pos) + "😒" -> bukan clash teks murni
            "Bagus banget padahal ancur" -> bagus=pos, ancur=neg -> CLASH

    Args:
        text: teks asli (sebelum emoji expansion)
        positive_words: set kata positif dari InSet
        negative_words: set kata negatif dari InSet

    Returns:
        (is_clash, details) dimana details berisi token yang ditemukan
    """
    # Untuk clash detection, gunakan teks tanpa emoji
    text_no_emoji = _remove_emojis_for_analysis(text)
    tokens = _simple_tokenize(text_no_emoji)

    found_positive = [t for t in tokens if t in positive_words]
    found_negative = [t for t in tokens if t in negative_words]

    is_clash = len(found_positive) > 0 and len(found_negative) > 0

    details = {
        "tokens_checked": tokens,
        "positive_found": found_positive,
        "negative_found": found_negative,
    }

    return is_clash, details


def _remove_emojis_for_analysis(text: str) -> str:
    """Hapus emoji dari teks untuk analisis sentimen teks murni."""
    if not text:
        return text

    result = []
    text_chars = list(text)
    n = len(text_chars)
    i = 0

    while i < n:
        char = text_chars[i]

        # Skip 2-char emoji
        two_char = "".join(text_chars[i:i+2]) if i + 1 < n else ""
        if two_char and two_char in EMOJI_ID_DICT:
            i += 2
            continue

        if char in EMOJI_ID_DICT:
            i += 1
            continue

        if EMOJI_LIB_AVAILABLE and emoji_lib.is_emoji(char):
            i += 1
            continue

        result.append(char)
        i += 1

    return "".join(result).strip()


# ===========================================================================
# SECTION 3: EMOJI-SENTIMENT CONFLICT DETECTION
# ===========================================================================

def detect_emoji_sentiment(
    text: str,
    positive_words: frozenset,
    negative_words: frozenset,
    min_text_sentiment_tokens: int = 1
) -> Tuple[bool, dict]:
    """
    Deteksi konflik antara sentimen emoji dan sentimen teks.

    Logika:
      1. Ekstrak emoji dari teks
      2. Hitung emoji positif vs negatif
      3. Tentukan "emoji_sentiment" (positif/negatif/neutral)
      4. Tentukan sentimen teks via InSet (tanpa emoji)
      5. Jika keduanya berlawanan arah -> CONFLICT

    Contoh konflik:
      "Bagus banget hari ini 😒" -> teks positif (bagus) + emoji negatif (😒) = CONFLICT
      "Hancur banget deh 👍" -> teks negatif (hancur) + emoji positif (👍) = CONFLICT

    Args:
        text: teks asli
        positive_words, negative_words: InSet lexicon
        min_text_sentiment_tokens: minimal berapa token sentimen di teks agar dihitung

    Returns:
        (is_conflict, details)
    """
    emojis = extract_emojis(text)

    if not emojis:
        return False, {"reason": "no_emoji"}

    # Hitung sentimen emoji
    n_pos_emoji = sum(1 for e in emojis if e in POSITIVE_EMOJIS)
    n_neg_emoji = sum(1 for e in emojis if e in NEGATIVE_EMOJIS)

    if n_pos_emoji == 0 and n_neg_emoji == 0:
        return False, {"reason": "emoji_sentiment_unknown", "emojis": emojis}

    # Majority vote sentimen emoji
    emoji_sentiment = "positive" if n_pos_emoji > n_neg_emoji else "negative"

    # Sentimen teks via InSet
    text_no_emoji = _remove_emojis_for_analysis(text)
    tokens = _simple_tokenize(text_no_emoji)
    n_pos_text = sum(1 for t in tokens if t in positive_words)
    n_neg_text = sum(1 for t in tokens if t in negative_words)

    total_sentiment_tokens = n_pos_text + n_neg_text
    if total_sentiment_tokens < min_text_sentiment_tokens:
        return False, {
            "reason": "text_sentiment_unclear",
            "emojis": emojis,
            "emoji_sentiment": emoji_sentiment,
            "text_pos": n_pos_text,
            "text_neg": n_neg_text
        }

    text_sentiment = "positive" if n_pos_text > n_neg_text else "negative"

    # Conflict jika berlawanan arah
    is_conflict = (emoji_sentiment != text_sentiment)

    details = {
        "emojis": emojis,
        "emoji_sentiment": emoji_sentiment,
        "n_pos_emoji": n_pos_emoji,
        "n_neg_emoji": n_neg_emoji,
        "text_sentiment": text_sentiment,
        "n_pos_text": n_pos_text,
        "n_neg_text": n_neg_text,
    }

    return is_conflict, details


# ===========================================================================
# SECTION 4: MAIN AUGMENTATION PIPELINE
# ===========================================================================

class AugmentPipeline:
    """
    Pipeline utama augmentasi teks.

    Usage:
        pipeline = AugmentPipeline(
            positive_path="real_data/reddit/positive.tsv",
            negative_path="real_data/reddit/negative.tsv"
        )
        result = pipeline.augment("Bagus banget hari ini 😒")
        # Output: "[CLASH] [EMO_CONFLICT] Bagus banget hari ini emoji_skeptis"
    """

    def __init__(
        self,
        positive_path: str,
        negative_path: str,
        use_clash: bool = True,
        use_emo_conflict: bool = True,
        use_emoji_expand: bool = True,
    ):
        """
        Args:
            positive_path: path ke positive.tsv InSet
            negative_path: path ke negative.tsv InSet
            use_clash: aktifkan [CLASH] marker
            use_emo_conflict: aktifkan [EMO_CONFLICT] marker
            use_emoji_expand: aktifkan emoji expansion
        """
        self.positive_words, self.negative_words = load_inset_lexicon(
            positive_path, negative_path
        )
        self.use_clash = use_clash
        self.use_emo_conflict = use_emo_conflict
        self.use_emoji_expand = use_emoji_expand

        print(f"[AugmentPipeline] InSet loaded: "
              f"{len(self.positive_words)} pos words, "
              f"{len(self.negative_words)} neg words")
        print(f"[AugmentPipeline] Config: clash={use_clash}, "
              f"emo_conflict={use_emo_conflict}, "
              f"emoji_expand={use_emoji_expand}")

    def augment(self, text: str) -> Tuple[str, dict]:
        """
        Apply full augmentation pipeline ke satu teks.

        Args:
            text: teks asli

        Returns:
            (augmented_text, debug_info)
        """
        if not text or not isinstance(text, str):
            return text, {"error": "invalid_input"}

        debug = {"original": text}
        markers = []

        # Step 1: Deteksi [CLASH]
        if self.use_clash:
            is_clash, clash_details = detect_polarity_clash(
                text, self.positive_words, self.negative_words
            )
            debug["clash"] = {"detected": is_clash, **clash_details}
            if is_clash:
                markers.append("[CLASH]")

        # Step 2: Deteksi [EMO_CONFLICT]
        if self.use_emo_conflict:
            is_conflict, conflict_details = detect_emoji_sentiment(
                text, self.positive_words, self.negative_words
            )
            debug["emo_conflict"] = {"detected": is_conflict, **conflict_details}
            if is_conflict:
                markers.append("[EMO_CONFLICT]")

        # Step 3: Expand emoji di teks
        if self.use_emoji_expand:
            body = expand_emoji_to_text(text)
        else:
            body = text

        # Step 4: Gabungkan marker + body
        if markers:
            augmented = " ".join(markers) + " " + body
        else:
            augmented = body

        debug["markers"] = markers
        debug["augmented"] = augmented

        return augmented, debug

    def augment_batch(self, texts: list[str]) -> Tuple[list[str], list[dict]]:
        """Apply augmentation ke list of texts."""
        results = [self.augment(t) for t in texts]
        augmented_texts = [r[0] for r in results]
        debug_infos = [r[1] for r in results]
        return augmented_texts, debug_infos


# ===========================================================================
# SECTION 5: STRUCTURAL MARKERS (EDA-VALIDATED)
# ===========================================================================

# Kata hiperbola yang menjadi signal sarkasme di Twitter (lift 2.11x)
HYPERBOLE_WORDS: FrozenSet[str] = frozenset([
    "banget", "sekali", "sangat", "paling", "amat",
    "sungguh", "benar-benar", "beneran", "literally",
])

# Lazy-loaded InSet untuk Twitter [CLASH] detection
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_TWITTER_POS_WORDS: Optional[frozenset] = None
_TWITTER_NEG_WORDS: Optional[frozenset] = None


def _get_twitter_inset() -> Tuple[frozenset, frozenset]:
    """Lazy-load InSet Twitter untuk [CLASH] detection."""
    global _TWITTER_POS_WORDS, _TWITTER_NEG_WORDS
    if _TWITTER_POS_WORDS is None:
        pos_path = _PROJECT_ROOT / "real_data" / "twitter" / "positive.tsv"
        neg_path = _PROJECT_ROOT / "real_data" / "twitter" / "negative.tsv"
        _TWITTER_POS_WORDS, _TWITTER_NEG_WORDS = load_inset_lexicon(
            str(pos_path), str(neg_path)
        )
    return _TWITTER_POS_WORDS, _TWITTER_NEG_WORDS


def add_structural_markers(text: str, dataset_name: str) -> str:
    """
    Prepend structural sarcasm markers berdasarkan signal EDA per-platform.

    Reddit  : [SHORT] jika word_count <= 8 (lift 1.73x)
    Twitter : [CLASH] jika ada clash positif-negatif InSet (lift 1.75x)
              [QUES]  jika ada '?' di teks (lift 1.84x)
              [HYPER] jika ada kata hiperbola di teks (lift 2.11x)
              Urutan prepend: [CLASH] [QUES] [HYPER] — semua additive.

    Teks asli TIDAK diubah — hanya marker yang di-prepend.
    Jika tidak ada marker yang aktif, kembalikan teks asli tanpa perubahan.

    Args:
        text: teks asli
        dataset_name: "reddit" atau "twitter"

    Returns:
        teks dengan marker di-prepend, atau teks asli jika tidak ada marker
    """
    if not text or not isinstance(text, str):
        return text

    if dataset_name == "reddit":
        if len(text.split()) <= 8:
            return "[SHORT] " + text
        return text

    elif dataset_name == "twitter":
        pos_words, neg_words = _get_twitter_inset()
        markers = []
        is_clash, _ = detect_polarity_clash(text, pos_words, neg_words)
        if is_clash:
            markers.append("[CLASH]")
        if "?" in text:
            markers.append("[QUES]")
        text_lower = text.lower()
        if any(hw in text_lower for hw in HYPERBOLE_WORDS):
            markers.append("[HYPER]")
        if markers:
            return " ".join(markers) + " " + text
        return text

    return text