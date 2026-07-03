import re, sys
sys.path.insert(0, "scripts")

_DELETED_RE = re.compile(r"\[(deleted|removed)[^\]]*\]", re.IGNORECASE)

def clean_parent(text) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if s.lower() == "none":
        return ""
    s = _DELETED_RE.sub("", s).strip()
    return s

# Test cleaning
cases = [
    "Hello world",
    "[deleted]",
    "[deleted by user]",
    "[removed]",
    "[DELETED]",
    "some text [deleted] more text",
    "None",
    None,
    "",
    "  ",
]
for c in cases:
    result = clean_parent(c)
    print(f"  {repr(c)!s:<40} -> {repr(result)}")

# Test dataset columns
from datasets import load_from_disk
ds = load_from_disk("reddit_with_context")
train = ds["train"]
print(f"\nTrain size: {len(train)}")
print(f"Columns: {train.column_names}")

# Count parent coverage
no_parent = sum(1 for r in train if clean_parent(r["parent_text"]) == "")
print(f"Rows without parent (after clean): {no_parent} / {len(train)}")
print(f"Rows with parent   (after clean): {len(train) - no_parent} / {len(train)}")
