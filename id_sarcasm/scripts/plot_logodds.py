import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({"font.family": "serif", "font.size": 9})

# Monroe et al. (2008) z-scores computed from IdSarcasm Twitter train split
# (470 sarcastic, 1408 non-sarcastic; min freq=5; function words removed)
entries = [
    # non-sarcastic: most negative first (bottom)
    ("tolol",        -3.59, "#2c5f8a"),
    ("goblok",       -3.26, "#2c5f8a"),
    ("sontoloyo",    -3.19, "#2c5f8a"),
    ("tau",          -2.82, "#2c5f8a"),
    ("pejabat",      -2.38, "#2c5f8a"),
    ("tetap",        -2.22, "#2c5f8a"),
    # sarcastic: least positive first (just above zero)
    ("datang",        3.53, "#888888"),
    ("terimakasih",   3.71, "#c0392b"),
    ("kolam",         3.86, "#888888"),
    ("keren",         3.91, "#c0392b"),
    ("berkah",        3.98, "#c0392b"),
    ("selamat",       4.64, "#c0392b"),
    ("warga",         4.70, "#888888"),
    ("jakarta",       4.94, "#888888"),
    ("pintar",        5.00, "#c0392b"),
    ("gubernur",      5.64, "#888888"),
    ("banjir",        6.91, "#888888"),
    ("anies",         8.34, "#888888"),
]

words  = [e[0] for e in entries]
scores = [e[1] for e in entries]
colors = [e[2] for e in entries]

fig, ax = plt.subplots(figsize=(3.5, 4.4))

ax.barh(words, scores, color=colors, height=0.7, edgecolor="none")
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("log-odds z-score (sarcastic vs. non-sarcastic)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    mpatches.Patch(facecolor="#c0392b", label="Positive-surface (sarcastic)"),
    mpatches.Patch(facecolor="#888888", label="Topical (sarcastic)"),
    mpatches.Patch(facecolor="#2c5f8a", label="Literal pejorative (non-sarc.)"),
]
ax.legend(handles=legend_handles, fontsize=7, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig("id_sarcasm/fig_logodds.pdf", bbox_inches="tight")
print("Saved id_sarcasm/fig_logodds.pdf")
