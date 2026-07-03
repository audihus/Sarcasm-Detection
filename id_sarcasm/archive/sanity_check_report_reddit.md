# Sanity Check Report: REDDIT
Generated: 2026-05-07 12:34
Total samples reviewed: 100

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total samples | 100 |
| Sarcastic | 50 (50.0%) |
| Non-sarcastic | 50 (50.0%) |
| Samples with emoji | 3 (3.0%) |

### Marker Coverage

| Marker | Count | % of samples |
|--------|-------|--------------|
| [CLASH] | 84 | 84.0% |
| [EMO_CONFLICT] | 0 | 0.0% |
| Both markers | 0 | 0.0% |
| No marker | 16 | 16.0% |

### Correlation with Sarcasm Labels

| Condition | Sarcasm Rate |
|-----------|--------------|
| Has [CLASH] | 48.8% |
| No [CLASH] | 56.2% |
| Has [EMO_CONFLICT] | 0.0% |

**Lift ([CLASH] vs no [CLASH])**: 0.87x -> BAD (counterproductive)

> Lift > 1.0 artinya teks dengan [CLASH] lebih sering sarkastik dari rata-rata.
> Lift < 1.0 artinya marker salah arah (counterproductive).

## Examples: True Positives (CLASH detected, label=sarcastic)

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `yg penting peduli lindungi lewat...`
  - Augmented: `[CLASH] yg penting peduli lindungi lewat...`
  - Clash tokens: pos=['penting', 'peduli', 'lewat'], neg=['penting', 'peduli', 'lewat']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Iklanin diri ya jelas2 di reddit aja yang di blokir pemerinrah`
  - Augmented: `[CLASH] Iklanin diri ya jelas2 di reddit aja yang di blokir pemerinrah`
  - Clash tokens: pos=['ya', 'aja'], neg=['diri', 'yang', 'blokir']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `kan kata mereka buat hancurin riba, jadi nggak apa" dong pinjol, tinggal gagal bayar aja`
  - Augmented: `[CLASH] kan kata mereka buat hancurin riba, jadi nggak apa" dong pinjol, tinggal gagal bayar aja`
  - Clash tokens: pos=['buat', 'jadi', 'nggak'], neg=['kata', 'riba', 'jadi']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Udah nikah rejekinya abis buat ganti rugi truk`
  - Augmented: `[CLASH] Udah nikah rejekinya abis buat ganti rugi truk`
  - Clash tokens: pos=['nikah', 'abis', 'buat'], neg=['nikah', 'abis', 'ganti']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Beli simbadda aja sudah`
  - Augmented: `[CLASH] Beli simbadda aja sudah`
  - Clash tokens: pos=['beli', 'aja', 'sudah'], neg=['beli', 'sudah']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Di jakarta public transport sudah cukup baik. Mereka buat keputusan apa-apa ngeliat kondisi jakarta. Jakarta merepresentasikan kondisi seluruh indonesia`
  - Augmented: `[CLASH] Di jakarta public transport sudah cukup baik. Mereka buat keputusan apa-apa ngeliat kondisi jakarta. Jakarta merepresentasikan kondisi seluruh indonesia`
  - Clash tokens: pos=['sudah', 'baik', 'buat'], neg=['sudah', 'cukup', 'baik']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `jangan pernah ngekritik jogja, pasti ditanyain KTP mana. Klo ketauan bukan orang jogja pasti disuruh keluar dari jogja`
  - Augmented: `[CLASH] jangan pernah ngekritik jogja, pasti ditanyain KTP mana. Klo ketauan bukan orang jogja pasti disuruh keluar dari jogja`
  - Clash tokens: pos=['pasti', 'pasti'], neg=['jangan', 'ketauan', 'bukan']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `berarti lo gak pintar bagi waktu, that's all`
  - Augmented: `[CLASH] berarti lo gak pintar bagi waktu, that's all`
  - Clash tokens: pos=['berarti', 'bagi'], neg=['bagi']

## Examples: False Positives (CLASH detected, label=NON-sarcastic)

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `100rb per meter ud jarang, tp mungkin ada di wonosobo, temanggung, jepara  barat daya (g dingin sih), dan jalan lingkar salatiga`
  - Augmented: `[CLASH] 100rb per meter ud jarang, tp mungkin ada di wonosobo, temanggung, jepara barat daya (g dingin sih), dan jalan lingkar salatiga`
  - Clash tokens: pos=['ada', 'dingin'], neg=['mungkin', 'ada', 'dingin']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Masih remaja? Kerasin dikit aja sih. Tapi ya kamu atur aja batasnya sampe gimana.`
  - Augmented: `[CLASH] Masih remaja? Kerasin dikit aja sih. Tapi ya kamu atur aja batasnya sampe gimana.`
  - Clash tokens: pos=['aja', 'ya', 'atur'], neg=['remaja', 'dikit', 'atur']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Yang jelas Kemenkeu rada rada out of touch si kalau maksain kenaikan PPN sekarang`
  - Augmented: `[CLASH] Yang jelas Kemenkeu rada rada out of touch si kalau maksain kenaikan PPN sekarang`
  - Clash tokens: pos=['jelas'], neg=['yang', 'jelas', 'kalau']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `pengen beli thinkcentre juga gak`
  - Augmented: `[CLASH] pengen beli thinkcentre juga gak`
  - Clash tokens: pos=['pengen', 'beli'], neg=['pengen', 'beli']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Good thing sih gue ga yg bocor gila2an terus ujung2nya duit abis. Cuman ya perubahan lifestyle inevitable, minimal banget dengan alesan balas dendam dulu ga mampu beli ini-itu.`
  - Augmented: `[CLASH] Good thing sih gue ga yg bocor gila2an terus ujung2nya duit abis. Cuman ya perubahan lifestyle inevitable, minimal banget dengan alesan balas dendam dulu ga mampu beli ini-itu.`
  - Clash tokens: pos=['good', 'abis', 'ya'], neg=['bocor', 'abis', 'minimal']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `kalo spam spam ngejar comment, sempet bingung juga dulu awal awal ngejar 1k dct, kenapa harus spam tidak berbobot kayak "ayo 50 comment lagi". tapi lama lama yaudah lah hobi mereka. dan yang ngejar co`
  - Augmented: `[CLASH] kalo spam spam ngejar comment, sempet bingung juga dulu awal awal ngejar 1k dct, kenapa harus spam tidak berbobot kayak "ayo 50 comment lagi". tapi lama lama yaudah lah hobi mereka. dan yang n`
  - Clash tokens: pos=['awal', 'awal', 'berbobot'], neg=['spam', 'spam', 'awal']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `&gt;antisipasi resesi bisa menyebabkan resesi. self-fulfilling prophecy.`
  - Augmented: `[CLASH] &gt;antisipasi resesi bisa menyebabkan resesi. self-fulfilling prophecy.`
  - Clash tokens: pos=['antisipasi'], neg=['antisipasi', 'resesi', 'resesi']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Coba mengkonsumsi makanan lezat dan minumat enak?`
  - Augmented: `[CLASH] Coba mengkonsumsi makanan lezat dan minumat enak?`
  - Clash tokens: pos=['coba', 'makanan', 'lezat'], neg=['coba', 'enak']

## Examples: Missed Sarcasm (sarcastic tapi tidak dapat marker)

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Mungkin S2 artinya SD+SMP`
  - Augmented: `Mungkin S2 artinya SD+SMP`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Lambemu kuwi nang!`
  - Augmented: `Lambemu kuwi nang!`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Salah JOKOWI DONG. Rizieq gak pernah salah.`
  - Augmented: `Salah JOKOWI DONG. Rizieq gak pernah salah.`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Dasar minoritas ga tau diri`
  - Augmented: `Dasar minoritas ga tau diri`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Ga akan nyalip kalo ga d tes......`
  - Augmented: `Ga akan nyalip kalo ga d tes......`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Jangan sexist dong bro, cwe juga bisa kayak gitu`
  - Augmented: `Jangan sexist dong bro, cwe juga bisa kayak gitu`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `balik ke minyak tanah lah`
  - Augmented: `balik ke minyak tanah lah`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Antrian Pengambilan Formulir Vaksinasi Oleh Mahasiswa Universitas Negeri Malang Jurusan Kedokteran`
  - Augmented: `Antrian Pengambilan Formulir Vaksinasi Oleh Mahasiswa Universitas Negeri Malang Jurusan Kedokteran`

## Examples: EMO_CONFLICT Detected

*(tidak ada sampel dengan [EMO_CONFLICT] dalam sample set ini)*

## Examples: Emoji Expansion

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Maaf Kak Raline, *sekadar mengingatkan*, kalau masuk masjid tolong sandalnya dilepas 🙏🏻`
  - Augmented: `[CLASH] Maaf Kak Raline, *sekadar mengingatkan*, kalau masuk masjid tolong sandalnya dilepas emoji_terima_kasih emoji_light_skin_tone`
  - Clash tokens: pos=['maaf', 'mengingatkan', 'masjid'], neg=['maaf', 'sekadar', 'kalau']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Gpp kan anak dprd berarti tajir 😤`
  - Augmented: `[CLASH] Gpp kan anak dprd berarti tajir emoji_kesal`
  - Clash tokens: pos=['berarti'], neg=['anak']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Maunya rame tapi ga ada drama lah. Masa gini aja ga peka sih 😤`
  - Augmented: `[CLASH] Maunya rame tapi ga ada drama lah. Masa gini aja ga peka sih emoji_kesal`
  - Clash tokens: pos=['rame', 'ada', 'aja'], neg=['maunya', 'rame', 'ada']

## GO / NO-GO Assessment

- [FAIL] CLASH coverage 5-50%: 84.0% OUT OF RANGE
- [FAIL] CLASH lift >= 1.0: 0.87x MARKER COUNTERPRODUCTIVE
- [FAIL] CLASH tidak overfire (< 80%): 84.0% TOO HIGH

**VERDICT: NO-GO - Review manual diperlukan sebelum training.**

Lihat contoh-contoh di atas dan identifikasi pola false positive.
