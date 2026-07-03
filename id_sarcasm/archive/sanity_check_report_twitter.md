# Sanity Check Report: TWITTER
Generated: 2026-05-07 13:19
Total samples reviewed: 100

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total samples | 100 |
| Sarcastic | 50 (50.0%) |
| Non-sarcastic | 50 (50.0%) |
| Samples with emoji | 0 (0.0%) |

### Marker Coverage

| Marker | Count | % of samples |
|--------|-------|--------------|
| [CLASH] | 33 | 33.0% |
| [EMO_CONFLICT] | 0 | 0.0% |
| Both markers | 0 | 0.0% |
| No marker | 67 | 67.0% |

### Correlation with Sarcasm Labels

| Condition | Sarcasm Rate |
|-----------|--------------|
| Has [CLASH] | 48.5% |
| No [CLASH] | 50.7% |
| Has [EMO_CONFLICT] | 0.0% |

**Lift ([CLASH] vs no [CLASH])**: 0.96x -> WEAK (signal marginal)

> Lift > 1.0 artinya teks dengan [CLASH] lebih sering sarkastik dari rata-rata.
> Lift < 1.0 artinya marker salah arah (counterproductive).

## Examples: True Positives (CLASH detected, label=sarcastic)

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Dua Keahlian Terbaik Gabener Wan Aibon: 1 . Lepas Tangan 2 . Tata Kata`
  - Augmented: `[CLASH] Dua Keahlian Terbaik Gabener Wan Aibon: 1 . Lepas Tangan 2 . Tata Kata`
  - Clash tokens: pos=['terbaik'], neg=['wan']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Partai Allah mah Bebas Sajadahpun boleh diinjak2 <hashtag>`
  - Augmented: `[CLASH] Partai Allah mah Bebas Sajadahpun boleh diinjak2 <hashtag>`
  - Clash tokens: pos=['allah'], neg=['bebas']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `<username> Mantap.....bos sama jubir mulai tidak singkron nich.....satu bilang A satunya bilang B`
  - Augmented: `[CLASH] <username> Mantap.....bos sama jubir mulai tidak singkron nich.....satu bilang A satunya bilang B`
  - Clash tokens: pos=['mantap'], neg=['tidak']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Prestasi luarbiasa gubernur <username> mampu memeratakan banjir ke seluruh jakarta . Dulu banjir hanya bisa dinikmati segelintir orang yang tinggal di daerah tertentu tapi sekarang berkat gubernur <us`
  - Augmented: `[CLASH] Prestasi luarbiasa gubernur <username> mampu memeratakan banjir ke seluruh jakarta . Dulu banjir hanya bisa dinikmati segelintir orang yang tinggal di daerah tertentu tapi sekarang berkat gube`
  - Clash tokens: pos=['banjir', 'banjir', 'banjir'], neg=['mampu', 'banjir', 'banjir']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Assyeeem... Luar biasa ini presiden. Coba kalo ini acaranya ada berjabat negara. Sudah ampruk ini di <username> ðŸ¤ª <hashtag>`
  - Augmented: `[CLASH] Assyeeem... Luar biasa ini presiden. Coba kalo ini acaranya ada berjabat negara. Sudah ampruk ini di <username> ðŸ¤ª <hashtag>`
  - Clash tokens: pos=['ada'], neg=['biasa']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Boleh Bohong Asal santun. 4nies Pembohong Santun. Prestasi yang membanggakan bagi uni Fahira & kelompoknya?? <hashtag>`
  - Augmented: `[CLASH] Boleh Bohong Asal santun. 4nies Pembohong Santun. Prestasi yang membanggakan bagi uni Fahira & kelompoknya?? <hashtag>`
  - Clash tokens: pos=['santun', 'santun'], neg=['bohong', 'pembohong', 'yang']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `<username> <username> <hashtag> saya TAULAH SIAPA yang SUKA TERIAK2 SAYA NKRI SAYA PANCASILA TAPI SEMUA DIJUALIN DAN MAU saja JADI JONGOS CINA!!!.`
  - Augmented: `[CLASH] <username> <username> <hashtag> saya TAULAH SIAPA yang SUKA TERIAK2 SAYA NKRI SAYA PANCASILA TAPI SEMUA DIJUALIN DAN MAU saja JADI JONGOS CINA!!!.`
  - Clash tokens: pos=['mau'], neg=['yang']

- **Label**: `SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Alhamdulillaaaaaah air berkah kalau banjir begini anies tidak akan nongol kalau sudah habis baru muncul lalu pengikut sampahnya teriak solusi anies`
  - Augmented: `[CLASH] Alhamdulillaaaaaah air berkah kalau banjir begini anies tidak akan nongol kalau sudah habis baru muncul lalu pengikut sampahnya teriak solusi anies`
  - Clash tokens: pos=['berkah', 'banjir'], neg=['banjir', 'tidak']

## Examples: False Positives (CLASH detected, label=NON-sarcastic)

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Kami menjamin seluruh biaya BPJS akan diselesaikan secara bertahap setiap bulannya demi pelayanan yang maksimal kepada masyarakat . Baik itu pelayan rawat jalan maupun rawat inap kata Terawan . <hasht`
  - Augmented: `[CLASH] Kami menjamin seluruh biaya BPJS akan diselesaikan secara bertahap setiap bulannya demi pelayanan yang maksimal kepada masyarakat . Baik itu pelayan rawat jalan maupun rawat inap kata Terawan `
  - Clash tokens: pos=['pelayan'], neg=['menjamin', 'yang']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `ari kaki kecium standar motor rasanya sangat romantis nyut nyutan gitu`
  - Augmented: `[CLASH] ari kaki kecium standar motor rasanya sangat romantis nyut nyutan gitu`
  - Clash tokens: pos=['romantis'], neg=['sangat']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Ya jelas aja . Duit habis banyak biayai kerusuhan gk ada hasil buat <username> malah memperburuk karir politiknya`
  - Augmented: `[CLASH] Ya jelas aja . Duit habis banyak biayai kerusuhan gk ada hasil buat <username> malah memperburuk karir politiknya`
  - Clash tokens: pos=['ya', 'ada'], neg=['kerusuhan']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Kalian lihat mereka ini saudara kita semua mereka punya anak istri dan keluarga mereka juga ingin tenang dalam melaksanakan ibadah puasa tapi klian sanggup melakukan tindakan biadab hanya kbodohan kal`
  - Augmented: `[CLASH] Kalian lihat mereka ini saudara kita semua mereka punya anak istri dan keluarga mereka juga ingin tenang dalam melaksanakan ibadah puasa tapi klian sanggup melakukan tindakan biadab hanya kbod`
  - Clash tokens: pos=['keluarga', 'tenang', 'ibadah'], neg=['biadab', 'yang']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `<username> PENDUKUNG LU BIKIN MALU NEGARA !!! <hashtag> <hashtag> <hashtag> <hashtag> <hashtag>`
  - Augmented: `[CLASH] <username> PENDUKUNG LU BIKIN MALU NEGARA !!! <hashtag> <hashtag> <hashtag> <hashtag> <hashtag>`
  - Clash tokens: pos=['pendukung'], neg=['malu']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Tips Belajar Pintar : \'\'Pilih waktu belajar yang tepat\'\' Tips Belajar Pintar : \'\'Bangun suasana belajar yang nyaman\'\'`
  - Augmented: `[CLASH] Tips Belajar Pintar : \'\'Pilih waktu belajar yang tepat\'\' Tips Belajar Pintar : \'\'Bangun suasana belajar yang nyaman\'\'`
  - Clash tokens: pos=['tips', 'tips', 'suasana'], neg=['yang', 'yang']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Semoga para sesepuh terutama para ulama tak lagi menjadi partisan politik yg membela salah satunya tp bisa memand <link>`
  - Augmented: `[CLASH] Semoga para sesepuh terutama para ulama tak lagi menjadi partisan politik yg membela salah satunya tp bisa memand <link>`
  - Clash tokens: pos=['terutama'], neg=['salah']

- **Label**: `NON-SARCASTIC` | **Markers**: `[CLASH]`
  - Original: `Wow.. banyak akun dungu baru terbit hari ini . Ayo ngeroyok gue lagi . Gue aplot dungu lu.`
  - Augmented: `[CLASH] Wow.. banyak akun dungu baru terbit hari ini . Ayo ngeroyok gue lagi . Gue aplot dungu lu.`
  - Clash tokens: pos=['ayo'], neg=['dungu', 'dungu']

## Examples: Missed Sarcasm (sarcastic tapi tidak dapat marker)

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Suatu hari nanti kita akan mengingat 14 February bukan hanya sebagai Valentine's Day tapi hari <hashtag> nasional wkwkwkwkwk..... :D`
  - Augmented: `Suatu hari nanti kita akan mengingat 14 February bukan hanya sebagai Valentine's Day tapi hari <hashtag> nasional wkwkwkwkwk..... :D`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Katanya bukan negara Islam tapi kok makai dana-dana umat Islam.?`
  - Augmented: `Katanya bukan negara Islam tapi kok makai dana-dana umat Islam.?`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Dukung Buka lapak sebatas tulisan doang :D :D belanja saja kagak pernah sok sokan dukung :D :D :D :D`
  - Augmented: `Dukung Buka lapak sebatas tulisan doang :D :D belanja saja kagak pernah sok sokan dukung :D :D :D :D`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Sungguh mulia sekali kader Partai Allah ini :) Setelah bagi2 kursi apalagi? Giveaway tiket surga?`
  - Augmented: `Sungguh mulia sekali kader Partai Allah ini :) Setelah bagi2 kursi apalagi? Giveaway tiket surga?`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Ngumpulin koin buat pembangunan gedung KPK? tidak malu apa? Satu lagi yang sudah menjadi budaya Indonesia . MENGEMIS !`
  - Augmented: `Ngumpulin koin buat pembangunan gedung KPK? tidak malu apa? Satu lagi yang sudah menjadi budaya Indonesia . MENGEMIS !`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `daripada ibu kota diurus sama ahli tata kata <username> mending pindah sekalian ??<hashtag>`
  - Augmented: `daripada ibu kota diurus sama ahli tata kata <username> mending pindah sekalian ??<hashtag>`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Keren sekali tampilan monas saat ini .... <hashtag> <hashtag>`
  - Augmented: `Keren sekali tampilan monas saat ini .... <hashtag> <hashtag>`

- **Label**: `SARCASTIC` | **Markers**: `(none)`
  - Original: `Selamat ya gaes ..jd duta dusta indonesia`
  - Augmented: `Selamat ya gaes ..jd duta dusta indonesia`

## Examples: EMO_CONFLICT Detected

*(tidak ada sampel dengan [EMO_CONFLICT] dalam sample set ini)*

## Examples: Emoji Expansion

*(tidak ada teks dengan emoji dalam sample set ini)*

## GO / NO-GO Assessment

- [PASS] CLASH coverage 5-50%: 33.0% OK
- [FAIL] CLASH lift >= 1.0: 0.96x MARKER COUNTERPRODUCTIVE
- [PASS] CLASH tidak overfire (< 80%): 33.0% OK

**VERDICT: NO-GO - Review manual diperlukan sebelum training.**

Lihat contoh-contoh di atas dan identifikasi pola false positive.
