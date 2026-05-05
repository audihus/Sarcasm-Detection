# Penjelasan Teknis Direktori `multichannel-sarcasm-detection`

Berdasarkan analisis teknis dari isi direktori ini, repositori ini merupakan implementasi model *Deep Learning* berbasis PyTorch untuk mendeteksi sarkasme menggunakan pendekatan multi-kanal (**Multi-Channel Sarcasm Detection**). Pendekatan ini menggabungkan fitur sintaksis (struktur kalimat) dan semantik (makna dan sentimen). 

Berikut adalah rincian teknis dari setiap komponen dan file yang ada di dalam direktori, yang sangat relevan untuk konteks penelitian Anda:

## 1. File Konfigurasi dan Eksekusi Utama
*   **`train.sh`**: Script bash yang berfungsi sebagai *wrapper* untuk menjalankan proses pelatihan secara terotomatisasi. Script ini mendefinisikan *hyperparameters* default (seperti `voc_size=30000`, `batch_size=128`, `n_layers=3`) dan mengeksekusi `main.py`.
*   **`main.py`**: Merupakan titik masuk (*entry point*) utama dari program pelatihan. File ini menangani argumen (menggunakan `argparse`), memuat data menggunakan `DataManager`, menginisialisasi model jaringan saraf (`bridgeModel`), dan menjalankan loop pelatihan serta evaluasi (menghitung loss, metrik, dll.).
*   **`README.md`**: Dokumentasi dasar yang menjelaskan tahapan *pipeline* penelitian (Mulai dari *Preprocessing*, *Dependency Graph*, *Sentiment Graph*, *Training*, hingga *Evaluation*).

## 2. Arsitektur Model (*Core Neural Network*)
*   **`bridgeModel.py`**: File ini membungkus model utama (`dualModel`). Secara spesifik, kelas ini bertanggung jawab menyiapkan *batch data* yang masuk dari *loader* (termasuk grafik dependensi dan sentimen), mengelola *optimizer* (Adam/Adadelta), dan menghitung fungsi *loss* gabungan (`loss_ce` untuk prediksi sarkasme, `loss_senti` untuk prediksi literatur, dan `loss_nonsenti` untuk probabilitas mendalam).
*   **`Model.py`** dan **`basicModel.py`**: Berisi arsitektur detail dari jaringan (*Neural Network architecture*). Menggunakan model RNN (BiLSTM/GRU) yang digabungkan dengan struktur kanal lain (seperti *ADGCN - Attention-Driven Graph Convolutional Networks*).
*   **`GarphModel.py`**: (Terdapat *typo* pada nama file aslinya). Kemungkinan besar berisi implementasi *Graph Convolutional Network* (GCN) untuk memproses struktur *graph* dari data kalimat.
*   **`attention.py`**: Modul yang secara khusus memuat mekanisme *Attention* (perhatian), yang kemungkinan digunakan untuk memberikan bobot lebih pada kata-kata atau fitur graf yang paling penting dalam kalimat sarkastis.
*   **`DynamicRNN.py`**: Implementasi RNN yang mendukung input sequence dengan panjang yang bervariasi (*variable-length sequence*), yang biasa digunakan dalam pemrosesan teks NLP agar komputasi lebih efisien.

## 3. Ekstraksi Fitur Multi-Kanal (Sintaksis & Semantik)
*   **`dependency_graph.py`**: Script yang bertanggung jawab untuk membangun kanal Sintaksis (*Syntax channel*). Ini mengekstrak relasi antar kata secara struktural menggunakan *dependency parsing* (kemungkinan via SpaCy).
*   **`sentic_graph.py`**: Script yang bertanggung jawab untuk membangun kanal Semantik (*Semantics channel*). Ini menghubungkan kata-kata dalam kalimat berdasarkan polaritas sentimennya.
*   **`SentiWordNet_3.0.0.txt`**: Sebuah leksikon (kamus) sentimen yang memuat nilai positif dan negatif dari setiap kata (*synset*). Digunakan sebagai referensi oleh sistem graf semantik untuk mendeteksi kata-kata bersentimen.

## 4. Pemrosesan Data & Utilitas
*   **`dataUtils.py`**: Memuat kelas `DataManager` yang bertugas untuk membangun kosa kata (*vocabulary*), mengubah teks mentah menjadi tensor/indeks yang dapat dibaca oleh PyTorch, dan membuat skema *batching* untuk *training*.
*   **`preposs.py`**: (Preprocessing) Berisi fungsi-fungsi untuk membersihkan teks (seperti menghapus tanda baca tak penting, *tokenization*, dan menormalisasi *case*).
*   **`bucket_iterator.py`**: Teknik optimasi NLP untuk mengelompokkan kalimat-kalimat dengan panjang yang mirip ke dalam satu *batch*. Hal ini mengurangi jumlah *padding* (nol) yang berlebihan sehingga melatih model menjadi lebih cepat.
*   **`myutils.py`**: File pembantu (*helpers*) umum yang biasanya digunakan untuk manipulasi struktur array/tensor di berbagai tempat.

## 5. Evaluasi dan Prediksi
*   **`evaluation.py`**: Script yang berisi fungsi-fungsi penghitungan metrik performa penelitian yang standar, seperti *Accuracy, F1-Score (Micro/Macro), Precision, Recall*, dan *AUC*. Hasil dari evaluasi ini dicatat ke dalam log.
*   **`predict_sarcasm.py`**: Script yang digunakan untuk tahap *inference*, yaitu menggunakan bobot (*checkpoint*) dari model yang telah dilatih untuk memprediksi apakah kalimat baru mengandung sarkasme atau tidak.

## 6. Direktori (*Folder*) Pendukung
*   **`logs/`**: Menyimpan file `.log` hasil dari proses *training*. Sangat berguna dalam penelitian untuk melihat kurva pergerakan loss dan akurasi per *epoch*.
*   **`tensorboard/`**: Menyimpan rekaman data dari `tensorboard_logger` di dalam kode (contohnya `log_value` pada `main.py`). Memungkinkan Anda memvisualisasikan metrik evaluasi secara interaktif melalui antarmuka web.
*   **`IAC1/`**: Merupakan sampel dataset yang digunakan dalam arsitektur ini. IAC merujuk pada *Internet Argument Corpus*.

## Kesimpulan Pipeline untuk Penelitian
Berdasarkan kodenya, metode yang diusulkan oleh *repository* ini menggunakan pendekatan kombinasi: 
1. Teks diproses secara **sekuensial** menggunakan RNN (LSTM/GRU).
2. Bersamaan dengan itu, teks juga diubah menjadi **graf** (Dependency Graph & Sentic Graph).
3. Informasi sekuensial dan struktural digabung melalui layer *Attention / Graph Neural Networks (ADGCN)*.
4. Model (`bridgeModel.py`) dievaluasi dari berbagai segi: pemahaman kalimat literal, sentimen mendalam, hingga keputusan akhir berupa klasifikasi **sarkastis atau tidak**.
