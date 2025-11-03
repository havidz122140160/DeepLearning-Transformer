# Eksplorasi Arsitektur Transformer: Mesin Translasi English-French

## Kelompok Tugas Eksplorasi
* Havidz Ridho Pratama - 122140160
* Royfran Roger Valentino - 122140239

Tugas ini adalah implementasi arsitektur Transformer "from scratch" menggunakan PyTorch untuk tugas penerjemahan mesin. Fokus dari eksplorasi ini adalah pada kejelasan proses pembangunan model, mulai dari persiapan data hingga inferensi.

## Fokus Implementasi

Notebook ini dibagi menjadi 4 bagian utama sesuai dengan fokus penilaian:

### 1. Persiapan Data (Text Preprocessing)

* **Dataset**: Kami menggunakan subset 20.000 kalimat dari dataset English-French Kaggle.
* **Tokenization**: Kami menggunakan library `spacy` untuk tokenisasi bahasa Inggris dan Prancis.
* **Vocabulary**: Kami membangun dua "kamus" (Vocabulary) terpisah untuk bahasa sumber (Inggris) dan target (Prancis), hanya menyertakan kata-kata yang muncul minimal 2 kali (`freq_threshold=2`). Token khusus seperti `<PAD>`, `<SOS>`, `<EOS>`, dan `<UNK>` juga disertakan.
* **Dataset & DataLoader**: Sebuah `class TranslationDataset` kustom dibuat untuk mengubah pasangan kalimat menjadi tensor numerik. Kami juga mengimplementasikan `MyCollate` (sebuah `collate_fn`) untuk melakukan *padding* pada setiap *batch* agar memiliki panjang sekuen yang seragam.

### 2. Definisi Arsitektur Transformer

Untuk memenuhi kravia "kejelasan pendefinisian class", arsitektur Transformer diimplementasikan dari komponen-komponen dasarnya:

* `PositionalEncoding`: Menambahkan informasi posisi ke dalam embedding, karena Transformer tidak memiliki sifat sekuensial seperti RNN.
* `MultiHeadAttention`: Implementasi inti dari mekanisme *scaled dot-product attention*.
* `PositionwiseFeedForward`: Lapisan MLP yang ada di setiap blok encoder/decoder.
* `EncoderLayer` & `DecoderLayer`: Blok penyusun utama yang menggabungkan *attention*, *feed-forward*, dan *layer normalization*.
* `Seq2SeqTransformer`: Kelas utama yang menyatukan semua komponen (Embedding, Positional Encoding, Encoder, Decoder, dan Linear output).

### 3. Proses Pelatihan

* **Spesifikasi**: Sesuai tugas, pelatihan hanya dijalankan untuk **1 epoch** dengan `BATCH_SIZE=100`.
* **Loop**: Kami membuat fungsi `train_epoch` dan `evaluate` kustom.
* **Loss Function**: Menggunakan `CrossEntropyLoss` dengan `ignore_index=PAD_IDX` agar token padding tidak ikut dihitung ke dalam loss.
* **Masking**: Kami mengimplementasikan `src_mask` (untuk *padding*) dan `trg_mask` (untuk *padding* dan *look-ahead*) di dalam *forward pass* model.
* **Pelaporan per Batch**: Sesuai permintaan, skrip pelatihan menampilkan **TrainLoss**, **ValLoss**, dan **ValAcc** untuk setiap *batch* yang diproses. (Akurasi dihitung per token).

### 4. Proses Inferensi (Translation)

* Kami membuat fungsi `translate_sentence` untuk menerapkan model pada data baru.
* **Proses**: Fungsi ini bekerja secara *autoregressive*.
    1.  Encoder memproses kalimat sumber (input) sekali.
    2.  Decoder dimulai dengan token `<SOS>`.
    3.  Model memprediksi kata berikutnya.
    4.  Kata yang diprediksi tersebut ditambahkan ke input decoder.
    5.  Proses diulang hingga model memprediksi token `<EOS>` atau mencapai panjang maksimum.
* **Hasil**: Karena pelatihan hanya 1 epoch, hasil terjemahan masih sangat acak. Fokusnya adalah untuk menunjukkan bahwa pipeline inferensi berhasil dijalankan.
