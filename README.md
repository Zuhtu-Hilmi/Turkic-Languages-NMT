# 🌍 Türk Dilleri Arası Nöral Makine Çevirisi (NMT) Algoritması

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FP16-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Spaces_&_Hub-ffc107.svg)](https://huggingface.co/)

Bu proje, Ankara Üniversitesi Bilgisayar Mühendisliği Bölümü Bitirme Projesi kapsamında, **Türkçe-Azerbaycanca (TR-AZ)** ve **Türkçe-Türkmence (TR-TK)** dilleri arasında İngilizce gibi aracı bir dil (pivot) kullanmadan, doğrudan çeviri yapabilen çok dilli bir Nöral Makine Çevirisi (NMT) sistemi geliştirmek amacıyla hazırlanmıştır.

🚀 **[CANLI DEMO (Web Arayüzü)](https://huggingface.co/spaces/Zuhtu-Hilmi/Turkic-Translator)** 📦 **[MODEL AĞIRLIKLARI (Hugging Face Hub)](https://huggingface.co/Zuhtu-Hilmi/TurkDilleriNMT)**

---

## 📌 Proje Özeti ve Motivasyon
Günümüzdeki ticari makine çevirisi sistemleri, düşük kaynaklı Türk dilleri arasında çeviri yaparken genellikle İngilizceyi köprü (pivot) dil olarak kullanmaktadır. Analitik bir dil olan İngilizcenin araya girmesi, sondan eklemeli dillerimizin morfolojik yapısını bozmakta ve anlam kaymalarına yol açmaktadır.

Bu projede, Meta AI tarafından geliştirilen **NLLB-200-distilled-600M** mimarisi temel alınmış ve sistem, **LoRA (Low-Rank Adaptation)** yöntemi ile parametre-verimli bir şekilde (PEFT) kendi derlediğimiz veri havuzu üzerinde ince-ayara (fine-tuning) tabi tutulmuştur. 

## 🛠️ Teknik Altyapı ve Hibrit Veri Mühendisliği
Projenin kalbini, kısıtlı kaynaklara sahip Türkmence ve Azerbaycanca için tasarlanan 3 katmanlı hibrit veri mühendisliği oluşturmaktadır:
* **Mimari:** NLLB-200 Transformer + LoRA (Sadece %1,39 oranında, 8.65M parametre eğitilmiştir).
* **Donanım İsteri:** Eğitim süreci Tesla T4 GPU (15.6 GB VRAM) üzerinde Gradient Checkpointing ve FP16 optimizasyonu ile OOM (Out-of-Memory) hatası yaşanmadan tamamlanmıştır.
* **Veri Havuzu (66.951 Satır):** * *Doğrudan Paralel:* TurkicNLP (UD Treebanks), TED2020, Tatoeba, Wikimedia.
  * *Pivot Madenciliği:* İngilizce köprüsü kullanılarak algoritmik olarak eşleştirilmiş organik veriler.
  * *LLM Sentetik Büyütme:* Gemini API kullanılarak üretilen Alpaca (Talimat/Komut tabanlı) ve OpenSubtitles (Günlük dil) veri setleri.

## 📊 Başarım Sonuçları (Sayısal Analiz)
### 1. Dahili Test Kümesi (İç Dağılım)
İnce-ayar sonucunda model, kendi referans noktası olan taban (zero-shot) başarımı üzerinde muazzam bir artış sergilemiştir. Özellikle Türkmence yönündeki +27.00 BLEU puanlık sıçrama, sentetik Alpaca verisinin model üzerindeki doğrudan başarısını kanıtlamaktadır.

| Çeviri Yönü | Taban Model (Zero-Shot) | İnce-Ayarlı Model (Fine-Tuned) | İyileşme | chrF | METEOR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TR ➔ AZ** (İç Test) | 14.40 | **25.19** | `+10.79` | 54.04 | 0.5354 |
| **TR ➔ TK** (İç Test) | 12.56 | **39.56** | `+27.00` | 65.79 | 0.6248 |

### 2. FLORES-200 Küresel Benchmark (Alan Dışı Dağılım)
Model, eğitim verisinden tamamen bağımsız, akademik ve gazetecilik metinlerinden oluşan zorlu FLORES-200 veri setinde de test edilmiştir.

| Çeviri Yönü | Taban Model (Zero-Shot) | İnce-Ayarlı Model (Fine-Tuned) | İyileşme |
| :--- | :---: | :---: | :---: |
| **TR ➔ AZ** (FLORES) | 9.96 | **10.80** | `+0.84` |
| **TR ➔ TK** (FLORES) | 7.18 | **11.34** | `+4.16` |

> **💡 Alan Kayması (Domain Shift) Notu:** FLORES-200 skorlarının dahili test kümesine göre daha düşük kalması bir başarısızlık değil; *Alan Kayması (Domain Shift)* olgusunun sonucudur. Eğitim verimizin %60'ını komut-cevap (Alpaca) formatındaki sentetik veriler oluştururken, FLORES-200 akademik ve edebi metinlerden oluşmaktadır. İnce-ayarlı model, anlamı mükemmel korusa da FLORES'in beklediği spesifik edebi/akademik eşanlamlıları kullanmadığı için n-gram tabanlı BLEU metriği tarafından cezalandırılmaktadır (Detaylı nitel analiz proje raporunda mevcuttur).

## 🔍 Nitel Çeviri Analizi (Örnek Çıktılar)
İnce-ayarlı model, sayısal başarısının yanı sıra morfolojik zaman kiplerini, resmi/akademik üslubu ve kelime seçimlerini korumakta üstün bir yetenek kazanmıştır.

**🇹🇷 Türkçe ➔ 🇦🇿 Azerbaycanca (Zaman Kipi Hassasiyeti):**
| Kaynak Cümle (TR) | Taban Model (Zero-Shot) | İnce-Ayarlı Model (Bizim) |
| :--- | :--- | :--- |
| *Hükümet, ekonomik kalkınma için yeni politikalar açıkladı.* | Hökumət iqtisadi inkişaf üçün yeni siyasətlər açıqlayıb. *(Belirsiz Geçmiş)*| Hökumət iqtisadi inkişaf üçün yeni siyasətlər **açıqladı.** *(Kesin Geçmiş)*|

**🇹🇷 Türkçe ➔ 🇹🇲 Türkmence (Kelime Dağarcığı ve Akademik Ton):**
| Kaynak Cümle (TR) | Taban Model (Zero-Shot) | İnce-Ayarlı Model (Bizim) |
| :--- | :--- | :--- |
| *Gelecek yıl yeni bir bilgisayar almayı planlıyorum.* | Men geljek ýyl täze kompýuteri aljak bolýaryn. *(Zayıf Yapı)* | Geljek ýylda täze kompýuter **satyn almagy maksat edinýärin.** *(Akademik)*|

---

## 💻 Yerel Kurulum ve Çıkarım (Local Inference)
Modeli kendi bilgisayarınızda (GPU gerektirmeden) çalıştırmak ve test etmek için aşağıdaki Python betiğini kullanabilirsiniz:

### Gereksinimler
```bash
pip install torch transformers sentencepiece
```

## Örnek Kullanım Kodu
```bash
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

MODEL_ID = "Zuhtu-Hilmi/TurkDilleriNMT" 
BASE_MODEL = "facebook/nllb-200-distilled-600M"

# Tokenizer ve Mimari facebook üzerinden, eğitilmiş ağırlıklar kendi depomuzdan çekilir
print("Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
config = AutoConfig.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, config=config)

# Çevrilecek metin ve dil ayarları
kaynak_metin = "Proje sunumuma katıldığınız için hepinize çok teşekkür ederim."
hedef_dil = "tuk_Latn" # Azerbaycanca için: "azj_Latn"
tokenizer.src_lang = "tur_Latn"

# Model Çıkarımı
inputs = tokenizer(kaynak_metin, return_tensors="pt", max_length=128)
forced_bos_token_id = tokenizer.convert_tokens_to_ids(hedef_dil)

with torch.no_grad():
    ciktilar = model.generate(
        **inputs, 
        forced_bos_token_id=forced_bos_token_id, 
        max_length=128,
        num_beams=4 # Daha kaliteli çeviri için
    )
    
ceviri = tokenizer.batch_decode(ciktilar, skip_special_tokens=True)[0]
print(f"🇹🇷 TR: {kaynak_metin}")
print(f"🇹🇲 TK: {ceviri}")
# Beklenen Çıktı: Taslamamyň çykyşyna gatnaşandygyňyz üçin hemmäňize örän minnetdar.
```
