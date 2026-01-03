# multi_etiket.py  
## PDF klasörleri için etiket OCR + sayfa resimlerini adlandırma (Windows 10 / 11)

Bu araç, **tarayıcıdan (scan) gelen PDF dosyalarını** tek tek veya **toplu (multi-PDF)** olarak işler.

PDF içindeki **gömülü sayfa görüntülerini** kalite kaybı olmadan çıkarır, sayfa üzerindeki **sarı etiket içindeki kırmızı numarayı** OCR ile okur ve çıktıları **etiket numarasına göre adlandırır**.

Ayrıca:
- Microsoft Picture Manager’daki **“Orta ton -100”** etkisine benzer bir iyileştirme
- IrfanView **Auto Adjust Colors** benzeri hafif auto-adjust

uygular. Amaç:  
📄 yazı ve el yazılarının daha okunur olması,  
🏷️ etiket ve fotoğrafların bozulmaması.

---

## Ne yapar?

- PDF içindeki **gömülü tarama görüntülerini** çıkarır (sayfayı yeniden render etmez)
- Sarı etiketi tespit eder, kırmızı rakamı OCR ile okur
- Dosyaları etikete göre adlandırır:
  - `35830.jpg`
  - `35831.jpg`
- Etiket okunamazsa:
  - `page_001.jpg`, `page_002.jpg` olarak bırakır
- Çıktıyı:
  - PDF’nin bulunduğu klasörde
  - **PDF adıyla oluşturulan tek bir klasöre**
  yazar
- Tek PDF veya **klasör içindeki tüm PDF’leri (alt klasörler dahil)** işleyebilir
- Multi-PDF modunda **global politika** ile “üzerine yaz / atla / tek tek sor” seçimi yapılabilir
- Terminalde ayrıntılı log yazar
- İş bitince popup göstermez, sadece kısa bir **bip** sesi verir

---

## Örnek çıktı yapısı

PDF:
```

C:\Belgeler\Patoloji\02.01.2026.pdf

```

Çıktı:
```

C:\Belgeler\Patoloji\02.01.2026
35830.jpg
35831.jpg
page_003.jpg
35832_2.jpg

````

---

## Gereksinimler (Windows 10 / 11)

### 1️⃣ Python
- Python **3.9 veya üzeri** önerilir

Kontrol:
```bat
python --version
pip --version
````

---

### 2️⃣ Tesseract OCR (ZORUNLU)

Etiket numarasını okumak için gereklidir.

İndirme (Windows):
👉 [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Kurulumdan sonra kontrol:

```bat
tesseract --version
```

> Eğer `tesseract` komutu bulunamazsa:
>
> * PATH’e ekleyin
> * veya `multi_etiket.py` içine şu satırı ekleyin:
>
> ```python
> pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
> ```

---

### 3️⃣ Python kütüphaneleri (ZORUNLU)

```bat
pip install pymupdf opencv-python numpy pytesseract
```

---

## Kurulum

1. `multi_etiket.py` dosyasını bir klasöre koy
2. Gerekli Python kütüphanelerini kur
3. Tesseract OCR kurulu olduğundan emin ol

---

## Çalıştırma

```bat
python multi_etiket.py
```

Başlangıçta program sorar:

* **Tek PDF mi?**
* **Klasör modu mu?**

---

## Multi-PDF (Klasör) Modu

Klasör modu seçildiğinde:

* Seçilen klasörün içindeki **tüm PDF’ler**
* **Alt klasörler dahil**
* Sırayla işlenir

---

## Global politika (ÖNEMLİ)

Multi-PDF modunda, başta **tek sefer** şu soru sorulur:

**“Çıktı klasörü zaten varsa ne yapalım?”**

Seçenekler:

* **Yes** → Tüm PDF’ler için **üzerine yaz**
* **No** → Tüm PDF’ler için **atla**
* **Cancel** → **Her PDF için tek tek sor**

Bu sayede:

* Büyük klasörlerde sürekli popup çıkmaz
* Kontrol tamamen kullanıcıdadır

---

## Etiket numarası ayarları

`multi_etiket.py` dosyasının **en üstünde** bulunur:

```python
LABEL_MIN_DEFAULT = 100
LABEL_MAX_DEFAULT = 45000
```

Bu aralık:

* Yanlış OCR sonuçlarının dosya adını bozmasını önler
* Kurum / dönem / cihaz değiştikçe güncellenebilir

### GUI ile min–max sormak (opsiyonel)

Varsayılan olarak **kapalıdır**.

Açmak için:

```python
ASK_LABEL_RANGE_GUI = True
```

Bu durumda:

* Program başında
* Etiket min–max aralığı GUI üzerinden sorulur

---

## Görüntü iyileştirme ayarları

Yine dosyanın en üstünde bulunur:

```python
MIDTONE_GAMMA = 0.88
CONTRAST = 6
APPLY_IRFAN_AUTO_ADJUST = True
```

### Ayar önerileri

* **Kurşun kalem yazılar silikse**:

  * `CONTRAST = 4`
  * veya `MIDTONE_GAMMA = 0.90`
* **Fotoğraflar fazla patlıyorsa**:

  * `IRFAN_HIGH_PCT = 98.5`

---

## ImageMagick (magick) gerekli mi?

❌ **Hayır.**

Bu script:

* ImageMagick
* `magick` komutu

**kullanmaz**.

Tüm işlemler:

* Python
* OpenCV
* PyMuPDF

ile yapılır.

---

## Kimler için uygun?

Özellikle:

* Patoloji
* Endoskopi
* Laboratuvar
* Form + etiket içeren arşiv taramaları

için optimize edilmiştir.

---

## Lisans

 MIT

