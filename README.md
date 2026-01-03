# etiket.py  
## PDF içinden sayfa resimleri çıkarma, etiket OCR ile adlandırma ve okunurluk iyileştirme (Windows 10/11)

Bu araç, **tarayıcıdan (scan) gelen PDF** dosyalarının her sayfasındaki **gömülü görüntüyü** kalite kaybı olmadan çıkarır, sayfa üzerindeki **sarı etiket içindeki kırmızı numarayı** OCR ile okur ve çıktıları **etiket numarasına göre adlandırır**.

Ayrıca:
- Microsoft Picture Manager’daki **“Orta ton -100”** etkisine benzer bir iyileştirme,
- IrfanView **Auto Adjust Colors** benzeri hafif bir auto-adjust

uygular. Amaç:  
📄 **form ve el yazılarının daha okunur olması**,  
🏷️ **etiket ve fotoğrafların bozulmaması**.

---

## Ne yapar?

- PDF içindeki **gömülü tarama resimlerini** çıkarır (yeniden render etmez)
- Sarı etiketi tespit eder, kırmızı rakamı OCR ile okur
- Dosya adını etikete göre verir:
  - `35830.jpg`
  - `35831.jpg`
- Etiket okunamazsa:
  - `page_001.jpg`, `page_002.jpg` olarak bırakır
- Çıktıyı:
  - PDF’nin bulunduğu klasörde
  - **PDF adıyla oluşturulan tek bir klasöre**
  yazar
- Klasör varsa **“Üzerine yazılsın mı?”** diye sorar
- İşlem boyunca **terminalde log yazar**
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

Etiket numarası OCR için gereklidir.

İndirme (Windows):
👉 [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Kurulumdan sonra kontrol:

```bat
tesseract --version
```

> Eğer `tesseract` komutu bulunamazsa:
>
> * PATH’e ekleyin
> * veya `etiket.py` içine şu satırı ekleyin:
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

1. `etiket.py` dosyasını bir klasöre koy
2. Gerekli Python kütüphanelerini kur
3. Tesseract OCR kurulu olduğundan emin ol

---

## Çalıştırma

```bat
python etiket.py
```

* PDF seçme penceresi açılır
* PDF seçilir
* Çıktılar otomatik üretilir

---

## Etiket numarası ayarları (ÖNEMLİ)

`etiket.py` dosyasının **en üstünde** şu ayarlar vardır:

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

* PDF seçtikten sonra
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

## Sık karşılaşılan sorunlar

### Etiket hiç okunmuyor

* Sarı etiket HSV aralığı farklı olabilir
* Kod içinde şu aralık ayarlanabilir:

```python
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
```

### Kırmızı rakam seçilemiyor

* Etiket baskısı farklıysa HSV kırmızı aralıkları ayarlanabilir

---

## Lisans

İhtiyacına göre ekleyebilirsin (örn. MIT).

---

## Not

Bu araç özellikle:

* Patoloji
* Endoskopi
* Laboratuvar
* Form + etiket içeren taramalar

için optimize edilmiştir.



