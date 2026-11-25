# CIFAR-10 ile Görüntü Sınıflandırma

Bu projede altı evrişim katmanlı bir CNN, CIFAR-10 görüntülerini 10 sınıfa ayırıyor. Gradio uygulaması fotoğraf yüklemeyi ve modelin en yüksek üç sınıf skorunu görmeyi sağlıyor.

## Uygulamayı çalıştırma

Proje klasöründeki mevcut Windows ortamını kullan:

```powershell
.\.venv\Scripts\python.exe app.py
```

Tarayıcıda [yerel uygulamayı](http://127.0.0.1:7860) aç. Uygulama `results/best.pt` dosyasındaki eğitilmiş modeli CPU’da çalıştırır. Sınıflar uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi ve kamyondur.

## Başka bir ortamda kurulum

Python 3.11 ile yeni bir sanal ortam oluştur. Uygulamayı çalıştırmak ve CPU ile eğitim yapmak için:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

NVIDIA GPU ile eğitim için PyTorch kurulumunda `/whl/cu128` dizinini kullan. Varsayılan eğitim ayarı CUDA kullanılabiliyorsa GPU’yu seçer. [PyTorch kurulum seçenekleri](https://pytorch.org/get-started/previous-versions/)

## Eğitimi ve raporu yeniden üretme

```powershell
.\.venv\Scripts\python.exe train.py --output results/yeni_egitim
.\.venv\Scripts\python.exe evaluate.py --output results/yeni_egitim
.\.venv\Scripts\python.exe report.py --output results/yeni_egitim --pdf results/yeni_egitim/rapor.pdf
```

Eğitim ilk çalıştırmada [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) arşivini indirir. Resmî eğitim kümesinden sınıf dengesi korunarak 45.000 eğitim ve 5.000 doğrulama görüntüsü ayrılır. Resmî 10.000 görüntülük test kümesi değerlendirme komutunda kullanılır. Eğitim 35 epoch sürer. Her epoch sonunda bir devam kaydı alınır. Kesilen bir eğitime aynı komut ve ayarlarla `--resume` ekleyerek devam edebilirsin.

Teslimde kullanılan eğitimin çıktıları `results/` klasöründedir. `training.json` epoch geçmişini, hiperparametreleri ve çalışma ortamını içerir. `evaluation.json` test metriklerini, sınıf bazında sonuçları ve tahmin sürelerini içerir. Eğitim eğrileri ve karışıklık matrisi PNG olarak kaydedilir. PDF raporunun depo alanı için `report.py --repository` seçeneğine depo URL’sini ver.

## Telefon fotoğrafları

Öğrencinin kendi çektiği beş farklı fotoğrafı bir klasöre koy. Aynı klasörde UTF-8 ile kaydedilmiş bir CSV dosyası açıp `image,label` başlıklarını ve her fotoğrafın dosya adı ile gerçek CIFAR-10 etiketini yaz. Aynı sınıf birden fazla karede yer alabilir.

```powershell
.\.venv\Scripts\python.exe photos.py telefon_fotograflari/etiketler.csv
.\.venv\Scripts\python.exe report.py --phone-notes telefon_fotograflari/yorum.txt
```

Fotoğraf değerlendirmesi varsayılan olarak uygulamadaki modeli kullanır. Başarı ve hata örnekleri hakkındaki gözlemleri `yorum.txt` dosyasına yaz. PDF’yi depo bağlantısıyla güncellerken rapor komutuna `--repository` seçeneğini de ekle. Telefon fotoğrafları klasörü yerel tutulur.
