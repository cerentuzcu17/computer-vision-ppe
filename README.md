# computer-vision-ppe

Kişi-bazlı baret örtüşme tespiti. Bir görüntüdeki her insanı tespit edip pose
keypoint'lerine indirger, kafa bölgesini çıkarır; bu kafa bölgesi ile bir
baret kutusunun örtüşmesi (IoU) üzerinden kişiyi "baretli/baretsiz" etiketler.
Amaç: kalabalık sahnede bareti doğru kişiye atamak.

## Klasör yapısı

```
.
├── core/       # Algoritma mantığı: pose çıkarımı, kafa kutusu, IoU (pose_prototype.py)
├── dataset/    # Test/örnek girdi medyası (repoya girmez, bkz. .gitignore)
├── model/      # Model ağırlıkları — .pt dosyaları otomatik iner (repoya girmez)
├── observe/    # Çalıştırma çıktıları: işlenmiş görüntü/video, loglar (repoya girmez)
└── ui/         # İleride: görselleştirme / dashboard katmanı (şimdilik boş)
```

Bu ayrım, baret modeli ve arayüz eklendikçe her parçanın kendi klasöründe
büyümesini sağlamak için var: algoritma (`core`), veri (`dataset`/`model`),
sonuç izleme (`observe`), sunum (`ui`).

## Ortam — uv

Proje [uv](https://docs.astral.sh/uv/) ile yönetilir. Python sürümü
`.python-version` dosyasında pinlenmiştir (**3.12.5**).

```bash
# Bağımlılıkları kur (pyproject.toml + uv.lock üzerinden, .venv otomatik oluşur)
uv sync

# Komutları .venv içinde çalıştır
uv run core/pose_prototype.py --source dataset/ornek.jpg --save observe/ornek_out.jpg
```

### Bağımlılıklar

- `ultralytics` — YOLO-pose çıkarımı (bkz. aşağıda model ağırlığı notu)
- `opencv-python` — görüntü/video okuma, çizim
- `numpy` — geometri hesapları

**Model ağırlığı:** `ultralytics` paketi, `yolo11n-pose.pt` (veya `yolo11s-pose.pt`
gibi daha büyük varyantlar) ağırlığını ilk çalıştırmada otomatik indirir ve
`model/` klasörüne kaydeder. Ayrı bir pip bağımlılığı gerekmez; sadece internet
erişimi ve `model/` klasörüne yazma izni yeterlidir. Daha isabetli ama daha
yavaş bir model için `--weights model/yolo11s-pose.pt` kullanılabilir.

## Çalıştırma (core/pose_prototype.py)

```bash
# Resim
uv run core/pose_prototype.py --source dataset/ornek.jpg --save observe/ornek_out.jpg

# Video
uv run core/pose_prototype.py --source dataset/video.mp4 --save observe/video_out.mp4

# Webcam
uv run core/pose_prototype.py --source 0 --show
```

Argümanlar:

| Bayrak       | Açıklama                                             | Varsayılan               |
|--------------|-------------------------------------------------------|---------------------------|
| `--source`   | Resim/video yolu veya webcam için `0`                 | *(zorunlu)*               |
| `--weights`  | YOLO-pose ağırlığı                                     | `model/yolo11n-pose.pt`  |
| `--conf`     | Tespit güven eşiği                                     | `0.25`                    |
| `--save`     | Çıktı dosyası (resim/video)                            | yok                       |
| `--show`     | Pencerede göster                                       | kapalı                    |

## Durum

- [x] YOLO-pose ile kişi + keypoint tespiti
- [x] Kafa keypoint'lerinden kafa kutusu (fallback: insan kutusunun üst %25'i)
- [x] `helmet_iou()` iskeleti (baret modeli sonra eklenecek)
- [ ] Baret tespiti + kafa-baret eşleştirmesi
- [ ] `ui/` katmanı
