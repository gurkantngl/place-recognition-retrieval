# Rapor

## Ozet

Bu calisma, gorsel icerik uzerinden yer tanima ve geri arama (retrieval) yapan hafif bir sistem sunar. Pretrained bir ResNet50 omurgasindan embedding cikarilir, L2 normalize edilir ve cosine benzerligi ile Top-K arama yapilir. Pipeline; bozuk gorseller, grayscale icerikler ve cok kucuk cozumunurluklere karsi dayanikli olacak sekilde tasarlanmistir. Embedding hesaplamasi cache ile hizlandirilir ve degerlendirme Recall@K ile mAP uzerinden raporlanir.

## Veri ve Split

Veri, `dataset/manifest.csv` uzerinden yonetilir ve her satirda `relpath`, `split`, `landmark_id` gibi alanlar bulunur. Split yapisi `gallery` ve `query` olarak ikiye ayrilmistir ve tanima sirasinda metadata (dosya adi, EXIF vb.) kullanilmaz. Gorsel yukleme asamasinda bozuk dosyalar, grayscale goruntuler ve cok kucuk cozumunurlukler (varsayilan 64 px alti) tespit edilip loglanir ve atlanir.

Split sizintisini azaltmak icin average hash tabanli near-duplicate kontrolu yapilmistir:

- Gallery hashes: 37
- Query hashes: 29
- Potential near-duplicates (<= 3): 3

Bu eslesmeler split sizintisi riski tasir; manuel inceleme ile gerekirse gallery veya query tarafindan cikarilabilir ya da esik daha sikilastirilabilir. Bu calismada split otomatik olarak degistirilmedi; 3 aday dosya manuel inceleme listesi olarak isaretlendi.

Ornek cikti:

```text
query=dataset\landmarks/Eiffel_Tower__lid_47378__line_47380/query/65522a18893d373d.jpg gallery=dataset\landmarks/Eiffel_Tower__lid_47378__line_47380/gallery/f556b084727eef23.jpg hamming=2
query=dataset\landmarks/Antkabir__lid_197321__line_197323/query/7a21a5ab7fa2a9a2.jpg gallery=dataset\landmarks/Grand_Canyon_National_Park__lid_44795__line_44797/gallery/7dc0d191ea6ca330.jpg hamming=3
```

## Yontem

### Embedding cikarma

- Omurga: ResNet50 (ImageNet pretrained).
- Preprocess: Torchvision agirliklarinin onerdigi donusumler.
- Batch inference: varsayilan `batch_size=32`.
- Normalizasyon: L2 normalization ile cosine benzerligi icin uygun embedding uretimi.

Embedding'ler `cache/embeddings/{split}` altinda saklanir. Manifest icerigi degistiginde yeniden hesaplama yapilmasini saglamak icin split hash tabanli bir `cache_key` kullanilir.

### Index ve arama

Gallery embedding'leri, label ve path bilgileri ile birlikte `cache/index` altinda saklanir. Arama asamasinda query embedding'leri ile gallery embedding'leri arasinda dot-product tabanli cosine similarity hesaplanir ve Top-K sonuclar dondurulur. CLI hem tek gorsel hem de klasor seklinde sorguya izin verir.

## Open-set (UNKNOWN) stratejisi

Open-set icin iki mekanizma desteklenir:

- `--threshold`: Top-1 benzerlik skoru esik altindaysa UNKNOWN.
- `--margin`: Top-1 ile Top-2 arasindaki fark kucukse UNKNOWN.

Kalibrasyon icin `scripts/calibrate_unknown.py` kullanilabilir. Mevcut calismada dogru eslesmelerin Top-1 skorlarinin p5 degeri 0.53 bulunmus ve esik onerisi bu sekilde yapilmistir. Esik arttikca yanlis kabul (false positive) azalir, ancak dogru eslesmelerin reddedilme riski (false negative) artar.

## Degerlendirme

Degerlendirme metrikleri:

- Recall@K: Ilk K sonucta dogru sinifin bulunma orani.
- mAP: Query bazinda average precision ortalamasi (multi-positive destegiyle).

Sonuclar:

- Recall@1: 0.6897
- Recall@5: 0.6897
- Recall@10: 0.6897
- mAP: 0.6220

Recall@K degerlerinin ayni cikmasi, ust siralarda cesitliligin sinirli olduguna isaret eder.

## Performans

Benchmark (warmup sonrasi):

| Donanim | Batch | Gallery img/s | Query img/s | Similarity (s) |
| --- | --- | --- | --- | --- |
| CPU | 16 | 3.64 | 3.57 | 0.0002 |
| CPU | 32 | 3.54 | 3.67 | 0.0007 |
| CPU | 64 | 3.70 | 3.80 | 0.0052 |
| GPU | 16 | 10.71 | 9.76 | 0.0012 |
| GPU | 32 | 11.70 | 10.49 | 0.0003 |
| GPU | 64 | 12.50 | 11.99 | 0.0004 |

Oneri: GPU uzerinde batch=64 en yuksek throughput degerini vermektedir.

## Testler

`tests/test_metrics.py` altinda en az iki metrik testi bulunmaktadir:

- Multi-positive query senaryosu.
- Kismi basarisizlik iceren senaryo.

## Hata modlari ve iyilestirmeler

- Goruntuleme acisi ve aydinlatma degisimleri performansi dusurebilir.
- Sezonsal degisimler ve gun/gece farklari icin daha guclu bir backbone (orn. CLIP) denenebilir.
- Hard negative mining ile benzer ama farkli yerlerin ayristirilmasi iyilestirilebilir.

## Reproduksiyon ve kullanim

Proje; `extract_embeddings.py`, `build_index.py`, `search.py`, `evaluate.py` ve destekleyici yardimci scriptler ile uctan uca calistirilabilir. Kurulum `requirements.txt` uzerinden yapilir ve tum adimlar README'de detaylandirilmistir.
