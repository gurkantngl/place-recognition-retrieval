# Place Recognition Retrieval

Bu proje, gorsellerden embedding cikarip benzerlik aramasi ile yer tanima yapar.

## Kurulum

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Veri Yapisi

- `dataset/manifest.csv` icinde `relpath`, `split`, `landmark_id` alanlari kullanilir.
- Gorseller `dataset/` altinda `relpath` ile cozulur.

## Embedding Cikarma

```bash
python scripts/extract_embeddings.py --manifest dataset/manifest.csv --dataset-root dataset --split gallery
python scripts/extract_embeddings.py --manifest dataset/manifest.csv --dataset-root dataset --split query
```

## Index Olusturma

```bash
python scripts/build_index.py --cache-dir cache/embeddings --index-dir cache/index
```

## Arama

```bash
python scripts/search.py --query "dataset/landmarks/Grand_Canyon_National_Park__lid_44795__line_44797/query/33ced200ba700719.jpg" --top-k 5
python scripts/search.py --query "dataset/landmarks/Grand_Canyon_National_Park__lid_44795__line_44797/query" --top-k 5 --threshold 0.25
```

## Near-Duplicate Kontrol

```bash
python scripts/check_duplicates.py --manifest dataset/manifest.csv --dataset-root dataset --hamming-threshold 3
```

Son calisma ozeti (threshold=3):
- Gallery hashes: 37
- Query hashes: 29
- Potential near-duplicates (<= 3): 3

## Degerlendirme

```bash
python scripts/evaluate.py --cache-dir cache/embeddings --k-values 1,5,10
```

Sonuclar (rapordan):
- Recall@1: 0.6897
- Recall@5: 0.6897
- Recall@10: 0.6897
- mAP: 0.6220

## Benchmark

```bash
python scripts/benchmark.py --manifest dataset/manifest.csv --dataset-root dataset --cache-dir cache/embeddings --device cpu --batch-size 32
```

### Sonuc Ozeti (warmup)

- CPU:
  - batch=16: gallery 3.64 img/s, query 3.57 img/s, similarity 0.0002s
  - batch=32: gallery 3.54 img/s, query 3.67 img/s, similarity 0.0007s
  - batch=64: gallery 3.70 img/s, query 3.80 img/s, similarity 0.0052s
- GPU:
  - batch=16: gallery 10.71 img/s, query 9.76 img/s, similarity 0.0012s
  - batch=32: gallery 11.70 img/s, query 10.49 img/s, similarity 0.0003s
  - batch=64: gallery 12.50 img/s, query 11.99 img/s, similarity 0.0004s
- Oneri: GPU icin batch=64.

## Testler

```bash
pytest -q
```

## Notlar

- Cache dogrulamasi split icerikleri ile yapilir; manifest degisirse yeniden hesaplanir.
- Open-set icin `--threshold` veya `--margin` kullanilabilir. Kalibrasyon icin `scripts/calibrate_unknown.py` kullanilabilir.
