-- Useful PostgreSQL metadata inspection queries.
-- Unique source images and quality statuses by official split:
SELECT dataset, split, status, COUNT(*) AS images
FROM image_assets GROUP BY dataset, split, status ORDER BY dataset, split, status;

-- Total augmented objects stored in S3:
SELECT i.dataset, a.recipe_version, COUNT(*) AS uploaded_variants
FROM augmentations a JOIN image_assets i ON i.id = a.asset_id
WHERE a.status='ready' GROUP BY i.dataset, a.recipe_version;

-- Class balance by data source (train only):
SELECT dataset, label, COUNT(*) AS n FROM image_assets
WHERE split='train' AND status='ready'
GROUP BY dataset, label ORDER BY dataset, label;

-- Benchmark runs and full-vs-partial-test provenance:
SELECT dataset, mode, seed, epochs, full_test,
       ROUND((100*test_accuracy)::numeric,2) AS test_accuracy_pct,
       ROUND(test_macro_f1::numeric,4) AS macro_f1,
       created_at
FROM experiments ORDER BY created_at DESC;

-- Potential exact train/test leakage (should be removed from *train* by dedup):
SELECT tr.dataset, COUNT(*) AS overlapping_pairs
FROM image_assets tr JOIN image_assets te
  ON tr.dataset=te.dataset AND tr.sha256=te.sha256
WHERE tr.split='train' AND te.split='test' AND tr.status='ready'
GROUP BY tr.dataset;
