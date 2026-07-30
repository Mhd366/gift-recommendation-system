# Cleaning report

| Stage | Rows before | Rows after | Removed |
|---|---:|---:|---:|
| drop missing price | 56,280 | 55,627 | 653 |
| drop free items | 55,627 | 55,210 | 417 |
| drop price outliers | 55,210 | 55,069 | 141 |
| deduplicate variants | 55,069 | 48,924 | 6,145 |

**56,280 in -> 48,924 out (86.9% retained)**

## Notes

- dropped column `Currency` - single constant value (SAR)
- dropped column `Interest Category` - byte-identical to Category in 100% of rows
- dropped column `Recommendation Tags` - string concatenation of six other columns
- dropped column `Data Quality` - metadata describing nulls, not a product attribute
- dropped column `Recipient Type` - conflates gender, age and household axes (ADR 0005)
- dropped column `Price Tier` - two incompatible binning schemes; recomputed from price
- dropped column `Luxury Level` - derived from price; recomputed deterministically
- dropped column `Source File` - scrape provenance, not a product attribute
- Category vocabulary 24 -> 14 values
- Gift Type vocabulary 15 -> 12 values
- safety floor min_age>=13 applied to 6,487 rows from `Steam` (no content rating available)
- 653 rows had no price; price is a hard constraint and cannot be imputed
- 417 zero-price items removed (free-to-play Steam titles); a free item is not a gift
- 141 items above 50,000 SAR removed (outside any realistic gift budget)
- price_band recomputed from numeric price using one consistent scheme
- variants collapsed on ['Product Name', 'Brand', 'Price (SAR)', 'Source Site'], keeping the cheapest representative
- occasion `Birthday`: 48,163 products (98.4%)
- occasion `Graduation`: 27,367 products (55.9%)
- occasion `Anniversary`: 16,690 products (34.1%)
- occasion `Wedding`: 13,791 products (28.2%)
- occasion `Eid / Religious`: 28,676 products (58.6%)
- occasion `Mother's Day`: 15,674 products (32.0%)
- occasion `Father's Day`: 16,141 products (33.0%)
- occasion `New Baby`: 18,894 products (38.6%)
- occasion `Housewarming`: 6,255 products (12.8%)
- occasion `Thank You`: 8,295 products (17.0%)
- `General / Any Occasion` dropped as a label - it appeared on 96.7% of raw rows and carried no discriminative power
- 1,104 products lack an image; flagged rather than dropped so the ranker can demote instead of losing catalogue coverage
