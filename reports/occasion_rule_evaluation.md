# Occasion rule engine — evaluation

Gold set: **300 products**, stratified over Category x price band.

> **Label provenance: machine annotator (archetype-based), not human review.**
> These figures measure agreement between two independent labelling methods,
> not accuracy against human judgement. See docs/decisions/0007.

## Per-label results

| Label | Support | Gold rate | Rule rate | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Birthday | 294 | 98.0% | 98.0% | 0.980 | 0.980 | 0.980 |
| Graduation | 150 | 50.0% | 59.0% | 0.706 | 0.833 | 0.765 |
| Anniversary | 72 | 24.0% | 20.3% | 0.639 | 0.542 | 0.586 |
| Wedding | 53 | 17.7% | 18.7% | 0.804 | 0.849 | 0.826 |
| Eid | 285 | 95.0% | 98.0% | 0.969 | 1.000 | 0.984 |
| MothersDay | 105 | 35.0% | 30.7% | 0.815 | 0.714 | 0.761 |
| FathersDay | 32 | 10.7% | 12.7% | 0.395 | 0.469 | 0.429 |
| NewBaby | 43 | 14.3% | 12.0% | 0.889 | 0.744 | 0.810 |
| Housewarming | 28 | 9.3% | 6.7% | 1.000 | 0.714 | 0.833 |
| ThankYou | 46 | 15.3% | 15.7% | 0.787 | 0.804 | 0.796 |

## Overall

| Metric | Value |
|---|---:|
| Macro-F1 | 0.777 |
| Micro-F1 | 0.865 |
| Hamming loss | 0.100 |
| Exact match | 33.7% |

