# Annotation guidelines - occasion labelling

## The question you are answering

For each product, for each occasion, answer one question:

> **Would giving this item at this occasion be socially acceptable?**

Not "is this the ideal gift". Not "is this an X-themed product". Acceptable.

Mark `1` for yes, `0` for no. Never leave a cell empty.

## Rules

1. **Multi-label.** A watch can be `1` for Birthday, Graduation and Anniversary
   at once. Most products will have several.
2. **Judge the product, not the shop.** Ignore which retailer it came from.
3. **Judge from the name, category and price.** Open the URL only when the name
   is genuinely opaque.
4. **Price matters.** A 40 SAR keychain is not a Wedding gift. A 12,000 SAR
   bracelet is not a Thank You gift.
5. **Cultural context is Saudi/Gulf.** Eid is a major gifting occasion across
   all ages. Valentine's is not observed in the same way.
6. **When you cannot decide, mark `0`** and write why in `notes`. A forced
   guess pollutes the ground truth; a note tells us the rule needs work.

## Occasion definitions

| Label | Give it to | Typical range |
|---|---|---|
| Birthday | Anyone | Any price |
| Graduation | Student finishing a stage, 16+ | Mid to high |
| Anniversary | Romantic partner, 18+ | Mid to high |
| Wedding | Couple or household | Mid to high, often household goods |
| Eid | Anyone, all ages, family-wide | Any price; sweets, perfume, clothing, toys |
| MothersDay | Mother figure | Any price |
| FathersDay | Father figure | Any price |
| NewBaby | Usually the **parents**, sometimes the infant | Low to mid |
| Housewarming | New home owner or household | Low to mid |
| ThankYou | Colleague, host, helper | Low to mid, never intimate |

## Audit columns

- `audit_gender` - is the stated Gender Target right? Both `Gender Target` and
  `Age Group` in the source catalogue were built by keyword matching on the
  product name and their accuracy is unverified. This column measures it.
- `audit_min_age` - the youngest age at which this gift makes sense.
- `notes` - anything wrong: wrong category, dead link, duplicate, unsafe item.

## Rules for the held-out set (`gold_test.csv`)

- Label it **without** looking at `gold_dev.csv`, the pipeline output, or
  `configs/taxonomy.yaml`.
- Label it **once**. Do not revise it after seeing model results. Revising a
  test set to match your model is how benchmarks become meaningless.
- If a definition above turns out to be unclear, fix the definition, then
  relabel from scratch - do not patch individual rows.
