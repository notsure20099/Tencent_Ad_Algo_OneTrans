---
license: cc-by-nc-4.0
---

# TAAC2026 Demo Dataset (1000 Samples)

A sample dataset containing 1000 user-item interaction records for the [TAAC2026 competition](https://algo.qq.com).

## Dataset Description

- **Rows**: 1,000
- **Format**: Parquet (`sample_data.parquet`)
- **File Size**: ~68 MB

## Columns

| Column | Type | Description |
|---|---|---|
| `item_id` | `int64` | **Target item** identifier. |
| `item_feature` | `array[struct]` | Array of **target item** feature dicts. Each element has `feature_id`, `feature_value_type`, and value fields (`float_value`, `int_array`, `int_value`). |
| `label` | `array[struct]` | Array of label dicts. Each element contains `action_time` and `action_type`. |
| `seq_feature` | `struct` | Sequence features dict with keys: `action_seq`, `content_seq`, `item_seq`. Each sub-key contains arrays of feature structs. |
| `timestamp` | `int64` | Event timestamp. |
| `user_feature` | `array[struct]` | Array of user feature dicts. Each element has `feature_id`, `feature_value_type`, and value fields (`float_array`, `int_array`, `int_value`). |
| `user_id` | `string` | User identifier. |

## Feature Struct Schema

Each feature element contains `feature_id`, `feature_value_type`, and several value fields. Depending on `feature_value_type`, the corresponding value fields are populated and the rest are `null`.

**`item_feature`** — value fields: `int_value`, `float_value`, `int_array`

```json
{
  "feature_id": 6,
  "feature_value_type": "int_value",
  "float_value": null,
  "int_array": null,
  "int_value": 96,
}
```

**`user_feature`** — value fields: `int_value`, `float_array`, `int_array`

```json
{
  "feature_id": 65,
  "feature_value_type": "int_value",
  "float_array": null,
  "int_array": null,
  "int_value": 19
}
```

**`seq_feature`** — value fields: `int_array`

```json
{
  "feature_id": 19,
  "feature_value_type": "int_array",
  "int_array": [1, 1, 1, ...]
}
```

Possible `"feature_value_type"` values and their corresponding fields:
- `"int_value"` → `int_value` 
- `"float_value"` → `float_value` 
- `"int_array"` → `int_array` 
- `"float_array"` → `float_array`
- Also there are some combinations of these types, e.g. `"int_array_and_float_array"` → both `int_array` and `float_array` are populated.

## Label Schema

Each element in the `label` array:

```json
{
  "action_time": 1770694299,
  "action_type": 1
}
```

## Usage

```python
import pandas as pd

df = pd.read_parquet("sample_data.parquet")
print(df.shape)       # (1000, 7)
print(df.columns)     # ['item_id', 'item_feature', 'label', 'seq_feature', 'timestamp', 'user_feature', 'user_id']
```

With Hugging Face `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("TAAC2026/data_sample_1000")
print(ds)
```
