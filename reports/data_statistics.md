# Data Statistics Report

Generated: 2026-02-01 17:32:45

## Dataset Overview

| Metric | Count |
|--------|-------|
| Original Records | 2,775 |
| After Empty Removal | 2,612 |
| After Length Filter | 2,608 |
| After Deduplication | 863 |

## Data Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 690 | 80% |
| Eval | 86 | 10% |
| Test | 87 | 10% |
| **Total** | 863 | 100% |

## Question Length Statistics

| Metric | Value |
|--------|-------|
| Minimum | 19 chars |
| Maximum | 2,690 chars |
| Average | 320.5 chars |

## Answer Length Statistics

| Metric | Value |
|--------|-------|
| Minimum | 51 chars |
| Maximum | 5,498 chars |
| Average | 1,023.3 chars |

## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
| depression | 136 | 15.8% |
| intimacy | 108 | 12.5% |
| relationships | 104 | 12.1% |
| anxiety | 100 | 11.6% |
| family-conflict | 60 | 7.0% |
| parenting | 54 | 6.3% |
| self-esteem | 42 | 4.9% |
| relationship-dissolution | 33 | 3.8% |
| behavioral-change | 31 | 3.6% |
| anger-management | 26 | 3.0% |
| trauma | 24 | 2.8% |
| marriage | 20 | 2.3% |
| domestic-violence | 16 | 1.9% |
| lgbtq | 15 | 1.7% |
| social-relationships | 12 | 1.4% |
| workplace-relationships | 11 | 1.3% |
| substance-abuse | 10 | 1.2% |
| grief-and-loss | 8 | 0.9% |
| spirituality | 7 | 0.8% |
| counseling-fundamentals | 7 | 0.8% |
| legal-regulatory | 6 | 0.7% |
| professional-ethics | 6 | 0.7% |
| sleep-improvement | 5 | 0.6% |
| eating-disorders | 5 | 0.6% |
| addiction | 4 | 0.5% |
| human-sexuality | 4 | 0.5% |
| stress | 3 | 0.3% |
| diagnosis | 3 | 0.3% |
| children-adolescents | 2 | 0.2% |
| military-issues | 1 | 0.1% |

## Cleaning Rules Applied

1. **Empty Removal**: Removed records with empty question or answer
2. **Length Filter**:
   - Minimum question length: 10 characters
   - Minimum answer length: 50 characters
3. **Deduplication**: Based on question text (case-insensitive)

## Output Files

- `data/processed/counsel_chat_train.jsonl` - Training data
- `data/processed/counsel_chat_eval.jsonl` - Evaluation data
- `data/processed/counsel_chat_test.jsonl` - Test data
- `data/processed/counsel_chat_cleaned.jsonl` - All cleaned data (combined)

## Record Format

```json
{
    "id": "counsel_00001",
    "question": "User's question text",
    "answer": "Counselor's response text",
    "topic": "Topic category"
}
```
