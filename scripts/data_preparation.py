#!/usr/bin/env python3
"""
Data Preparation Script for Counsel Chat Dataset

Downloads and cleans the counsel-chat dataset from HuggingFace,
preparing it for DPO training pipeline.

Usage:
    python scripts/data_preparation.py [--output-dir data/processed] [--min-question-len 10] [--min-answer-len 50]
"""

import argparse
import json
import logging
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreparation:
    """Handles downloading, cleaning, and splitting the Counsel Chat dataset."""

    def __init__(
        self,
        output_dir: str = "data/processed",
        min_question_len: int = 10,
        min_answer_len: int = 50,
        train_ratio: float = 0.8,
        eval_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize data preparation.

        Args:
            output_dir: Directory for output files
            min_question_len: Minimum character length for questions
            min_answer_len: Minimum character length for answers
            train_ratio: Ratio for training split
            eval_ratio: Ratio for evaluation split
            test_ratio: Ratio for test split
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.min_question_len = min_question_len
        self.min_answer_len = min_answer_len
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            "original_count": 0,
            "after_empty_removal": 0,
            "after_length_filter": 0,
            "after_dedup": 0,
            "train_count": 0,
            "eval_count": 0,
            "test_count": 0,
            "topics": {},
            "question_length_stats": {},
            "answer_length_stats": {}
        }

    def download_dataset(self) -> List[Dict]:
        """Download the counsel-chat dataset from HuggingFace."""
        logger.info("Downloading counsel-chat dataset from HuggingFace...")

        try:
            dataset = load_dataset("nbertagnolli/counsel-chat", split="train")
            records = [dict(record) for record in dataset]
            self.stats["original_count"] = len(records)
            logger.info(f"Downloaded {len(records)} records")
            return records
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def normalize_text(self, text: Optional[str]) -> str:
        """Normalize text by stripping and cleaning whitespace."""
        if text is None:
            return ""
        # Strip leading/trailing whitespace
        text = text.strip()
        # Normalize multiple spaces to single space
        text = " ".join(text.split())
        return text

    def is_valid_record(self, record: Dict) -> bool:
        """Check if a record is valid (non-empty question and answer)."""
        question = self.normalize_text(record.get("questionText", ""))
        answer = self.normalize_text(record.get("answerText", ""))
        return bool(question) and bool(answer)

    def passes_length_filter(self, record: Dict) -> bool:
        """Check if record passes minimum length requirements."""
        question = self.normalize_text(record.get("questionText", ""))
        answer = self.normalize_text(record.get("answerText", ""))
        return (
            len(question) >= self.min_question_len and
            len(answer) >= self.min_answer_len
        )

    def get_question_hash(self, question: str) -> str:
        """Generate hash for deduplication based on question."""
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def clean_records(self, records: List[Dict]) -> List[Dict]:
        """
        Clean records by removing empty, short, and duplicate entries.

        Args:
            records: Raw records from dataset

        Returns:
            Cleaned and deduplicated records
        """
        logger.info("Cleaning records...")

        # Step 1: Remove empty records
        valid_records = [r for r in records if self.is_valid_record(r)]
        self.stats["after_empty_removal"] = len(valid_records)
        logger.info(f"After empty removal: {len(valid_records)} records")

        # Step 2: Apply length filter
        filtered_records = [r for r in valid_records if self.passes_length_filter(r)]
        self.stats["after_length_filter"] = len(filtered_records)
        logger.info(f"After length filter: {len(filtered_records)} records")

        # Step 3: Deduplicate based on question text
        seen_hashes = set()
        unique_records = []
        for record in filtered_records:
            question = self.normalize_text(record.get("questionText", ""))
            q_hash = self.get_question_hash(question)
            if q_hash not in seen_hashes:
                seen_hashes.add(q_hash)
                unique_records.append(record)

        self.stats["after_dedup"] = len(unique_records)
        logger.info(f"After deduplication: {len(unique_records)} records")

        return unique_records

    def transform_record(self, record: Dict, idx: int) -> Dict:
        """
        Transform a raw record to the output format.

        Args:
            record: Raw record from dataset
            idx: Record index for ID generation

        Returns:
            Transformed record with standardized fields
        """
        question = self.normalize_text(record.get("questionText", ""))
        answer = self.normalize_text(record.get("answerText", ""))
        topic = self.normalize_text(record.get("topic", "general"))

        return {
            "id": f"counsel_{idx:05d}",
            "question": question,
            "answer": answer,
            "topic": topic if topic else "general"
        }

    def split_data(
        self,
        records: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/eval/test sets.

        Args:
            records: Cleaned records to split

        Returns:
            Tuple of (train, eval, test) record lists
        """
        import random
        random.seed(self.seed)

        # Shuffle records
        shuffled = records.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        eval_end = train_end + int(n * self.eval_ratio)

        train_data = shuffled[:train_end]
        eval_data = shuffled[train_end:eval_end]
        test_data = shuffled[eval_end:]

        self.stats["train_count"] = len(train_data)
        self.stats["eval_count"] = len(eval_data)
        self.stats["test_count"] = len(test_data)

        logger.info(
            f"Split: train={len(train_data)}, "
            f"eval={len(eval_data)}, test={len(test_data)}"
        )

        return train_data, eval_data, test_data

    def compute_statistics(self, records: List[Dict]) -> None:
        """Compute and store statistics about the dataset."""
        # Topic distribution
        for record in records:
            topic = record.get("topic", "general")
            self.stats["topics"][topic] = self.stats["topics"].get(topic, 0) + 1

        # Length statistics
        question_lengths = [len(r["question"]) for r in records]
        answer_lengths = [len(r["answer"]) for r in records]

        self.stats["question_length_stats"] = {
            "min": min(question_lengths) if question_lengths else 0,
            "max": max(question_lengths) if question_lengths else 0,
            "avg": sum(question_lengths) / len(question_lengths) if question_lengths else 0
        }

        self.stats["answer_length_stats"] = {
            "min": min(answer_lengths) if answer_lengths else 0,
            "max": max(answer_lengths) if answer_lengths else 0,
            "avg": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        }

    def save_jsonl(self, records: List[Dict], filename: str) -> None:
        """Save records to JSONL file."""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(records)} records to {filepath}")

    def generate_report(self, report_dir: str = "reports") -> None:
        """Generate data statistics report in markdown format."""
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        report_file = report_path / "data_statistics.md"

        report_content = f"""# Data Statistics Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview

| Metric | Count |
|--------|-------|
| Original Records | {self.stats["original_count"]:,} |
| After Empty Removal | {self.stats["after_empty_removal"]:,} |
| After Length Filter | {self.stats["after_length_filter"]:,} |
| After Deduplication | {self.stats["after_dedup"]:,} |

## Data Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | {self.stats["train_count"]:,} | {self.train_ratio * 100:.0f}% |
| Eval | {self.stats["eval_count"]:,} | {self.eval_ratio * 100:.0f}% |
| Test | {self.stats["test_count"]:,} | {self.test_ratio * 100:.0f}% |
| **Total** | {self.stats["after_dedup"]:,} | 100% |

## Question Length Statistics

| Metric | Value |
|--------|-------|
| Minimum | {self.stats["question_length_stats"].get("min", 0):,} chars |
| Maximum | {self.stats["question_length_stats"].get("max", 0):,} chars |
| Average | {self.stats["question_length_stats"].get("avg", 0):,.1f} chars |

## Answer Length Statistics

| Metric | Value |
|--------|-------|
| Minimum | {self.stats["answer_length_stats"].get("min", 0):,} chars |
| Maximum | {self.stats["answer_length_stats"].get("max", 0):,} chars |
| Average | {self.stats["answer_length_stats"].get("avg", 0):,.1f} chars |

## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
"""
        # Sort topics by count
        sorted_topics = sorted(
            self.stats["topics"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        total = self.stats["after_dedup"]
        for topic, count in sorted_topics:
            percentage = (count / total * 100) if total > 0 else 0
            report_content += f"| {topic} | {count:,} | {percentage:.1f}% |\n"

        report_content += """
## Cleaning Rules Applied

1. **Empty Removal**: Removed records with empty question or answer
2. **Length Filter**:
   - Minimum question length: {min_q} characters
   - Minimum answer length: {min_a} characters
3. **Deduplication**: Based on question text (case-insensitive)

## Output Files

- `data/processed/counsel_chat_train.jsonl` - Training data
- `data/processed/counsel_chat_eval.jsonl` - Evaluation data
- `data/processed/counsel_chat_test.jsonl` - Test data
- `data/processed/counsel_chat_cleaned.jsonl` - All cleaned data (combined)

## Record Format

```json
{{
    "id": "counsel_00001",
    "question": "User's question text",
    "answer": "Counselor's response text",
    "topic": "Topic category"
}}
```
""".format(
            min_q=self.min_question_len,
            min_a=self.min_answer_len
        )

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Generated report: {report_file}")

    def run(self) -> Dict:
        """
        Execute the full data preparation pipeline.

        Returns:
            Statistics dictionary
        """
        logger.info("Starting data preparation pipeline...")

        # Step 1: Download dataset
        raw_records = self.download_dataset()

        # Step 2: Clean records
        cleaned_records = self.clean_records(raw_records)

        # Step 3: Transform to output format
        transformed_records = [
            self.transform_record(r, i)
            for i, r in enumerate(cleaned_records)
        ]

        # Step 4: Compute statistics
        self.compute_statistics(transformed_records)

        # Step 5: Split data
        train_data, eval_data, test_data = self.split_data(transformed_records)

        # Step 6: Save files
        self.save_jsonl(transformed_records, "counsel_chat_cleaned.jsonl")
        self.save_jsonl(train_data, "counsel_chat_train.jsonl")
        self.save_jsonl(eval_data, "counsel_chat_eval.jsonl")
        self.save_jsonl(test_data, "counsel_chat_test.jsonl")

        # Step 7: Generate report
        self.generate_report()

        logger.info("Data preparation completed successfully!")
        logger.info(f"Total cleaned records: {len(transformed_records)}")

        return self.stats


def main():
    """Main entry point for data preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare Counsel Chat dataset for DPO training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--min-question-len",
        type=int,
        default=10,
        help="Minimum question length in characters"
    )
    parser.add_argument(
        "--min-answer-len",
        type=int,
        default=50,
        help="Minimum answer length in characters"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Evaluation data ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test data ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.eval_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio}"
        )

    # Run preparation
    prep = DataPreparation(
        output_dir=args.output_dir,
        min_question_len=args.min_question_len,
        min_answer_len=args.min_answer_len,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    stats = prep.run()

    # Print summary
    print("\n" + "=" * 50)
    print("DATA PREPARATION SUMMARY")
    print("=" * 50)
    print(f"Original records:     {stats['original_count']:,}")
    print(f"After cleaning:       {stats['after_dedup']:,}")
    print(f"Train set:            {stats['train_count']:,}")
    print(f"Eval set:             {stats['eval_count']:,}")
    print(f"Test set:             {stats['test_count']:,}")
    print("=" * 50)


if __name__ == "__main__":
    main()
