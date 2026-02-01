#!/usr/bin/env python3
"""
Deepseek Judge Script for DPO Dataset Construction

Uses Deepseek-V3 to evaluate and compare professional responses (chosen)
vs baseline responses (rejected) to build the DPO training dataset.

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    python scripts/deepseek_judge.py [--input data/baseline/responses.jsonl] [--output-dir data/dpo]
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Judge prompt template
JUDGE_PROMPT_TEMPLATE = """As a mental health counseling expert, evaluate the following two responses to a counseling question.

## Question
{prompt}

## Response A (Professional Counselor Response)
{chosen}

## Response B (Baseline AI Response)
{rejected}

## Evaluation Criteria
Compare the responses based on:
1. **Empathy**: Does the response show genuine understanding and validation of feelings?
2. **Professionalism**: Does it follow mental health counseling best practices?
3. **Safety**: Does it appropriately handle sensitive topics and provide resources if needed?
4. **Helpfulness**: Does it provide actionable guidance or coping strategies?
5. **Appropriate Boundaries**: Does it acknowledge limitations and suggest professional help when appropriate?

## Your Task
Determine if Response A (Professional) is better than Response B (Baseline).

Output your evaluation in the following JSON format ONLY:
```json
{{
    "is_chosen_better": true,
    "reason": "Brief explanation of why Response A is better or not"
}}
```

If Response A is NOT clearly better (they are equal quality or Response B is better), set "is_chosen_better" to false.

IMPORTANT: Output ONLY the JSON, no other text."""


class DeepseekJudge:
    """Uses Deepseek-V3 API to judge response quality for DPO dataset."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        mock_mode: bool = False
    ):
        """
        Initialize judge.

        Args:
            api_key: Deepseek API key
            base_url: API base URL
            model: Model to use for judging
            mock_mode: If True, use mock responses for testing
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com"
        )
        self.model = model
        self.mock_mode = mock_mode or (os.getenv("LLM_TYPE", "").upper() == "MOCK")

        if not self.mock_mode and not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable required "
                "(or use --mock for testing)"
            )

        self._client = None

    async def _ensure_client(self):
        """Initialize HTTP client."""
        if self._client is not None:
            return

        if self.mock_mode:
            return

        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _create_judge_prompt(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> str:
        """Create the judge evaluation prompt."""
        return JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected
        )

    def _parse_judge_response(self, response_text: str) -> Tuple[bool, str]:
        """
        Parse judge response to extract decision.

        Args:
            response_text: Raw response from judge

        Returns:
            Tuple of (is_chosen_better, reason)
        """
        # Try to extract JSON from response
        json_pattern = r'\{[^{}]*"is_chosen_better"[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        if matches:
            try:
                # Try the last match (in case there are multiple)
                for match in reversed(matches):
                    data = json.loads(match)
                    if "is_chosen_better" in data:
                        return (
                            bool(data["is_chosen_better"]),
                            data.get("reason", "No reason provided")
                        )
            except json.JSONDecodeError:
                pass

        # Fallback: look for keywords
        response_lower = response_text.lower()
        if '"is_chosen_better": true' in response_lower or '"is_chosen_better":true' in response_lower:
            return True, "Parsed from response text"
        elif '"is_chosen_better": false' in response_lower or '"is_chosen_better":false' in response_lower:
            return False, "Parsed from response text"

        # Default to True if we can't parse (conservative approach)
        logger.warning(f"Could not parse judge response, defaulting to True")
        return True, "Default (parsing failed)"

    async def judge(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> Tuple[bool, str]:
        """
        Judge whether chosen response is better than rejected.

        Args:
            prompt: The original question
            chosen: Professional counselor response
            rejected: Baseline AI response

        Returns:
            Tuple of (is_chosen_better, reason)
        """
        if self.mock_mode:
            return self._mock_judge(prompt, chosen, rejected)

        await self._ensure_client()

        judge_prompt = self._create_judge_prompt(prompt, chosen, rejected)

        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": judge_prompt}
                    ],
                    "temperature": 0.1,  # Low temperature for consistent judgments
                    "max_tokens": 500
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("choices"):
                    response_text = data["choices"][0]["message"]["content"]
                    return self._parse_judge_response(response_text)

            logger.error(f"API error: {response.status_code} - {response.text}")
            return True, f"API error (defaulting to True): {response.status_code}"

        except Exception as e:
            logger.error(f"Judge request failed: {e}")
            return True, f"Error (defaulting to True): {str(e)}"

    def _mock_judge(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> Tuple[bool, str]:
        """Mock judge for testing."""
        # Simple heuristic: professional responses are usually longer
        # and contain more empathetic language
        chosen_len = len(chosen)
        rejected_len = len(rejected)

        empathy_words = ["understand", "feel", "valid", "challenging", "support"]
        chosen_empathy = sum(1 for w in empathy_words if w in chosen.lower())
        rejected_empathy = sum(1 for w in empathy_words if w in rejected.lower())

        is_better = (
            chosen_len > rejected_len * 0.8 or
            chosen_empathy > rejected_empathy
        )

        reason = (
            f"[MOCK] Length: {chosen_len} vs {rejected_len}, "
            f"Empathy words: {chosen_empathy} vs {rejected_empathy}"
        )

        return is_better, reason


class DPODatasetBuilder:
    """Builds DPO training dataset using Deepseek judge."""

    def __init__(
        self,
        judge: DeepseekJudge,
        input_file: str,
        output_dir: str,
        train_ratio: float = 0.9,
        seed: int = 42
    ):
        """
        Initialize dataset builder.

        Args:
            judge: DeepseekJudge instance
            input_file: Path to baseline responses file
            output_dir: Output directory for DPO data
            train_ratio: Ratio of data for training (rest is eval)
            seed: Random seed
        """
        self.judge = judge
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.seed = seed

        # Statistics
        self.stats = {
            "total_records": 0,
            "chosen_better_count": 0,
            "rejected_better_count": 0,
            "error_count": 0,
            "train_count": 0,
            "eval_count": 0
        }

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_baseline_responses(self) -> List[Dict]:
        """Load baseline responses from file."""
        records = []
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(records)} baseline responses")
        return records

    async def process_record(
        self,
        record: Dict,
        idx: int
    ) -> Optional[Dict]:
        """
        Process a single record through the judge.

        Args:
            record: Record with question, original_answer, baseline_response
            idx: Record index

        Returns:
            DPO record if chosen is better, None otherwise
        """
        prompt = record["question"]
        chosen = record["original_answer"]
        rejected = record["baseline_response"]

        try:
            is_chosen_better, reason = await self.judge.judge(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected
            )

            if is_chosen_better:
                self.stats["chosen_better_count"] += 1
                return {
                    "id": record["id"],
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "topic": record.get("topic", "general"),
                    "judge_reason": reason
                }
            else:
                self.stats["rejected_better_count"] += 1
                logger.debug(f"Record {record['id']} rejected: {reason}")
                return None

        except Exception as e:
            self.stats["error_count"] += 1
            logger.error(f"Error processing record {record['id']}: {e}")
            return None

    async def build_dataset(
        self,
        batch_size: int = 5,
        delay: float = 0.5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Build DPO dataset by judging all records.

        Args:
            batch_size: Number of concurrent requests
            delay: Delay between batches (rate limiting)

        Returns:
            Tuple of (train_data, eval_data)
        """
        records = self.load_baseline_responses()
        self.stats["total_records"] = len(records)

        dpo_records = []

        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            # Process batch concurrently
            tasks = [
                self.process_record(record, i + j)
                for j, record in enumerate(batch)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect valid results
            for result in results:
                if isinstance(result, dict):
                    dpo_records.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    self.stats["error_count"] += 1

            # Progress logging
            processed = min(i + batch_size, len(records))
            logger.info(
                f"Progress: {processed}/{len(records)} "
                f"(accepted: {len(dpo_records)})"
            )

            # Rate limiting delay
            if i + batch_size < len(records):
                await asyncio.sleep(delay)

        # Split into train/eval
        random.seed(self.seed)
        random.shuffle(dpo_records)

        split_idx = int(len(dpo_records) * self.train_ratio)
        train_data = dpo_records[:split_idx]
        eval_data = dpo_records[split_idx:]

        self.stats["train_count"] = len(train_data)
        self.stats["eval_count"] = len(eval_data)

        return train_data, eval_data

    def save_dataset(
        self,
        train_data: List[Dict],
        eval_data: List[Dict]
    ):
        """Save DPO dataset to files."""
        # Save training data
        train_file = self.output_dir / "train.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for record in train_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(train_data)} training records to {train_file}")

        # Save evaluation data
        eval_file = self.output_dir / "eval.jsonl"
        with open(eval_file, "w", encoding="utf-8") as f:
            for record in eval_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(eval_data)} evaluation records to {eval_file}")

    def generate_report(self, report_dir: str = "reports"):
        """Generate judge statistics report."""
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        report_file = report_path / "judge_statistics.md"

        pass_rate = (
            self.stats["chosen_better_count"] / self.stats["total_records"] * 100
            if self.stats["total_records"] > 0 else 0
        )

        report_content = f"""# Judge Statistics Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

| Metric | Count |
|--------|-------|
| Total Records Processed | {self.stats["total_records"]:,} |
| Chosen Better (Accepted) | {self.stats["chosen_better_count"]:,} |
| Rejected Better (Filtered) | {self.stats["rejected_better_count"]:,} |
| Errors | {self.stats["error_count"]:,} |

## Pass Rate

**{pass_rate:.1f}%** of records passed the judge evaluation.

## DPO Dataset Split

| Split | Count | Percentage |
|-------|-------|------------|
| Train | {self.stats["train_count"]:,} | {self.train_ratio * 100:.0f}% |
| Eval | {self.stats["eval_count"]:,} | {(1 - self.train_ratio) * 100:.0f}% |
| **Total** | {self.stats["chosen_better_count"]:,} | 100% |

## Output Files

- `data/dpo/train.jsonl` - Training data
- `data/dpo/eval.jsonl` - Evaluation data

## Record Format

```json
{{
    "id": "counsel_00001",
    "prompt": "User's question",
    "chosen": "Professional counselor response",
    "rejected": "Baseline AI response",
    "topic": "Topic category",
    "judge_reason": "Why chosen is better"
}}
```

## Evaluation Criteria

The Deepseek-V3 judge evaluated responses based on:
1. Empathy and validation
2. Professional counseling practices
3. Safety and appropriate handling of sensitive topics
4. Actionable guidance
5. Appropriate professional boundaries
"""

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Generated report: {report_file}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build DPO dataset using Deepseek judge"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/baseline/responses.jsonl",
        help="Input file with baseline responses"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dpo",
        help="Output directory for DPO data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Training data ratio"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Concurrent API requests"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batches (seconds)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock judge for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Initialize judge
    judge = DeepseekJudge(mock_mode=args.mock)

    # Initialize builder
    builder = DPODatasetBuilder(
        judge=judge,
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    try:
        # Build dataset
        logger.info("Starting DPO dataset construction...")
        train_data, eval_data = await builder.build_dataset(
            batch_size=args.batch_size,
            delay=args.delay
        )

        # Save dataset
        builder.save_dataset(train_data, eval_data)

        # Generate report
        builder.generate_report()

        # Print summary
        print("\n" + "=" * 50)
        print("DPO DATASET CONSTRUCTION SUMMARY")
        print("=" * 50)
        print(f"Total processed:    {builder.stats['total_records']:,}")
        print(f"Accepted:           {builder.stats['chosen_better_count']:,}")
        print(f"Filtered out:       {builder.stats['rejected_better_count']:,}")
        print(f"Errors:             {builder.stats['error_count']:,}")
        print(f"Train set:          {builder.stats['train_count']:,}")
        print(f"Eval set:           {builder.stats['eval_count']:,}")
        pass_rate = (
            builder.stats['chosen_better_count'] /
            builder.stats['total_records'] * 100
            if builder.stats['total_records'] > 0 else 0
        )
        print(f"Pass rate:          {pass_rate:.1f}%")
        print("=" * 50)

    finally:
        await judge.close()


if __name__ == "__main__":
    asyncio.run(main())
