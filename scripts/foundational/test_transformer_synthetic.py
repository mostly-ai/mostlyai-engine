#!/usr/bin/env python
"""
Test the Foundational Tabular Transformer with synthetic data.

This script validates the model works end-to-end without requiring T4 access.
"""

import logging
import random

import pandas as pd
import torch

from mostlyai.engine._tabular.transformer import create_model
from mostlyai.engine._tabular.transformer_training import (
    classify,
    evaluate_classification,
    pretrain_mcm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
_LOG = logging.getLogger(__name__)


def generate_synthetic_tables(num_tables: int = 20, rows_per_table: int = 100) -> list[pd.DataFrame]:
    """Generate synthetic categorical tables for testing."""
    tables = []

    # Different "schemas" to simulate diverse tables
    schemas = [
        {
            "columns": ["color", "size", "shape", "material"],
            "values": {
                "color": ["red", "blue", "green", "yellow", "black", "white"],
                "size": ["small", "medium", "large", "xlarge"],
                "shape": ["round", "square", "triangle", "oval"],
                "material": ["wood", "metal", "plastic", "glass"],
            },
        },
        {
            "columns": ["country", "city", "weather", "season"],
            "values": {
                "country": ["USA", "UK", "Germany", "France", "Japan", "Brazil"],
                "city": ["New York", "London", "Berlin", "Paris", "Tokyo", "Rio"],
                "weather": ["sunny", "rainy", "cloudy", "snowy", "windy"],
                "season": ["spring", "summer", "fall", "winter"],
            },
        },
        {
            "columns": ["animal", "habitat", "diet", "size_class"],
            "values": {
                "animal": ["lion", "eagle", "shark", "elephant", "snake", "wolf"],
                "habitat": ["forest", "ocean", "desert", "jungle", "arctic"],
                "diet": ["carnivore", "herbivore", "omnivore"],
                "size_class": ["tiny", "small", "medium", "large", "huge"],
            },
        },
        {
            "columns": ["food", "cuisine", "taste", "temperature"],
            "values": {
                "food": ["pizza", "sushi", "tacos", "curry", "pasta", "burger"],
                "cuisine": ["italian", "japanese", "mexican", "indian", "american"],
                "taste": ["sweet", "salty", "sour", "spicy", "savory"],
                "temperature": ["hot", "cold", "warm", "room_temp"],
            },
        },
        {
            "columns": ["genre", "mood", "tempo", "instrument"],
            "values": {
                "genre": ["rock", "jazz", "classical", "electronic", "hip-hop"],
                "mood": ["happy", "sad", "energetic", "calm", "angry"],
                "tempo": ["slow", "medium", "fast", "very_fast"],
                "instrument": ["guitar", "piano", "drums", "violin", "synth"],
            },
        },
    ]

    for i in range(num_tables):
        schema = random.choice(schemas)
        data = {}
        for col in schema["columns"]:
            data[col] = [random.choice(schema["values"][col]) for _ in range(rows_per_table)]
        tables.append(pd.DataFrame(data))

    return tables


def generate_classification_dataset(num_samples: int = 200) -> pd.DataFrame:
    """Generate a dataset with a predictable target for classification testing."""
    data = {
        "feature1": [],
        "feature2": [],
        "feature3": [],
        "target": [],
    }

    # Create a simple rule: target depends on feature1 and feature2
    feature1_values = ["A", "B", "C"]
    feature2_values = ["X", "Y", "Z"]
    feature3_values = ["P", "Q", "R", "S"]

    for _ in range(num_samples):
        f1 = random.choice(feature1_values)
        f2 = random.choice(feature2_values)
        f3 = random.choice(feature3_values)

        # Simple rule: if f1 is A and f2 is X, target is "positive"
        # Otherwise, mostly "negative" with some noise
        if f1 == "A" and f2 == "X":
            target = "positive" if random.random() < 0.9 else "negative"
        elif f1 == "B" and f2 == "Y":
            target = "positive" if random.random() < 0.7 else "negative"
        else:
            target = "negative" if random.random() < 0.8 else "positive"

        data["feature1"].append(f1)
        data["feature2"].append(f2)
        data["feature3"].append(f3)
        data["target"].append(target)

    return pd.DataFrame(data)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOG.info(f"Using device: {device}")

    # Step 1: Create model
    _LOG.info("Creating small model...")
    model = create_model("small")
    num_params = sum(p.numel() for p in model.parameters())
    _LOG.info(f"Model has {num_params:,} parameters")

    # Step 2: Generate synthetic pre-training data
    _LOG.info("Generating synthetic tables for pre-training...")
    tables = generate_synthetic_tables(num_tables=50, rows_per_table=100)
    _LOG.info(f"Generated {len(tables)} tables with ~100 rows each")

    # Step 3: Pre-train with MCM
    _LOG.info("Pre-training with Masked Cell Modeling...")
    model = pretrain_mcm(
        model=model,
        tables=tables,
        num_epochs=3,
        batch_size=32,
        learning_rate=1e-3,
        device=device,
        log_every=50,
    )
    _LOG.info("Pre-training complete!")

    # Step 4: Generate classification test data
    _LOG.info("Generating classification dataset...")
    train_df = generate_classification_dataset(num_samples=500)
    test_df = generate_classification_dataset(num_samples=100)

    _LOG.info(f"Train set: {len(train_df)} samples")
    _LOG.info(f"Test set: {len(test_df)} samples")
    _LOG.info(f"Target distribution (train): {train_df['target'].value_counts().to_dict()}")

    # Step 5: Evaluate classification (zero-shot after pre-training)
    _LOG.info("Evaluating zero-shot classification...")
    metrics = evaluate_classification(
        model=model,
        df=test_df,
        target_column="target",
        device=device,
    )
    _LOG.info(f"Zero-shot accuracy: {metrics['accuracy']:.2%}")

    # Step 6: Fine-tune on classification task (optional - just more MCM on task data)
    _LOG.info("Fine-tuning on classification data...")
    model = pretrain_mcm(
        model=model,
        tables=[train_df],
        num_epochs=5,
        batch_size=32,
        learning_rate=5e-4,
        device=device,
        log_every=50,
    )

    # Step 7: Evaluate after fine-tuning
    _LOG.info("Evaluating after fine-tuning...")
    metrics = evaluate_classification(
        model=model,
        df=test_df,
        target_column="target",
        device=device,
    )
    _LOG.info(f"Fine-tuned accuracy: {metrics['accuracy']:.2%}")

    # Step 8: Show sample predictions
    _LOG.info("\nSample predictions:")
    sample = test_df.head(5).copy()
    probs = classify(
        model=model,
        df=sample,
        target_column="target",
        target_classes=["positive", "negative"],
        device=device,
    )

    for i in range(len(sample)):
        true_label = sample.iloc[i]["target"]
        prob_pos = probs.iloc[i]["prob_positive"]
        prob_neg = probs.iloc[i]["prob_negative"]
        predicted = "positive" if prob_pos > prob_neg else "negative"
        _LOG.info(
            f"  Row {i}: f1={sample.iloc[i]['feature1']}, f2={sample.iloc[i]['feature2']} "
            f"-> pred={predicted} (p={max(prob_pos, prob_neg):.2f}), true={true_label}"
        )

    _LOG.info("\nTest complete!")


if __name__ == "__main__":
    main()
