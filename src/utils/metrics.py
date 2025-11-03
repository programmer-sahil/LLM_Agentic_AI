"""
===============================================================
File: metrics.py
Author: SK Sahil
Project: Bachelor's Thesis – From Transformers to Agents
Module: Utility - Metrics Calculation
===============================================================
Description:
Provides helper functions for evaluating model efficiency based on
latency, token usage, and compression ratios.
===============================================================
"""

import numpy as np


def calculate_throughput(tokens_generated, latency_seconds):
    """Compute tokens generated per second."""
    if latency_seconds == 0:
        return 0.0
    return round(tokens_generated / latency_seconds, 3)


def compute_compression_ratio(raw_length, tokenized_length):
    """Compute text compression ratio."""
    if tokenized_length == 0:
        return 0.0
    return round(raw_length / tokenized_length, 3)


def summarize_metrics(model_name, tokens, latency, compression):
    """Returns formatted metric summary for quick logging."""
    return {
        "model": model_name,
        "generated_tokens": tokens,
        "latency_sec": latency,
        "throughput": calculate_throughput(tokens, latency),
        "compression_ratio": compression,
    }


if __name__ == "__main__":
    print("️ Metric Test Example:")
    print(summarize_metrics("Phi-2", 180, 9.84, 1.56))
