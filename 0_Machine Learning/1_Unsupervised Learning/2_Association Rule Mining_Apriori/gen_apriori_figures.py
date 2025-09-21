#!/usr/bin/env python3
"""Generate figures for the Apriori association rule tutorial."""

import os
import random
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

Transaction = Sequence[str]
Itemset = Tuple[str, ...]

# Figures referenced by the LaTeX documents.
FIGURE_LIST = [
    "apriori_support_confidence.png",
    "apriori_lift_distribution.png",
]


def make_output_dir() -> str:
    """Return the figures directory path, creating it if necessary."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_transactions(
    n_transactions: int = 600,
    seed: int = 21,
) -> List[Transaction]:
    """Create synthetic market-basket data with embedded patterns."""

    rng = random.Random(seed)
    catalog = ["bread", "milk", "eggs", "butter", "cheese", "apples", "bananas", "coffee", "tea", "cereal"]
    transactions: List[List[str]] = []

    for _ in range(n_transactions):
        basket = set()

        # Base purchase size between 2 and 6 items.
        basket_size = rng.randint(2, 6)

        # Seed baskets with frequent combinations.
        if rng.random() < 0.35:
            basket.update(["bread", "milk"])
        if rng.random() < 0.25:
            basket.update(["coffee", "cereal"])
        if rng.random() < 0.15:
            basket.update(["apples", "bananas"])

        while len(basket) < basket_size:
            basket.add(rng.choice(catalog))

        transactions.append(sorted(basket))

    return transactions


def calculate_support(
    transactions: Iterable[Transaction],
    candidates: Iterable[Itemset],
) -> Dict[Itemset, float]:
    """Compute support for candidate itemsets."""

    transaction_list = list(transactions)
    total = len(transaction_list)
    counts: Dict[Itemset, int] = defaultdict(int)

    for transaction in transaction_list:
        transaction_set = set(transaction)
        for candidate in candidates:
            if set(candidate).issubset(transaction_set):
                counts[candidate] += 1

    return {itemset: count / total for itemset, count in counts.items()}


def generate_candidates(prev_frequents: List[Itemset], k: int) -> List[Itemset]:
    """Join frequent (k-1)-itemsets to generate k-itemset candidates."""

    candidates = set()
    for i in range(len(prev_frequents)):
        for j in range(i + 1, len(prev_frequents)):
            left, right = prev_frequents[i], prev_frequents[j]
            union = tuple(sorted(set(left) | set(right)))
            if len(union) == k:
                # Prune if any (k-1)-subset is not frequent.
                subsets = combinations(union, k - 1)
                if all(tuple(sorted(subset)) in prev_frequents for subset in subsets):
                    candidates.add(union)
    return sorted(candidates)


def apriori(
    transactions: List[Transaction],
    min_support: float = 0.05,
) -> Dict[Itemset, float]:
    """Return frequent itemsets and their support."""

    transaction_list = list(transactions)
    # Generate 1-item candidates.
    items = sorted({item for txn in transaction_list for item in txn})
    candidates = [(item,) for item in items]
    frequent_itemsets: Dict[Itemset, float] = {}

    k = 1
    while candidates:
        support_map = calculate_support(transaction_list, candidates)
        frequents_k = [itemset for itemset, supp in support_map.items() if supp >= min_support]
        if not frequents_k:
            break
        for itemset in frequents_k:
            frequent_itemsets[itemset] = support_map[itemset]
        k += 1
        candidates = generate_candidates(frequents_k, k)

    return frequent_itemsets


def derive_rules(
    frequent_itemsets: Dict[Itemset, float],
    min_confidence: float = 0.4,
) -> List[Tuple[Itemset, Itemset, float, float, float]]:
    """Generate association rules from frequent itemsets."""

    rules: List[Tuple[Itemset, Itemset, float, float, float]] = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for split in range(1, len(itemset)):
            for lhs in combinations(itemset, split):
                lhs = tuple(sorted(lhs))
                rhs = tuple(sorted(set(itemset) - set(lhs)))
                if lhs not in frequent_itemsets or rhs not in frequent_itemsets:
                    continue
                confidence = support / frequent_itemsets[lhs]
                if confidence < min_confidence:
                    continue
                lift = confidence / frequent_itemsets[rhs]
                rules.append((lhs, rhs, support, confidence, lift))
    return rules


def plot_support_confidence(
    rules: List[Tuple[Itemset, Itemset, float, float, float]],
    output_path: str,
) -> None:
    """Scatter plot showing support vs. confidence with lift as marker size."""

    support = [rule[2] for rule in rules]
    confidence = [rule[3] for rule in rules]
    lift = [rule[4] for rule in rules]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sizes = [80 * (value if value > 1 else 0.8) for value in lift]
    scatter = ax.scatter(support, confidence, s=sizes, c=lift, cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Support vs. Confidence (marker size/colors reflect lift)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Lift")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_lift_distribution(
    rules: List[Tuple[Itemset, Itemset, float, float, float]],
    output_path: str,
) -> None:
    """Histogram of lift values highlighting strong associations."""

    lift = [rule[4] for rule in rules]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(lift, bins=15, color="#1f77b4", alpha=0.8, edgecolor="k")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Number of rules")
    ax.set_title("Distribution of Rule Lift")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="Lift = 1")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Generate Apriori tutorial figures and print their absolute paths."""

    figures_dir = make_output_dir()
    transactions = synthesize_transactions()
    frequent_itemsets = apriori(transactions, min_support=0.06)
    rules = derive_rules(frequent_itemsets, min_confidence=0.45)

    if not rules:
        raise SystemExit("No rules were generated with the chosen thresholds.")

    scatter_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_support_confidence(rules, scatter_path)

    lift_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_lift_distribution(rules, lift_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()