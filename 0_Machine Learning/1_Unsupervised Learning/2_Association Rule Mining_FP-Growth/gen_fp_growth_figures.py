#!/usr/bin/env python3
"""Generate figures for the FP-Growth association rule tutorial."""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

Transaction = Sequence[str]
Itemset = Tuple[str, ...]

FIGURE_LIST = [
    "fpgrowth_support_confidence.png",
    "fpgrowth_lift_distribution.png",
]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_transactions(n_transactions: int = 600, seed: int = 19) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    catalog = np.array([
        "bread",
        "milk",
        "eggs",
        "butter",
        "cheese",
        "apples",
        "bananas",
        "coffee",
        "tea",
        "cereal",
    ])

    transactions: List[List[str]] = []
    for _ in range(n_transactions):
        basket_size = rng.integers(2, 6)
        basket: set[str] = set()

        if rng.random() < 0.32:
            basket.update({"bread", "milk"})
        if rng.random() < 0.24:
            basket.update({"coffee", "cereal"})
        if rng.random() < 0.18:
            basket.update({"apples", "bananas"})

        while len(basket) < basket_size:
            basket.add(rng.choice(catalog))

        transactions.append(sorted(basket))
    return transactions


# ---------------------------------------------------------------------------
# FP-tree implementation
# ---------------------------------------------------------------------------

@dataclass
class FPNode:
    item: Optional[str]
    count: int
    parent: Optional["FPNode"]
    children: Dict[str, "FPNode"] = field(default_factory=dict)
    link: Optional["FPNode"] = None

    def add_child(self, item: str) -> "FPNode":
        child = FPNode(item=item, count=0, parent=self)
        self.children[item] = child
        return child


def build_fp_tree(
    transactions: Iterable[List[str]],
    min_support: int,
) -> Tuple[FPNode, Dict[str, Tuple[int, Optional[FPNode]]]]:
    item_counts: Dict[str, int] = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    item_counts = {item: count for item, count in item_counts.items() if count >= min_support}
    if not item_counts:
        return FPNode(item=None, count=0, parent=None), {}

    ordered_items = sorted(item_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    order_index = {item: idx for idx, (item, _) in enumerate(ordered_items)}

    header_table: Dict[str, Tuple[int, Optional[FPNode]]] = {
        item: (support, None) for item, support in item_counts.items()
    }

    root = FPNode(item=None, count=0, parent=None)

    for transaction in transactions:
        filtered = [item for item in transaction if item in item_counts]
        filtered.sort(key=lambda item: order_index[item])
        current = root
        for item in filtered:
            if item not in current.children:
                current = current.add_child(item)
                current.count = 0
                # Link into header table
                _, node_link = header_table[item]
                if node_link is None:
                    header_table[item] = (header_table[item][0], current)
                else:
                    while node_link.link is not None:
                        node_link = node_link.link
                    node_link.link = current
            else:
                current = current.children[item]
            current.count += 1

    return root, header_table


def extract_conditional_pattern_base(
    item: str,
    header_table: Dict[str, Tuple[int, Optional[FPNode]]],
) -> List[Tuple[List[str], int]]:
    patterns: List[Tuple[List[str], int]] = []
    _, node = header_table[item]
    while node is not None:
        path: List[str] = []
        parent = node.parent
        while parent is not None and parent.item is not None:
            path.append(parent.item)
            parent = parent.parent
        if path:
            patterns.append((list(reversed(path)), node.count))
        node = node.link
    return patterns


def fpgrowth_recursive(
    tree: FPNode,
    header_table: Dict[str, Tuple[int, Optional[FPNode]]],
    min_support: int,
    prefix: Tuple[str, ...],
    frequent_itemsets: Dict[Tuple[str, ...], int],
):
    items = sorted(header_table.items(), key=lambda kv: kv[1][0])  # ascend support
    for item, (support, _) in items:
        new_pattern = tuple(sorted(prefix + (item,)))
        frequent_itemsets[new_pattern] = support

        conditional_base = extract_conditional_pattern_base(item, header_table)
        conditional_transactions: List[List[str]] = []
        for path, count in conditional_base:
            for _ in range(count):
                conditional_transactions.append(path)

        if not conditional_transactions:
            continue

        conditional_tree, conditional_header = build_fp_tree(conditional_transactions, min_support)
        if conditional_header:
            fpgrowth_recursive(
                conditional_tree,
                conditional_header,
                min_support,
                new_pattern,
                frequent_itemsets,
            )


def fpgrowth(transactions: List[List[str]], min_support: float) -> Dict[Tuple[str, ...], float]:
    absolute_support = max(1, int(min_support * len(transactions)))
    tree, header = build_fp_tree(transactions, absolute_support)
    frequent_counts: Dict[Tuple[str, ...], int] = {}
    if header:
        fpgrowth_recursive(tree, header, absolute_support, tuple(), frequent_counts)
    return {itemset: count / len(transactions) for itemset, count in frequent_counts.items()}


# ---------------------------------------------------------------------------
# Rule derivation and plotting
# ---------------------------------------------------------------------------

def derive_rules(
    frequent_itemsets: Dict[Itemset, float],
    min_confidence: float,
) -> List[Tuple[Itemset, Itemset, float, float, float]]:
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
    support_vals = [rule[2] for rule in rules]
    confidence_vals = [rule[3] for rule in rules]
    lift_vals = [rule[4] for rule in rules]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sizes = [90 * (val if val > 1 else 0.8) for val in lift_vals]
    scatter = ax.scatter(
        support_vals,
        confidence_vals,
        s=sizes,
        c=lift_vals,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.3,
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Support vs. Confidence (lift encoded by size/color)")
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
    lift_vals = [rule[4] for rule in rules]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(lift_vals, bins=15, color="#1f77b4", alpha=0.8, edgecolor="k")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Number of rules")
    ax.set_title("Distribution of Rule Lift")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.1, label="Lift = 1")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    transactions = synthesize_transactions()
    frequent_itemsets = fpgrowth(transactions, min_support=0.06)
    rules = derive_rules(frequent_itemsets, min_confidence=0.5)

    if not rules:
        raise SystemExit("No rules generated; adjust support/confidence thresholds.")

    support_conf_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_support_confidence(rules, support_conf_path)

    lift_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_lift_distribution(rules, lift_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()