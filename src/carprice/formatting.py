"""Lightweight number formatting shared by plots, reports, CLI and GUI (A7 formats).

Kept free of plotting imports so the predictor loads quickly (D-031).
"""


def format_inr_indian(value: float) -> str:
    """Format rupees with Indian digit grouping, e.g. 550000 -> '₹ 5,50,000' (A7).

    Args:
        value: Amount in INR (rounded to whole rupees; negatives keep their sign).

    Returns:
        str: Grouped string with the rupee sign.
    """
    sign = "-" if value < 0 else ""
    digits = str(round(abs(value)))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = [head[max(i - 2, 0) : i] for i in range(len(head), 0, -2)][::-1]
        digits = ",".join([*groups, tail])
    return f"{sign}₹ {digits}"


def format_lakh(value: float) -> str:
    """Format rupees in lakh with two decimals, e.g. 550000 -> '₹ 5.50 lakh' (A7)."""
    return f"₹ {value / 100000:.2f} lakh"
