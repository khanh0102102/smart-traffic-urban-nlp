"""Vietnamese stopwords for incident reports.

Negations are deliberately excluded because removing them can reverse sentence meaning.
"""

STOPWORDS = {
    "và", "là", "của", "những", "các", "một", "cho", "với", "tại",
    "ở", "trên", "dưới", "từ", "đến", "này", "đó", "kia", "thì",
    "mà", "theo", "về", "đã", "đang", "sẽ", "rất", "quá",
}

NEGATION_WORDS = {"không", "chưa", "chẳng", "chả", "đừng"}


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOPWORDS or token in NEGATION_WORDS]
