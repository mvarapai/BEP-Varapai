from __future__ import annotations


BINARY_SENTIMENT_TEMPLATE = """
Classify this software engineering text as negative or non-negative.

Definitions:
- negative: frustration, criticism, complaint, anger, rejection, or other clearly negative sentiment
- non-negative: neutral, factual, technical, appreciative, positive, or otherwise not negative

Important:
- Software engineering terms such as bug, crash, kill, fatal, error, or deadlock are not negative by themselves.
- Focus on the author's attitude, not only the technical topic.
- Return exactly one label.
- Do not explain your answer.

Text:
$text
""".strip()