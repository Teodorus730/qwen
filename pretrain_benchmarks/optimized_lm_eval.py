"""lm-eval CLI wrapper that enables the opt-in MMMLU logits adapter."""
from optimized_mmmlu_adapter import enable
from lm_eval.__main__ import cli_evaluate

enable()
cli_evaluate()
