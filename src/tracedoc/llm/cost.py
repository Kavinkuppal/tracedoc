"""Token and cost accounting: price lookups, live spend tracking, and the --max-cost guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import Price

if TYPE_CHECKING:
    from .client import LLMResult


class BudgetExceededError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English and JSON)."""
    return max(1, len(text) // 4)


def price_of(
    prices: dict[str, Price], model: str | None, tokens_in: int, tokens_out: int
) -> float | None:
    if model is None:
        return None
    price = prices.get(model)
    if price is None:
        # Tolerate dated or provider-prefixed IDs, e.g. "anthropic.claude-opus-5".
        matches = [
            p for name, p in prices.items() if model.endswith(name) or model.startswith(name)
        ]
        price = matches[0] if len(matches) == 1 else None
    if price is None:
        return None
    return (tokens_in * price.input + tokens_out * price.output) / 1_000_000


@dataclass
class Spend:
    """Tracks what analysis has spent, and refuses calls that would cross the budget."""

    prices: dict[str, Price]
    max_cost: float | None = None
    cost: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    unpriced_models: set[str] = field(default_factory=set)
    reused_cost: float = 0.0
    """What answers reused from the cache cost when they were first made (nothing now)."""
    reused_calls: int = 0

    def check(
        self, model: str, projected_in: int, projected_out: int, *, reserved: float = 0.0
    ) -> None:
        """Raise if this call (plus calls already in flight, `reserved`) could cross the budget."""
        if self.max_cost is None:
            return
        projected = price_of(self.prices, model, projected_in, projected_out)
        if projected is None:
            raise BudgetExceededError(
                f"no price for {model}; add it under [prices] to use --max-cost"
            )
        if self.cost + reserved + projected > self.max_cost:
            raise BudgetExceededError(
                f"next call could bring spend to ${self.cost + reserved + projected:.2f}, "
                f"over --max-cost ${self.max_cost:.2f}"
            )

    def record(self, model: str, tokens_in: int, tokens_out: int) -> None:
        self.calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        cost = price_of(self.prices, model, tokens_in, tokens_out)
        if cost is None:
            self.unpriced_models.add(model)
        else:
            self.cost += cost

    def record_reused(self, model: str, tokens_in: int, tokens_out: int) -> None:
        """An answer taken from the cache: free now, but part of what the report cost."""
        self.reused_calls += 1
        cost = price_of(self.prices, model, tokens_in, tokens_out)
        if cost is None:
            self.unpriced_models.add(model)
        else:
            self.reused_cost += cost

    def record_result(self, result: LLMResult) -> None:  # type: ignore[type-arg]
        if result.cached:
            self.record_reused(result.model, result.tokens_in, result.tokens_out)
        else:
            self.record(result.model, result.tokens_in, result.tokens_out)

    @property
    def report_cost(self) -> float:
        """The cost of every model answer behind the report, whether made now or reused."""
        return self.cost + self.reused_cost
