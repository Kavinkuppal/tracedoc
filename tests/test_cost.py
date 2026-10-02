from __future__ import annotations

import pytest

from tracedoc.config import DEFAULT_PRICES, Price, load_config
from tracedoc.llm.cost import BudgetExceededError, Spend, price_of


def test_price_of_exact_and_prefixed() -> None:
    assert price_of(DEFAULT_PRICES, "claude-opus-5", 1_000_000, 0) == 5.0
    assert price_of(DEFAULT_PRICES, "anthropic.claude-haiku-4-5", 0, 1_000_000) == 5.0
    assert price_of(DEFAULT_PRICES, "unknown", 10, 10) is None
    assert price_of(DEFAULT_PRICES, None, 10, 10) is None


def test_spend_records_and_flags_unpriced() -> None:
    spend = Spend(prices=DEFAULT_PRICES)
    spend.record("claude-haiku-4-5", 1_000_000, 0)
    spend.record("mystery", 5, 5)
    assert spend.cost == 1.0
    assert spend.calls == 2
    assert spend.unpriced_models == {"mystery"}


def test_budget_includes_in_flight_reservations() -> None:
    spend = Spend(prices=DEFAULT_PRICES, max_cost=1.0)
    spend.check("claude-haiku-4-5", 500_000, 0)  # $0.50 fits
    with pytest.raises(BudgetExceededError):
        spend.check("claude-haiku-4-5", 500_000, 0, reserved=0.6)
    with pytest.raises(BudgetExceededError, match="no price"):
        spend.check("mystery", 1, 1)


def test_user_prices_extend_defaults(tmp_path, monkeypatch) -> None:
    from tracedoc import config as config_module

    monkeypatch.setattr(config_module, "user_config_path", lambda: tmp_path / "none.toml")
    path = tmp_path / "c.toml"
    path.write_text(
        '[prices."my-model"]\ninput = 0.5\noutput = 1.5\n'
        '[prices."claude-opus-5"]\ninput = 4.0\noutput = 20.0\n'
    )
    config = load_config(path)
    assert config.prices["my-model"] == Price(input=0.5, output=1.5)
    assert config.prices["claude-opus-5"].input == 4.0
    assert "claude-haiku-4-5" in config.prices


def test_default_models_are_priced_and_public() -> None:
    from tracedoc.config import LLMConfig
    from tracedoc.share.payload import public_model_name

    llm = LLMConfig()
    assert (llm.triage_model, llm.confirm_model) == (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-5-5",
    )
    for model in (llm.triage_model, llm.confirm_model):
        assert model in DEFAULT_PRICES
        assert public_model_name(model) == model
