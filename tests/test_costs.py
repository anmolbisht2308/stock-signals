import pytest

from stock_signals.costs import CostModel


def test_buy_and_sell_charges():
    c = CostModel(slippage_pct=0.0)
    fill, charges = c.buy(1000.0, 100)  # turnover ₹1,00,000
    exchange, sebi = 2.97, 0.10
    expected = 100 + 15 + exchange + sebi + 0.18 * (exchange + sebi)  # STT + stamp + ...
    assert fill == 1000.0 and charges == pytest.approx(expected)
    _, sell = c.sell(1000.0, 100)
    assert sell == pytest.approx(100 + exchange + sebi + 0.18 * (exchange + sebi) + 15.34)


def test_slippage_is_adverse():
    c = CostModel(slippage_pct=0.1)
    assert c.buy(100.0, 1)[0] == pytest.approx(100.1)
    assert c.sell(100.0, 1)[0] == pytest.approx(99.9)
