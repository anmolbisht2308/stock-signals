"""Indian equity delivery costs (NSE) and slippage.

Rates as of 2025–26; check your broker's contract note and adjust via settings.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    brokerage_per_order: float = 0.0
    stt_pct: float = 0.1  # delivery: buy and sell
    exchange_pct: float = 0.00297  # NSE transaction charge
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.015
    gst_pct: float = 18.0  # on brokerage + exchange + SEBI fee
    dp_per_sell: float = 15.34  # depository charge per scrip per sell (incl. GST, varies by broker)
    slippage_pct: float = 0.1

    def _charges(self, turnover: float, buy: bool) -> float:
        exchange = turnover * self.exchange_pct / 100
        sebi = turnover * self.sebi_per_crore / 1e7
        gst = (self.brokerage_per_order + exchange + sebi) * self.gst_pct / 100
        stt = turnover * self.stt_pct / 100
        stamp = turnover * self.stamp_buy_pct / 100 if buy else 0.0
        dp = 0.0 if buy else self.dp_per_sell
        return self.brokerage_per_order + exchange + sebi + gst + stt + stamp + dp

    def buy(self, price: float, qty: int) -> tuple[float, float]:
        """(fill price after slippage, charges in ₹)"""
        fill = price * (1 + self.slippage_pct / 100)
        return fill, self._charges(fill * qty, buy=True)

    def sell(self, price: float, qty: int) -> tuple[float, float]:
        fill = price * (1 - self.slippage_pct / 100)
        return fill, self._charges(fill * qty, buy=False)
