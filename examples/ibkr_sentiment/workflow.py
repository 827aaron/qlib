import qlib
from qlib.config import REG_US
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.data import D
from qlib.contrib.evaluate import backtest_daily, risk_analysis

from ib_insync import IB, Stock, MarketOrder
from transformers import pipeline

import pandas as pd
import fire

class IBKRSentimentBot:
    """Example trading bot integrating Qlib, IBKR and news sentiment."""

    def __init__(self, provider_uri, ib_host="127.0.0.1", ib_port=7497, client_id=1):
        qlib.init(provider_uri=provider_uri, region=REG_US)
        self.ib = IB()
        self.ib.connect(ib_host, ib_port, clientId=client_id)
        # Simple sentiment analyzer from transformers
        self.sentiment = pipeline("sentiment-analysis")
        self.model = None

    def fetch_sentiment(self, headlines):
        """Return average sentiment score for a list of headlines."""
        if not headlines:
            return 0.0
        scores = []
        for text in headlines:
            result = self.sentiment(text)[0]
            score = result["score"] * (1 if result["label"] == "POSITIVE" else -1)
            scores.append(score)
        return float(sum(scores) / len(scores))

    def train(self, start="2008-01-01", end="2014-12-31", market="AAPL"):
        conf = {
            "handler": {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": {
                    "start_time": start,
                    "end_time": end,
                    "fit_start_time": start,
                    "fit_end_time": end,
                    "instruments": market,
                    "freq": "day",
                },
            }
        }
        dataset = DatasetH(**conf)
        self.model = LGBModel()
        self.model.fit(dataset)

    def generate_signal(self, df, sentiment):
        pred = self.model.predict(df)
        return pred + sentiment

    def order(self, symbol, size, action="BUY"):
        contract = Stock(symbol, "SMART", "USD")
        order = MarketOrder(action, size)
        self.ib.placeOrder(contract, order)


def run(provider_uri="~/.qlib/qlib_data/us_data"):
    bot = IBKRSentimentBot(provider_uri)
    bot.train()
    # Demo sentiment from example headlines
    sentiment = bot.fetch_sentiment(["Stocks climb on upbeat earnings", "Market worries fade"])
    # Get latest features
    df = D.features(["AAPL"], ["$close"], start_time="2015-01-02", end_time="2015-01-02")
    signal = bot.generate_signal(df, sentiment)
    print("Predicted signal:", signal.mean())
    # Example order placement
    # bot.order("AAPL", 10, "BUY")
    bot.ib.disconnect()


if __name__ == "__main__":
    fire.Fire(run)
