# IBKR Short-Term Trading with News Sentiment

This example demonstrates how to combine Qlib models, a simple news sentiment feature and order execution via Interactive Brokers. It relies on the third-party package [ib_insync](https://github.com/erdewit/ib_insync) for IBKR connectivity and uses a pretrained transformer for basic sentiment analysis.

## Disclaimer

See [LICENSE](../../LICENSE) for the MIT license and warranty disclaimer. This example is for educational purposes only and does not constitute financial advice.

## Usage

1. Prepare U.S. stock data and initialize Qlib in `REG_US` mode.
2. Install extra dependencies:
   ```bash
   pip install ib_insync transformers
   ```
3. Start IB Gateway or Trader Workstation and then run:
   ```bash
   python workflow.py run
   ```
   The script trains a LightGBM model, fetches headline sentiment and shows how to submit an order.
