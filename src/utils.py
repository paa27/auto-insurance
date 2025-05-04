def metrics(prices_model, prices_competitors):
    has_sold = prices_model < prices_competitors
    market_share = has_sold.mean()
    avg_loss = (prices_competitors[has_sold] - prices_model[has_sold]).mean()
    return avg_loss, market_share