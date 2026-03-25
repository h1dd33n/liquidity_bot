if __name__ == "__main__":
    import ccxt
    ex = ccxt.binance({"options": {"defaultType": "future"}})
    ex.load_markets()
    print([s for s in ex.symbols if "PEPE" in s])