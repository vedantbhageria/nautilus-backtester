# Best-effort sector/category tags for the Binance perp futures symbols this
# project trades — mirrors the small faded tag Binance shows next to a
# symbol's name on the futures markets page. Not authoritative; update as
# new symbols get added to fetch_binance_csv.SYMBOLS.

SYMBOL_CLASS = {
    'BTCUSDT': 'Major', 'ETHUSDT': 'Major',
    'BNBUSDT': 'Exchange', 'XRPUSDT': 'Payments', 'SOLUSDT': 'Layer 1',
    'TRXUSDT': 'Layer 1', 'HYPEUSDT': 'DeFi', 'DOGEUSDT': 'Meme',
    'ZECUSDT': 'Privacy', 'LABUSDT': 'Other',
    'XLMUSDT': 'Payments', 'XMRUSDT': 'Privacy', 'CCUSDT': 'Other',
    'LINKUSDT': 'Oracle/Infra', 'ADAUSDT': 'Layer 1',
    'BCHUSDT': 'Payments', 'LTCUSDT': 'Payments', 'HBARUSDT': 'Layer 1',
    'SUIUSDT': 'Layer 1', 'AVAXUSDT': 'Layer 1',
    '1000SHIBUSDT': 'Meme', 'NEARUSDT': 'Layer 1', 'TAOUSDT': 'AI',
    'WLFIUSDT': 'DeFi', 'PAXGUSDT': 'RWA',
    'UNIUSDT': 'DeFi', 'ASTERUSDT': 'DeFi', 'WLDUSDT': 'AI',
    'ONDOUSDT': 'RWA', 'DOTUSDT': 'Layer 1',
    'AAVEUSDT': 'DeFi', 'SKYUSDT': 'DeFi', 'MUSDT': 'Other',
    'ETCUSDT': 'Layer 1', 'MORPHOUSDT': 'DeFi',
    'DEXEUSDT': 'DeFi', '1000PEPEUSDT': 'Meme', 'QNTUSDT': 'Oracle/Infra',
    'ATOMUSDT': 'Layer 1', 'RENDERUSDT': 'AI',
    'POLUSDT': 'Layer 2', 'KASUSDT': 'Layer 1', 'ALGOUSDT': 'Layer 1',
    'ENAUSDT': 'DeFi', 'JUPUSDT': 'DeFi',
    'JSTUSDT': 'DeFi', 'BEATUSDT': 'Other', 'VVVUSDT': 'Other',
    'FILUSDT': 'Oracle/Infra', 'NIGHTUSDT': 'Other',
    'APTUSDT': 'Layer 1', 'ARBUSDT': 'Layer 2', 'AEROUSDT': 'DeFi',
    'INJUSDT': 'DeFi', 'DASHUSDT': 'Payments',
    'CAKEUSDT': 'DeFi', 'TRUMPUSDT': 'Meme', 'VETUSDT': 'Layer 1',
    'FETUSDT': 'AI', 'PENGUUSDT': 'Meme',
    'SEIUSDT': 'Layer 1', 'JTOUSDT': 'DeFi', '1000BONKUSDT': 'Meme',
    '1000LUNCUSDT': 'Layer 1', 'ETHFIUSDT': 'DeFi',
    'VIRTUALUSDT': 'AI', 'KITEUSDT': 'Other', 'TIAUSDT': 'Oracle/Infra',
    'SUNUSDT': 'DeFi', 'SKYAIUSDT': 'AI',
    'STXUSDT': 'Layer 2', 'SPXUSDT': 'Meme', 'CRVUSDT': 'DeFi',
    'XPLUSDT': 'Layer 1', 'GRASSUSDT': 'AI',
    'GWEIUSDT': 'Other', 'PYTHUSDT': 'Oracle/Infra', 'XTZUSDT': 'Layer 1',
    'OPUSDT': 'Layer 2', 'MONUSDT': 'Layer 1',
    'CFXUSDT': 'Layer 1', 'JASMYUSDT': 'DePIN/IoT', 'BSVUSDT': 'Payments',
    'BUSDT': 'Other', '1000FLOKIUSDT': 'Meme',
    'PENDLEUSDT': 'DeFi', 'VELVETUSDT': 'DeFi', 'LDOUSDT': 'DeFi',
    'ZROUSDT': 'Oracle/Infra', 'KAIAUSDT': 'Layer 1',
    'AKTUSDT': 'DePIN/IoT', 'GRTUSDT': 'Oracle/Infra', 'STRKUSDT': 'Layer 2',
    'CHZUSDT': 'Gaming', 'UBUSDT': 'Other',
    'AXSUSDT': 'Gaming', 'IOTAUSDT': 'DePIN/IoT', 'ENSUSDT': 'Oracle/Infra',
    'EIGENUSDT': 'DeFi', 'COMPUSDT': 'DeFi',
}


def classify(symbol):
    return SYMBOL_CLASS.get(symbol, 'Other')
