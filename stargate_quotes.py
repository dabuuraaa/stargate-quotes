import argparse, os
from datetime import datetime, timezone
import itertools, json
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd
from tqdm import tqdm

getcontext().prec = 50

API_BASE = "https://stargate.finance/api/v1"

TARGET_CHAIN_KEYS = [
    "ethereum", "bsc", "arbitrum", "polygon", "base", "optimism", "unichain", "solana", "avalanche"
]
TARGET_SYMBOLS = ["USDC", "USDT", "USDT0", "ETH", "BTC"]

DEFAULT_HUMAN_AMOUNTS = {
    "USDC": Decimal("1000"),
    "USDT": Decimal("1000"),
    "USDT0": Decimal("1000"),
    "ETH": Decimal("0.1"),
    "BTC": Decimal("0.1"),
}

SYMBOL_EQUIVALENTS: Dict[str, List[str]] = {
    "ETH":  ["ETH", "WETH"],
    "BTC":  ["WBTC", "BTC.b", "BTCB", "cbBTC"],
    "USDC": ["USDC", "USDC.e", "USDC.n", "stgUSDC"],
    "USDT": ["USDT", "USDt", "m.USDT", "stgUSDT"],
    "USDT0": ["USD₮0"],
}

ADDRESS_BOOK = {
    "evm": {
        "src": "0x1111111111111111111111111111111111111111",
        "dst": "0x2222222222222222222222222222222222222222",
    },
    "solana": {
        "src": "8nVxw8tQsgVYQY7m8L5m88o1fkv4r5HfKxbm3Qz3F8Vt",
        "dst": "6JQb2aL9r1wJ4SbN2s6oT4X6vQw2uWJ1Kk1vQnH1XbYd",
    }
}

OUT_QUOTES_CSV = "stargate_quotes_same_symbol.csv"

class HttpError(Exception):
    def __init__(self, status: int, body: str, headers: Dict[str, str]):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body or ""
        self.headers = dict(headers or {})

def to_wei_str(amount_human: Decimal, decimals: int) -> str:
    q = (amount_human * (Decimal(10) ** decimals)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return str(int(q))

def first_existing_symbol(base: str, available_symbols: List[str]) -> str | None:
    for cand in SYMBOL_EQUIVALENTS.get(base, [base]):
        if cand in available_symbols:
            return cand
    return None

def get_json(url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise HttpError(r.status_code, r.text, r.headers)
    return r.json()

def to_readable(amount_str: str, decimals: int) -> str:
    try:
        return str(Decimal(amount_str) / (Decimal(10) ** decimals))
    except Exception:
        return ""

def _safe_decimal(x: Any) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal(0)

def _sum_decimals_str(nums: List[str]) -> str:
    s = Decimal(0)
    for n in nums:
        try:
            s += Decimal(str(n))
        except Exception:
            pass
    return str(s)

def main():
    chains_json = get_json(f"{API_BASE}/chains")
    tokens_json = get_json(f"{API_BASE}/tokens")

    CHAIN_INFO = {c["chainKey"]: {"chainType": c["chainType"], "name": c["name"]} for c in chains_json["chains"]}

    token_addr_symbol = {}
    token_addr_decimals = {}
    for t in tokens_json["tokens"]:
        key = (t.get("chainKey"), (t.get("address") or "").lower())
        token_addr_symbol[key] = t.get("symbol", "")
        try:
            token_addr_decimals[key] = int(t.get("decimals"))
        except Exception:
            pass

    token_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    allowed_variants = set(s for eqs in SYMBOL_EQUIVALENTS.values() for s in eqs)
    for t in tokens_json["tokens"]:
        if not t.get("isBridgeable"):
            continue
        ck, sym = t["chainKey"], t["symbol"]
        if ck not in TARGET_CHAIN_KEYS or sym not in allowed_variants:
            continue
        token_map.setdefault(ck, {}).setdefault(sym, t)

    target_chain_keys = [ck for ck in TARGET_CHAIN_KEYS if ck in CHAIN_INFO]

    rows = []
    pairs = [(src, dst) for src, dst in itertools.product(target_chain_keys, target_chain_keys) if src != dst]

    for base_symbol in TARGET_SYMBOLS:
        for src_ck, dst_ck in tqdm(pairs, desc=f"Collecting {base_symbol}"):
            src_tokens = token_map.get(src_ck, {})
            dst_tokens = token_map.get(dst_ck, {})
            src_sym = first_existing_symbol(base_symbol, list(src_tokens.keys()))
            dst_sym = first_existing_symbol(base_symbol, list(dst_tokens.keys()))
            if not src_sym or not dst_sym:
                continue

            src_tok, dst_tok = src_tokens[src_sym], dst_tokens[dst_sym]
            src_dec, dst_dec = int(src_tok["decimals"]), int(dst_tok["decimals"])

            src_addr_acc = ADDRESS_BOOK.get(CHAIN_INFO[src_ck]["chainType"], ADDRESS_BOOK["evm"])["src"]
            dst_addr_acc = ADDRESS_BOOK.get(CHAIN_INFO[dst_ck]["chainType"], ADDRESS_BOOK["evm"])["dst"]

            base_human_amount = DEFAULT_HUMAN_AMOUNTS.get(base_symbol, Decimal("1"))
            src_amount_wei = to_wei_str(base_human_amount, src_dec)

            params = {
                "srcToken": src_tok["address"],
                "dstToken": dst_tok["address"],
                "srcAddress": src_addr_acc,
                "dstAddress": dst_addr_acc,
                "srcChainKey": src_ck,
                "dstChainKey": dst_ck,
                "srcAmount": src_amount_wei,
                "dstAmountMin": "0",
            }

            try:
                data = get_json(f"{API_BASE}/quotes", params=params)
            except Exception:
                continue

            quotes = data.get("quotes", []) or []
            if not quotes:
                continue

            for q in quotes:
                fees_list = q.get("fees", []) or []

                lz_fee_items = [f for f in fees_list if (f.get("type") or "").lower() == "message"]
                lz_fee_amounts_wei = [f.get("amount", "0") for f in lz_fee_items]
                lz_fee_amount_total_wei = _sum_decimals_str(lz_fee_amounts_wei)

                first_msg_token = (lz_fee_items[0].get("token") or "") if lz_fee_items else ""
                first_msg_chain = (lz_fee_items[0].get("chainKey") or "") if lz_fee_items else ""
                key0 = (first_msg_chain, first_msg_token.lower()) if first_msg_token else None
                msg_symbol = token_addr_symbol.get(key0, "") if key0 else ""
                msg_decimals = token_addr_decimals.get(key0, 18) if key0 else 18
                lz_fee_amount_total_readable = to_readable(lz_fee_amount_total_wei, msg_decimals) if lz_fee_items else "0"

                lz_fee_token_addresses = ";".join((f.get("token") or "") for f in lz_fee_items)
                lz_fee_token_symbols  = ";".join(token_addr_symbol.get((f.get("chainKey"), (f.get("token") or "").lower()), "") for f in lz_fee_items)
                lz_fee_chainKeys      = ";".join((f.get("chainKey") or "") for f in lz_fee_items)
                lz_fee_types          = ";".join((f.get("type") or "") for f in lz_fee_items)

                src_amt_readable = to_readable(q.get("srcAmount", "0"), src_dec)
                dst_amt_readable = to_readable(q.get("dstAmount", "0"), dst_dec)

                src_amt_dec = _safe_decimal(src_amt_readable)
                dst_amt_dec = _safe_decimal(dst_amt_readable)
                if src_amt_dec > 0:
                    protocol_fee_bps = ((src_amt_dec - dst_amt_dec) / src_amt_dec) * Decimal(10000)
                else:
                    protocol_fee_bps = Decimal(0)

                all_fee_token_addresses = ";".join(f.get("token", "") for f in fees_list)
                all_fee_chainKeys       = ";".join(f.get("chainKey", "") for f in fees_list)
                all_fee_amounts         = ";".join(f.get("amount", "") for f in fees_list)
                all_fee_types           = ";".join(f.get("type", "") for f in fees_list)
                all_fee_token_symbols   = ";".join(
                    token_addr_symbol.get((f.get("chainKey"), (f.get("token") or "").lower()), "")
                    for f in fees_list
                )

                row = {
                    "route": q.get("route"),
                    "src_chain": src_ck,
                    "dst_chain": dst_ck,
                    "src_token": src_sym,
                    "dst_token": dst_sym,
                    "src_amount_wei": q.get("srcAmount"),
                    "dst_amount_wei": q.get("dstAmount"),
                    "src_amount_readable": src_amt_readable,
                    "dst_amount_readable": dst_amt_readable,
                    "fee_bps": float(protocol_fee_bps),  # 負の値になり得る（=実質リワード）
                    # ---- LayerZero message fee（＝Messaging fee）----
                    "lz_message_fee_amount_wei": lz_fee_amount_total_wei,
                    "lz_message_fee_amount_readable": lz_fee_amount_total_readable,
                    "lz_message_fee_token_symbol": msg_symbol,
                    "lz_message_fee_chain": first_msg_chain,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_QUOTES_CSV, index=False)
    print(f"✅ Done. quotes={len(df)} rows")
    print(f"- Quotes: {OUT_QUOTES_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 出力ファイル名にUTC時刻を付与
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    OUT_QUOTES_CSV = os.path.join(args.out_dir, f"stargate_quotes_{ts}.csv")

    main()