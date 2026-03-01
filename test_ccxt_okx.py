#!/usr/bin/env python3
import os, time, argparse, ccxt

def f(x,d=0.0):
    try: return float(x)
    except: return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ex",default="okx",help="exchange id: okx/bybit/binance/coinbase...")
    ap.add_argument("--symbol",required=True)
    ap.add_argument("--side",choices=["buy","sell"],required=True)
    ap.add_argument("--type",default="market",choices=["market","limit"])
    ap.add_argument("--amount",type=float,required=True)
    ap.add_argument("--price",type=float)
    ap.add_argument("--poll",type=float,default=1.0)
    ap.add_argument("--sandbox",action="store_true")
    ap.add_argument("--params",default="",help='extra params json-ish, e.g. {"tdMode":"cash"} for OKX (optional)')
    args=ap.parse_args()

    excls=getattr(ccxt,args.ex)
    ex=excls({
        "apiKey":"46444df5-6392-4f3e-8704-a72198e3e4c7",
        "secret":"4484CBD56AC869C1CEE767CE8F1A0ACF",
        "password":"Wyf@8244398",  # passphrase/password for some exchanges
        "enableRateLimit":True,
    })
    ex.session.proxies = {"http":"http://127.0.0.1:7890","https":"http://127.0.0.1:7890"}
    ex.session.trust_env = False    
    ex.has["fetchCurrencies"] = False
    ex.fetch_currencies = lambda params={}: {}

    # 2) Demo Trading：切 REST 域名 + 加 header
    if args.sandbox:
        ex.urls["api"]["rest"] = "https://us.okx.com"
        ex.headers = {"x-simulated-trading": "1"}
    else:
        ex.urls["api"]["rest"] = "https://www.okx.com"

    ex.load_markets()

    params={}
    if args.params.strip():
        import json
        params=json.loads(args.params)

    if args.type=="limit" and args.price is None: raise SystemExit("limit order requires --price")

    o = ex.create_order(args.symbol, args.type, args.side, args.amount, args.price if args.type=="limit" else None, params)
    oid=o.get("id"); print("order_id:", oid)

    avg0=f(o.get("average")); filled0=f(o.get("filled")); cost0=f(o.get("cost"))
    t0=time.time()

    while True:
        try:
            o=ex.fetch_order(oid,args.symbol)
        except Exception as e:
            print("fetch_order err:",e); time.sleep(args.poll); continue

        st=o.get("status"); filled=f(o.get("filled")); avg=f(o.get("average")); cost=f(o.get("cost"))
        if avg==0 and filled>0 and cost>0: avg=cost/filled
        if avg0==0 and avg>0: avg0=avg
        if filled0==0 and filled>0: filled0=filled

        # try derivatives position pnl first
        upnl=None
        try:
            if hasattr(ex,"fetch_positions"):
                ps=ex.fetch_positions([args.symbol])
                if ps:
                    p=ps[0]
                    upnl = p.get("unrealizedPnl") or p.get("unrealizedPnlUsd") or p.get("info",{}).get("unrealizedPnl")
        except: pass

        last=None
        try:
            t=ex.fetch_ticker(args.symbol); last=f(t.get("last"))
        except: pass

        # spot-style approximate pnl (needs avg entry)
        pnl=None
        if last and avg0 and filled0:
            dir=1 if args.side=="buy" else -1
            pnl = (last-avg0)*filled0*dir

        age=time.time()-t0
        print(f"t+{age:6.1f}s status={st} filled={filled}/{args.amount} avg={avg or avg0:.8g} last={last or 0:.8g} upnl={upnl if upnl is not None else 'na'} pnl~={pnl if pnl is not None else 'na'}")

        if st in ("closed","canceled","rejected","expired"): break
        time.sleep(args.poll)

if __name__=="__main__": main()


