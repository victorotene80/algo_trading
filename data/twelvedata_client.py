import os
import requests
import pandas as pd

class TwelveDataClient:
    def __init__(self, api_key: str, base_url: str = "https://api.twelvedata.com"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def from_env(base_url: str) -> "TwelveDataClient":
        api_key = os.getenv("TWELVEDATA_API_KEY")
        if not api_key:
            raise RuntimeError("Missing TWELVEDATA_API_KEY in environment (.env).")
        return TwelveDataClient(api_key=api_key, base_url=base_url)

    def fetch_time_series(self, symbol: str, interval: str, outputsize: int = 800) -> pd.DataFrame:
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON",
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if data.get("status") == "error":
            raise RuntimeError(f"TwelveData error: {data.get('message')}")

        values = data.get("values", [])
        if not values:
            raise RuntimeError("No values returned from TwelveData.")

        df = pd.DataFrame(values)
        df = df.rename(columns={"datetime": "ts"})
        df["ts"] = pd.to_datetime(df["ts"])

        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        else:
            df["volume"] = None

        # newest-first -> oldest-first
        df = df.sort_values("ts").set_index("ts")
        return df[["open", "high", "low", "close", "volume"]].copy()
