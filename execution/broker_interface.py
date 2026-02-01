from abc import ABC, abstractmethod

class Broker(ABC):
    @abstractmethod
    def place_market_order(self, pair: str, side: str, units: float, sl: float, tp: float) -> str:
        raise NotImplementedError

    @abstractmethod
    def close_position(self, position_id: str) -> None:
        raise NotImplementedError
