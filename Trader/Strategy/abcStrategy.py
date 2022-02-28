from abc import ABC, abstractmethod


class Strategy(ABC):
    @property
    def signal(self):
        return {
            'position': self._position(),
            'st': self._estimate_st(),
            'tp': self._estimate_tp(),
            'volume': self._estimate_volume(),
            # 'expiration': self._estimate_expire_time(),
        }

    def _position(self):
        if self._sell_zone():
            return "SELL"
        elif self._buy_zone():
            return "BUY"
        else:
            return False

    @abstractmethod
    def _buy_zone(self) -> bool:
        raise NotImplemented

    @abstractmethod
    def _sell_zone(self) -> bool:
        raise NotImplemented

    @abstractmethod
    def _estimate_st(self) -> float:
        raise NotImplemented

    @abstractmethod
    def _estimate_tp(self) -> float:
        raise NotImplemented

    @abstractmethod
    def _estimate_volume(self) -> float:
        raise NotImplemented

    # @abstractmethod
    # def _estimate_expire_time(self) :
    #     raise NotImplemented

