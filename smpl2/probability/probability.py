from collections import defaultdict
from itertools import product
from math import ceil, floor
from typing import Callable, Generic, ItemsView, Iterator, Self, TypeVar


def _proba_string(propability: float):
    return f"{propability * 100:5.2f} %" if propability >= 1e-3 else format(propability, ".2e")

T = TypeVar("T")

class Discrete_Probability(Generic[T]):
    """A set of discrete probabilities. Mathematical operations leave the total probability unchanged. Bool operations do not."""

    _distribution: dict[T, float] = {}

    def __init__(self, distribution: dict[T, float]):
        self._distribution = distribution

    def __getitem__(self, key: T):
        return self._distribution[key] if key in self._distribution else 0
    
    def __str__(self) -> str:
        return "Probability distribution:\n  " + "\n  ".join(f"{key}: {_proba_string(self._distribution[key])}" for key in sorted(self._distribution))
        
    def __repr__(self) -> str:
        return repr(self._distribution)
    
    def __len__(self) -> int:
        return len(self._distribution)
    
    def __iter__(self) -> Iterator[T]:
        return self._distribution.__iter__()
    
    def items(self) -> ItemsView[T, float]:
        return self._distribution.items()
    
    def __min__(self) -> T:
        return min(self._distribution)
    
    def __max__(self) -> T:
        return max(self._distribution)
    
    def __contains__(self, item: T):
        return item in self._distribution
    
    def __neg__(self) -> Self:
        dist = defaultdict(int)
        for key, value in self:
            dist[-key] += value
        return Discrete_Probability(dist)
    
    def __pos__(self) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[+key] += value
        return Discrete_Probability(dist)
    
    def __invert__(self) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key] += 1 - value
        return Discrete_Probability(dist)
    
    def __abs__(self) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[abs(key)] += value
        return Discrete_Probability(dist)

    def __round__(self, ndigits:int=0) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[round(key, ndigits)] += value
        return Discrete_Probability(dist)
    
    def __floor__(self) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[floor(key)] += value
        return Discrete_Probability(dist)
    
    def __ceil__(self) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[ceil(key)] += value
        return Discrete_Probability(dist)

    def __add__(self, other: T | Self) -> Self:
        return self.__add_self(other) if isinstance(other, Discrete_Probability) else self.__add_T(other)
    
    def __radd__(self, other: T | Self) -> Self:
        return self + other
    
    def __add_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key + other] += value
        return Discrete_Probability(dist)
    
    def __add_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key1 + key2] += value1 * value2
        return Discrete_Probability(dist)

    def __sub__(self, other: T | Self) -> Self:
        return self.__sub_self(other) if isinstance(other, Discrete_Probability) else self.__sub_T(other)
    
    def __rsub__(self, other: T | Self) -> Self:
        return -self + other
    
    def __sub_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key - other] += value
        return Discrete_Probability(dist)
    
    def __sub_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key1 - key2] += value1 * value2
        return Discrete_Probability(dist)

    def __mul__(self, other: T | Self) -> Self:
        return self.__mul_self(other) if isinstance(other, Discrete_Probability) else self.__mul_T(other)
    
    def __rmul__(self, other: T | Self) -> Self:
        return self * other
    
    def __mul_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key * other] += value
        return Discrete_Probability(dist)
    
    def __mul_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key1 * key2] += value1 * value2
        return Discrete_Probability(dist)
    
    def __truediv__(self, other: T | Self) -> Self:
        return self.__truediv_self(other) if isinstance(other, Discrete_Probability) else self.__truediv_T(other)
    
    def __truediv_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key / other] += value
        return Discrete_Probability(dist)
    
    def __truediv_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key1 / key2] += value1 * value2
        return Discrete_Probability(dist)
    
    def __rtruediv__(self, other: T | Self) -> Self:
        return self.__rtruediv_self(other) if isinstance(other, Discrete_Probability) else self.__rtruediv_T(other)
    
    def __rtruediv_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[other / key] += value
        return Discrete_Probability(dist)
    
    def __rtruediv_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key2 / key1] += value1 * value2
        return Discrete_Probability(dist)
    
    def __floordiv__(self, other: T | Self) -> Self:
        return self.__floordiv_self(other) if isinstance(other, Discrete_Probability) else self.__floordiv_T(other)
    
    def __floordiv_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key // other] += value
        return Discrete_Probability(dist)
    
    def __floordiv_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key1 // key2] += value1 * value2
        return Discrete_Probability(dist)
    
    def __rfloordiv__(self, other: T | Self) -> Self:
        return self.__rfloordiv_self(other) if isinstance(other, Discrete_Probability) else self.__rfloordiv_T(other)
    
    def __rfloordiv_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[other // key] += value
        return Discrete_Probability(dist)
    
    def __rfloordiv_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key2 // key1] += value1 * value2
        return Discrete_Probability(dist)
    
    def __mod__(self, other: T | Self) -> Self:
        return self.__mod_self(other) if isinstance(other, Discrete_Probability) else self.__mod_T(other)
    
    def __mod_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key % other] += value
        return Discrete_Probability(dist)
    
    def __mod_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key1 % key2] += value1 * value2
        return Discrete_Probability(dist)
    
    def __rmod__(self, other: T | Self) -> Self:
        return self.__rmod_self(other) if isinstance(other, Discrete_Probability) else self.__rmod_T(other)
    
    def __rmod_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[other % key] += value
        return Discrete_Probability(dist)
    
    def __rmod_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key2 % key1] += value1 * value2
        return Discrete_Probability(dist)
    
    def __pow__(self, other: T | Self) -> Self:
        return self.__pow_self(other) if isinstance(other, Discrete_Probability) else self.__pow_T(other)
    
    def __pow_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[key ** other] += value
        return Discrete_Probability(dist)
    
    def __pow_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key1 ** key2] += value1 * value2
        return Discrete_Probability(dist)
    
    def __rpow__(self, other: T | Self) -> Self:
        return self.__rpow_self(other) if isinstance(other, Discrete_Probability) else self.__rpow_T(other)
    
    def __rpow_T(self, other: T) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[other ** key] += value
        return Discrete_Probability(dist)
    
    def __rpow_self(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                dist[key2 ** key1] += value1 * value2
        return Discrete_Probability(dist)
        
    def __or__(self, other: Self) -> Self:
        return ~(~self & ~other)
    
    def __and__(self, other: Self) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            if key in other: dist[key] += value * other[key]
        return Discrete_Probability(dist)
    
    def __lt__(self, other: T | Self) -> float:
        return self.__lt_self(other) if isinstance(other, Discrete_Probability) else self.__lt_T(other)
    
    def __lt_T(self, other: T) -> float:
        sumvar = 0
        for key, value in self.items():
            if key < other: sumvar += value
        return sumvar
    
    def __lt_self(self, other: Self) -> float:
        sumvar = 0
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                if key1 < key2: sumvar += value1 * value2
        return sumvar
    
    def __le__(self, other: T | Self) -> float:
        return self.__le_self(other) if isinstance(other, Discrete_Probability) else self.__le_T(other)
    
    def __le_T(self, other: T) -> float:
        sumvar = 0
        for key, value in self.items():
            if key <= other: sumvar += value
        return sumvar
    
    def __le_self(self, other: Self) -> float:
        sumvar = 0
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                if key1 <= key2: sumvar += value1 * value2
        return sumvar
        
    def __eq__(self, other: T | Self) -> float:
        return self.__eq_self(other) if isinstance(other, Discrete_Probability) else self.__eq_T(other)
    
    def __eq_T(self, other: T) -> float:
        sumvar = 0
        for key, value in self.items():
            if key == other: sumvar += value
        return sumvar
    
    def __eq_self(self, other: Self) -> float:
        sumvar = 0
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                if key1 == key2: sumvar += value1 * value2
        return sumvar
        
    def __ne__(self, other: T | Self) -> float:
        return self.__ne_self(other) if isinstance(other, Discrete_Probability) else self.__ne_T(other)
    
    def __ne_T(self, other: T) -> float:
        sumvar = 0
        for key, value in self.items():
            if key != other: sumvar += value
        return sumvar
    
    def __ne_self(self, other: Self) -> float:
        sumvar = 0
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                if key1 != key2: sumvar += value1 * value2
        return sumvar
        
    def __gt__(self, other: T | Self) -> float:
        return self.__gt_self(other) if isinstance(other, Discrete_Probability) else self.__gt_T(other)
    
    def __gt_T(self, other: T) -> float:
        sumvar = 0
        for key, value in self.items():
            if key > other: sumvar += value
        return sumvar
    
    def __gt_self(self, other: Self) -> float:
        sumvar = 0
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                if key1 > key2: sumvar += value1 * value2
        return sumvar
        
    def __ge__(self, other: T | Self) -> float:
        return self.__ge_self(other) if isinstance(other, Discrete_Probability) else self.__ge_T(other)
    
    def __ge_T(self, other: T) -> float:
        sumvar = 0
        for key, value in self.items():
            if key >= other: sumvar += value
        return sumvar
    
    def __ge_self(self, other: Self) -> float:
        sumvar = 0
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                if key1 >= key2: sumvar += value1 * value2
        return sumvar

    @property
    def norm(self) -> Self:
        return self / self.total
    
    @property
    def total(self):
        return sum(self._distribution.values())
    
    @property
    def mean(self) -> T:
        return sum(key * value for key, value in self.items()) / self.total
    
    @property
    def variance(self) -> T:
        m = self.mean
        return sum(abs(key - m)**2 * value for key, value in self.items()) / self.total
    
    def map(self, func: Callable[[T], T]) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            dist[func(key)] += value
        return Discrete_Probability(dist)

    def filter(self, func: Callable[[T], bool]) -> Self:
        dist = defaultdict(int)
        for key, value in self.items():
            if func(key): dist[key] += value
        return Discrete_Probability(dist)

class Die(Discrete_Probability[float]):
    """A die with homogenous probabilities for n sides."""

    def __init__(self, sides=6):
        distribution = { i+1: 1/sides for i in range(sides) }
        super().__init__(distribution)

class Dice(Discrete_Probability[tuple[float]]):
    """Multiple dice."""

    def __init__(self, n, sides=6):
        prob = 1/sides**n 
        distribution = { s: prob for s in product(range(1,sides+1), repeat = n) }
        super().__init__(distribution)


dice = Dice(6,20)

def handle(t):
    t = list(t)
    t.remove(min(t))
    t.remove(min(t))
    t.remove(min(t))
    return tuple(t)

d2 = dice.map(handle)
d3 = d2.map(sum)
print(d3)
