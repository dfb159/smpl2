import math 

dim = dict({'length':-1,'mass':1,'time':-1,'temperature':1,'momentum':1,'energy':1})

class natural_unit:
    def __init__(self, val=1, massdim=0):
        self.value = val
        if massdim in dim:
            massdim = dim[massdim]
        self.massdimension = massdim

    def check(self, other):
        assert self.massdimension == other.massdimension

    def __pow__(self, other, modulo=None):
        if isinstance(other, natural_unit):
            assert other.massdimension == 0
            other = other.value
        return natural_unit(self.value**other, self.massdimension*other)
    def __rpow__(self, other, modulo=None):
        assert self.massdimension == 0
        return other**self.value
        
    def __add__(self, other):
        if isinstance(other, natural_unit):
            assert self.massdimension == other.massdimension
            return natural_unit(self.value + other.value, self.massdimension)
        else:
            assert self.massdimension == 0
            return natural_unit(self.value + other, self.massdimension)
    def __radd__(self, other):
        if isinstance(other, natural_unit):
            assert self.massdimension == other.massdimension
            return natural_unit(other.value + self.value, self.massdimension)
        else:
            assert self.massdimension == 0
            return natural_unit(other + self.value, self.massdimension)

    def __sub__(self, other):
        if isinstance(other, natural_unit):
            assert self.massdimension == other.massdimension
            return natural_unit(self.value - other.value, self.massdimension)
        else:
            assert self.massdimension == 0
            return natural_unit(self.value - other, self.massdimension)
    def __rsub__(self, other):
        if isinstance(other, natural_unit):
            assert self.massdimension == other.massdimension
            return natural_unit(other.value - self.value, self.massdimension)
        else:
            assert self.massdimension == 0
            return natural_unit(other - self.value, self.massdimension)

    def __mul__(self, other):
        if isinstance(other, natural_unit):
            return natural_unit(self.value * other.value,self.massdimension + other.massdimension)
        else:
            return natural_unit(self.value * other, self.massdimension)
    def __rmul__(self, other):
        if isinstance(other, natural_unit):
            return natural_unit(other.value * self.value ,self.massdimension + other.massdimension)
        else: 
            return natural_unit(other * self.value, self.massdimension)

    def __truediv__(self, other):
        if isinstance(other,natural_unit):
            return natural_unit(self.value / other.value, self.massdimension - other.massdimension)
        else:
            return natural_unit(self.value / other, self.massdimension)
    def __rtruediv__(self, other):
        if isinstance(other, natural_unit):
            return natural_unit(other.value / self.value, other.massdimension - self.massdimension)
        else: 
            return natural_unit(other / self.value, -self.massdimension)

    def __str__(self):
        if self.massdimension == 0:
            return str(self.value)
        return "%s[%i]" % (str(self.value), self.massdimension)
    def __repr__(self):
        if self.massdimension == 0:
            return repr(self.value)
        return "%s[%i]" % (repr(self.value), self.massdimension)
    def __format__(self, fmt):
        if self.massdimension == 0:
            return self.value.__format__(fmt)
        return self.value.__format__(fmt) + "[%i]" % self.massdimension

# factors
Y = yotta  = 1e24
Z = zetta  = 1e21
E = exa    = 1e18
P = peta   = 1e15
T = tera   = 1e12
G = giga   = 1e9
M = mega   = 1e6
K = k=kilo = 1e3
d = deci   = 1e-1
ct = centi  = 1e-2
m = milli  = 1e-3
mu= micro  = 1e-6
n = nano   = 1e-9
p = pico   = 1e-12
f = fempto = 1e-15
a = atto   = 1e-18
z = zepto  = 1e-21
y = yocto  = 1e-24

pi = math.pi
e = math.e

eV = natural_unit(massdim='energy')
c = natural_unit(massdim='length')/natural_unit(massdim='time')
kb = natural_unit(massdim='energy')/natural_unit(massdim='temperature')
hbar = natural_unit(massdim='energy')*natural_unit(massdim='time')


#fundamental
c0 = 299792458 *c# m/s
hbar0 = 4.135667662e-15/(2*pi)*hbar
kb0 = 8.617333262145e-5*kb

# from wikipedia https://de.wikipedia.org/wiki/Nat%C3%BCrliche_Einheiten
J = joule = 1/(1.60218e-19)*eV

meter = 1/hbar0/c0 * natural_unit(massdim='length')
meter.check(natural_unit(massdim='length'))

second = 1/hbar0*natural_unit(massdim='time')
second.check(natural_unit(massdim='time'))

gram = 1/kilo*(1/1.78266e-36) * natural_unit(massdim='mass')
gram.check(natural_unit(massdim='mass'))

kelvin = 1*kb0*natural_unit(massdim='temperature')
kelvin.check(natural_unit(massdim='temperature'))

#composite
barn = 1e-28*meter**2
barn.check(natural_unit(massdim='length')**2)

u = 1/(6.022141e26)*kilo*gram
u.check(natural_unit(massdim='mass'))

Bq = 1/second
Ci = 37*giga *Bq
minute = 60 * second
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 31556952 * second # inclusive 
Hz = hertz = 1/second
W = watt = J/second
