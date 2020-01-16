from pint import UnitRegistry

units = UnitRegistry(auto_reduce_dimensions=False)
Quantity = units.Quantity
