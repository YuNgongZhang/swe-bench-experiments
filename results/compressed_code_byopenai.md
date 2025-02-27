#introduction

ID: astropy__astropy-11693
Problem: 'WCS.all_world2pix' failed to converge when plotting WCS with non linear distortions

### Description

When plotting an image with a WCS containing non-linear distortions, it fails with a `NoConvergence` error.

### Expected behavior

Adding `quiet=True` to the call `pixel = self.all_world2pix(*world_arrays, 0)` at line 326 of `astropy/wcs/wcsapi/fitswcs.py` should produce a satisfactory plot.

### Actual behavior

The plotting call fails with a `NoConvergence` error.

### Steps to Reproduce

Code to reproduce the problem:

```python
from astropy.wcs import WCS, Sip
import numpy as np
import matplotlib.pyplot as plt

wcs = WCS(naxis=2)
a = [[0.00000000e+00, 0.00000000e+00, 6.77532513e-07, -1.76632141e-10],
     [0.00000000e+00, 9.49130161e-06, -1.50614321e-07, 0.00000000e+00],
     [7.37260409e-06, 2.07020239e-09, 0.00000000e+00, 0.00000000e+00],
     [-1.20116753e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]
b = [[0.00000000e+00, 0.00000000e+00, 1.34606617e-05, -1.41919055e-07],
     [0.00000000e+00, 5.85158316e-06, -1.10382462e-09, 0.00000000e+00],
     [1.06306407e-05, -1.36469008e-07, 0.00000000e+00, 0.00000000e+00],
     [3.27391123e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]
crpix = [1221.87375165, 994.90917378]
ap = bp = np.zeros((4, 4))

wcs.sip = Sip(a, b, ap, bp, crpix)

plt.subplot(projection=wcs)
plt.imshow(np.zeros((1944, 2592)))
plt.grid(color='white', ls='solid')
```

### System Details

```
Linux-5.11.10-arch1-1-x86_64-with-glibc2.33
Python 3.9.2 (default, Feb 20 2021, 18:40:11) [GCC 10.2.0]
Numpy 1.20.2
astropy 4.3.dev690+g7811614f8
Scipy 1.6.1
Matplotlib 3.3.4
```Patch: diff --git a/astropy/wcs/wcsapi/fitswcs.py b/astropy/wcs/wcsapi/fitswcs.py
--- a/astropy/wcs/wcsapi/fitswcs.py
+++ b/astropy/wcs/wcsapi/fitswcs.py
@@ -323,7 +323,17 @@ def pixel_to_world_values(self, *pixel_arrays):
         return world[0] if self.world_n_dim == 1 else tuple(world)
 
     def world_to_pixel_values(self, *world_arrays):
-        pixel = self.all_world2pix(*world_arrays, 0)
+        # avoid circular import
+        from astropy.wcs.wcs import NoConvergence
+        try:
+            pixel = self.all_world2pix(*world_arrays, 0)
+        except NoConvergence as e:
+            warnings.warn(str(e))
+            # use best_solution contained in the exception and format the same
+            # way as all_world2pix does (using _array_converter)
+            pixel = self._array_converter(lambda *args: e.best_solution,
+                                         'input', *world_arrays, 0)
+
         return pixel[0] if self.pixel_n_dim == 1 else tuple(pixel)
 
     @propertyID: astropy__astropy-12057
Problem: Add helpers to convert between different types of uncertainties

Currently there no easy way to convert between different uncertainty classes, hindering interoperability with external tools. Provide a system to convert NDData objects to external tools that expect uncertainties stored as variances.

```python
from astropy.nddata import (
    VarianceUncertainty, StdDevUncertainty, InverseVariance,
)

def std_to_var(obj):
    return VarianceUncertainty(obj.array ** 2, unit=obj.unit ** 2)

def var_to_invvar(obj):
    return InverseVariance(obj.array ** -1, unit=obj.unit ** -1)

def invvar_to_var(obj):
    return VarianceUncertainty(obj.array ** -1, unit=obj.unit ** -1)

def var_to_std(obj):
    return VarianceUncertainty(obj.array ** 1/2, unit=obj.unit ** 1/2)

FUNC_MAP = {
    (StdDevUncertainty, VarianceUncertainty): std_to_var,
    (StdDevUncertainty, InverseVariance): lambda x: var_to_invvar(std_to_var(x)),
    (VarianceUncertainty, StdDevUncertainty): var_to_std,
    (VarianceUncertainty, InverseVariance): var_to_invvar,
    (InverseVariance, StdDevUncertainty): lambda x: var_to_std(invvar_to_var(x)),
    (InverseVariance, VarianceUncertainty): invvar_to_var,
    (StdDevUncertainty, StdDevUncertainty): lambda x: x,
    (VarianceUncertainty, VarianceUncertainty): lambda x: x,
    (InverseVariance, InverseVariance): lambda x: x,
}

def convert_uncertainties(obj, new_class):
    return FUNC_MAP[(type(obj), new_class)](obj)
```Patch: diff --git a/astropy/nddata/nduncertainty.py b/astropy/nddata/nduncertainty.py
--- a/astropy/nddata/nduncertainty.py
+++ b/astropy/nddata/nduncertainty.py
@@ -395,6 +395,40 @@ def _propagate_multiply(self, other_uncert, result_data, correlation):
     def _propagate_divide(self, other_uncert, result_data, correlation):
         return None
 
+    def represent_as(self, other_uncert):
+        """Convert this uncertainty to a different uncertainty type.
+
+        Parameters
+        ----------
+        other_uncert : `NDUncertainty` subclass
+            The `NDUncertainty` subclass to convert to.
+
+        Returns
+        -------
+        resulting_uncertainty : `NDUncertainty` instance
+            An instance of ``other_uncert`` subclass containing the uncertainty
+            converted to the new uncertainty type.
+
+        Raises
+        ------
+        TypeError
+            If either the initial or final subclasses do not support
+            conversion, a `TypeError` is raised.
+        """
+        as_variance = getattr(self, "_convert_to_variance", None)
+        if as_variance is None:
+            raise TypeError(
+                f"{type(self)} does not support conversion to another "
+                "uncertainty type."
+            )
+        from_variance = getattr(other_uncert, "_convert_from_variance", None)
+        if from_variance is None:
+            raise TypeError(
+                f"{other_uncert.__name__} does not support conversion from "
+                "another uncertainty type."
+            )
+        return from_variance(as_variance())
+
 
 class UnknownUncertainty(NDUncertainty):
     """This class implements any unknown uncertainty type.
@@ -748,6 +782,17 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
     def _data_unit_to_uncertainty_unit(self, value):
         return value
 
+    def _convert_to_variance(self):
+        new_array = None if self.array is None else self.array ** 2
+        new_unit = None if self.unit is None else self.unit ** 2
+        return VarianceUncertainty(new_array, unit=new_unit)
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        new_array = None if var_uncert.array is None else var_uncert.array ** (1 / 2)
+        new_unit = None if var_uncert.unit is None else var_uncert.unit ** (1 / 2)
+        return cls(new_array, unit=new_unit)
+
 
 class VarianceUncertainty(_VariancePropagationMixin, NDUncertainty):
     """
@@ -834,6 +879,13 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
     def _data_unit_to_uncertainty_unit(self, value):
         return value ** 2
 
+    def _convert_to_variance(self):
+        return self
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        return var_uncert
+
 
 def _inverse(x):
     """Just a simple inverse for use in the InverseVariance"""
@@ -933,3 +985,14 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
 
     def _data_unit_to_uncertainty_unit(self, value):
         return 1 / value ** 2
+
+    def _convert_to_variance(self):
+        new_array = None if self.array is None else 1 / self.array
+        new_unit = None if self.unit is None else 1 / self.unit
+        return VarianceUncertainty(new_array, unit=new_unit)
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        new_array = None if var_uncert.array is None else 1 / var_uncert.array
+        new_unit = None if var_uncert.unit is None else 1 / var_uncert.unit
+        return cls(new_array, unit=new_unit)ID: astropy__astropy-12318
Problem: BlackBody bolometric flux is incorrect with scale as dimensionless_unscaled Quantity
The `astropy.modeling.models.BlackBody` class calculates incorrect bolometric flux with `scale` as a Quantity with `dimensionless_unscaled` units, but correct flux with `scale` as a float.

### Description

### Expected behavior

Expected output from sample code:

```
4.823870774433646e-16 erg / (cm2 s)
4.823870774433646e-16 erg / (cm2 s)
```

### Actual behavior

Actual output from sample code:

```
4.5930032795393893e+33 erg / (cm2 s)
4.823870774433646e-16 erg / (cm2 s)
```

### Steps to Reproduce

Sample code:

```python
from astropy.modeling.models import BlackBody
from astropy import units as u
import numpy as np

T = 3000 * u.K
r = 1e14 * u.cm
DL = 100 * u.Mpc
scale = np.pi * (r / DL)**2

print(BlackBody(temperature=T, scale=scale).bolometric_flux)
print(BlackBody(temperature=T, scale=scale.to_value(u.dimensionless_unscaled)).bolometric_flux)
```

### System Details

```pycon
>>> import numpy; print("Numpy", numpy.__version__)
Numpy 1.20.2
>>> import astropy; print("astropy", astropy.__version__)
astropy 4.3.dev758+g1ed1d945a
>>> import scipy; print("Scipy", scipy.__version__)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'scipy'
>>> import matplotlib; print("Matplotlib", matplotlib.__version__)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'matplotlib'
```Patch: diff --git a/astropy/modeling/physical_models.py b/astropy/modeling/physical_models.py
--- a/astropy/modeling/physical_models.py
+++ b/astropy/modeling/physical_models.py
@@ -27,7 +27,12 @@ class BlackBody(Fittable1DModel):
         Blackbody temperature.
 
     scale : float or `~astropy.units.Quantity` ['dimensionless']
-        Scale factor
+        Scale factor.  If dimensionless, input units will assumed
+        to be in Hz and output units in (erg / (cm ** 2 * s * Hz * sr).
+        If not dimensionless, must be equivalent to either
+        (erg / (cm ** 2 * s * Hz * sr) or erg / (cm ** 2 * s * AA * sr),
+        in which case the result will be returned in the requested units and
+        the scale will be stripped of units (with the float value applied).
 
     Notes
     -----
@@ -70,12 +75,40 @@ class BlackBody(Fittable1DModel):
     scale = Parameter(default=1.0, min=0, description="Scale factor")
 
     # We allow values without units to be passed when evaluating the model, and
-    # in this case the input x values are assumed to be frequencies in Hz.
+    # in this case the input x values are assumed to be frequencies in Hz or wavelengths
+    # in AA (depending on the choice of output units controlled by units on scale
+    # and stored in self._output_units during init).
     _input_units_allow_dimensionless = True
 
     # We enable the spectral equivalency by default for the spectral axis
     input_units_equivalencies = {'x': u.spectral()}
 
+    # Store the native units returned by B_nu equation
+    _native_units = u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr)
+
+    # Store the base native output units.  If scale is not dimensionless, it
+    # must be equivalent to one of these.  If equivalent to SLAM, then
+    # input_units will expect AA for 'x', otherwise Hz.
+    _native_output_units = {'SNU': u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr),
+                            'SLAM': u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)}
+
+    def __init__(self, *args, **kwargs):
+        scale = kwargs.get('scale', None)
+
+        # Support scale with non-dimensionless unit by stripping the unit and
+        # storing as self._output_units.
+        if hasattr(scale, 'unit') and not scale.unit.is_equivalent(u.dimensionless_unscaled):
+            output_units = scale.unit
+            if not output_units.is_equivalent(self._native_units, u.spectral_density(1*u.AA)):
+                raise ValueError(f"scale units not dimensionless or in surface brightness: {output_units}")
+
+            kwargs['scale'] = scale.value
+            self._output_units = output_units
+        else:
+            self._output_units = self._native_units
+
+        return super().__init__(*args, **kwargs)
+
     def evaluate(self, x, temperature, scale):
         """Evaluate the model.
 
@@ -83,7 +116,8 @@ def evaluate(self, x, temperature, scale):
         ----------
         x : float, `~numpy.ndarray`, or `~astropy.units.Quantity` ['frequency']
             Frequency at which to compute the blackbody. If no units are given,
-            this defaults to Hz.
+            this defaults to Hz (or AA if `scale` was initialized with units
+            equivalent to erg / (cm ** 2 * s * AA * sr)).
 
         temperature : float, `~numpy.ndarray`, or `~astropy.units.Quantity`
             Temperature of the blackbody. If no units are given, this defaults
@@ -119,30 +153,18 @@ def evaluate(self, x, temperature, scale):
         else:
             in_temp = temperature
 
+        if not isinstance(x, u.Quantity):
+            # then we assume it has input_units which depends on the
+            # requested output units (either Hz or AA)
+            in_x = u.Quantity(x, self.input_units['x'])
+        else:
+            in_x = x
+
         # Convert to units for calculations, also force double precision
         with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
-            freq = u.Quantity(x, u.Hz, dtype=np.float64)
+            freq = u.Quantity(in_x, u.Hz, dtype=np.float64)
             temp = u.Quantity(in_temp, u.K)
 
-        # check the units of scale and setup the output units
-        bb_unit = u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr)  # default unit
-        # use the scale that was used at initialization for determining the units to return
-        # to support returning the right units when fitting where units are stripped
-        if hasattr(self.scale, "unit") and self.scale.unit is not None:
-            # check that the units on scale are covertable to surface brightness units
-            if not self.scale.unit.is_equivalent(bb_unit, u.spectral_density(x)):
-                raise ValueError(
-                    f"scale units not surface brightness: {self.scale.unit}"
-                )
-            # use the scale passed to get the value for scaling
-            if hasattr(scale, "unit"):
-                mult_scale = scale.value
-            else:
-                mult_scale = scale
-            bb_unit = self.scale.unit
-        else:
-            mult_scale = scale
-
         # Check if input values are physically possible
         if np.any(temp < 0):
             raise ValueError(f"Temperature should be positive: {temp}")
@@ -158,7 +180,17 @@ def evaluate(self, x, temperature, scale):
         # Calculate blackbody flux
         bb_nu = 2.0 * const.h * freq ** 3 / (const.c ** 2 * boltzm1) / u.sr
 
-        y = mult_scale * bb_nu.to(bb_unit, u.spectral_density(freq))
+        if self.scale.unit is not None:
+            # Will be dimensionless at this point, but may not be dimensionless_unscaled
+            if not hasattr(scale, 'unit'):
+                # during fitting, scale will be passed without units
+                # but we still need to convert from the input dimensionless
+                # to dimensionless unscaled
+                scale = scale * self.scale.unit
+            scale = scale.to(u.dimensionless_unscaled).value
+
+        # NOTE: scale is already stripped of any input units
+        y = scale * bb_nu.to(self._output_units, u.spectral_density(freq))
 
         # If the temperature parameter has no unit, we should return a unitless
         # value. This occurs for instance during fitting, since we drop the
@@ -169,10 +201,13 @@ def evaluate(self, x, temperature, scale):
 
     @property
     def input_units(self):
-        # The input units are those of the 'x' value, which should always be
-        # Hz. Because we do this, and because input_units_allow_dimensionless
-        # is set to True, dimensionless values are assumed to be in Hz.
-        return {self.inputs[0]: u.Hz}
+        # The input units are those of the 'x' value, which will depend on the
+        # units compatible with the expected output units.
+        if self._output_units.is_equivalent(self._native_output_units['SNU']):
+            return {self.inputs[0]: u.Hz}
+        else:
+            # only other option is equivalent with SLAM
+            return {self.inputs[0]: u.AA}
 
     def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
         return {"temperature": u.K}
@@ -180,9 +215,15 @@ def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
     @property
     def bolometric_flux(self):
         """Bolometric flux."""
+        if self.scale.unit is not None:
+            # Will be dimensionless at this point, but may not be dimensionless_unscaled
+            scale = self.scale.quantity.to(u.dimensionless_unscaled)
+        else:
+            scale = self.scale.value
+
         # bolometric flux in the native units of the planck function
         native_bolflux = (
-            self.scale.value * const.sigma_sb * self.temperature ** 4 / np.pi
+            scale * const.sigma_sb * self.temperature ** 4 / np.pi
         )
         # return in more "astro" units
         return native_bolflux.to(u.erg / (u.cm ** 2 * u.s))**Problem:** Can Table masking be turned off?

As of Astropy 5, `astropy.table.Table.read()` creates a `MaskedTable` when encountering `NaN` values. This can be inconvenient for intermediate data in pipelines.

**Description:**

A keyword like `mask=False` in `Table.read(filename, ...)` should disable this behavior for users who don't need masking.Patch: diff --git a/astropy/io/fits/connect.py b/astropy/io/fits/connect.py
--- a/astropy/io/fits/connect.py
+++ b/astropy/io/fits/connect.py
@@ -112,7 +112,8 @@ def _decode_mixins(tbl):
 
 
 def read_table_fits(input, hdu=None, astropy_native=False, memmap=False,
-                    character_as_bytes=True, unit_parse_strict='warn'):
+                    character_as_bytes=True, unit_parse_strict='warn',
+                    mask_invalid=True):
     """
     Read a Table object from an FITS file
 
@@ -145,6 +146,8 @@ def read_table_fits(input, hdu=None, astropy_native=False, memmap=False,
         fit the table in memory, you may be better off leaving memory mapping
         off. However, if your table would not fit in memory, you should set this
         to `True`.
+        When set to `True` then ``mask_invalid`` is set to `False` since the
+        masking would cause loading the full data array.
     character_as_bytes : bool, optional
         If `True`, string columns are stored as Numpy byte arrays (dtype ``S``)
         and are converted on-the-fly to unicode strings when accessing
@@ -158,6 +161,11 @@ def read_table_fits(input, hdu=None, astropy_native=False, memmap=False,
         :class:`~astropy.units.core.UnrecognizedUnit`.
         Values are the ones allowed by the ``parse_strict`` argument of
         :class:`~astropy.units.core.Unit`: ``raise``, ``warn`` and ``silent``.
+    mask_invalid : bool, optional
+        By default the code masks NaNs in float columns and empty strings in
+        string columns. Set this parameter to `False` to avoid the performance
+        penalty of doing this masking step. The masking is always deactivated
+        when using ``memmap=True`` (see above).
 
     """
 
@@ -214,6 +222,11 @@ def read_table_fits(input, hdu=None, astropy_native=False, memmap=False,
 
     else:
 
+        if memmap:
+            # using memmap is not compatible with masking invalid value by
+            # default so we deactivate the masking
+            mask_invalid = False
+
         hdulist = fits_open(input, character_as_bytes=character_as_bytes,
                             memmap=memmap)
 
@@ -222,6 +235,7 @@ def read_table_fits(input, hdu=None, astropy_native=False, memmap=False,
                 hdulist, hdu=hdu,
                 astropy_native=astropy_native,
                 unit_parse_strict=unit_parse_strict,
+                mask_invalid=mask_invalid,
             )
         finally:
             hdulist.close()
@@ -246,9 +260,9 @@ def read_table_fits(input, hdu=None, astropy_native=False, memmap=False,
             # Return a MaskedColumn even if no elements are masked so
             # we roundtrip better.
             masked = True
-        elif issubclass(coltype, np.inexact):
+        elif mask_invalid and issubclass(coltype, np.inexact):
             mask = np.isnan(data[col.name])
-        elif issubclass(coltype, np.character):
+        elif mask_invalid and issubclass(coltype, np.character):
             mask = col.array == b''
 
         if masked or np.any(mask):ID: astropy__astropy-12825
Problem: SkyCoord in Table breaks aggregate on group_by

```python
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

ras = [10, 20] * u.deg
decs = [32, -2] * u.deg
str_col = ['foo', 'bar']
coords = SkyCoord(ra=ras, dec=decs)
table = Table([str_col, coords], names=['col1', 'col2'])
table.group_by('col1').groups.aggregate(np.mean)
```

Aggregation fails with AttributeError for SkyCoord columns.

Aggregation should work except for columns where it doesn't make sense.

System Details:

- Linux-5.14.11-arch1-1-x86_64-with-glibc2.33
- Python 3.9.7
- Numpy 1.21.2
- astropy 5.0.dev945+g7dfa1edb2

- Linux-5.14.11-arch1-1-x86_64-with-glibc2.33
- Python 3.9.7
- Numpy 1.21.2
- astropy 4.3.1
- Scipy 1.7.1
- Matplotlib 3.4.3Patch: diff --git a/astropy/table/column.py b/astropy/table/column.py
--- a/astropy/table/column.py
+++ b/astropy/table/column.py
@@ -340,7 +340,9 @@ class ColumnInfo(BaseColumnInfo):
     This is required when the object is used as a mixin column within a table,
     but can be used as a general way to store meta information.
     """
-    attrs_from_parent = BaseColumnInfo.attr_names
+    attr_names = BaseColumnInfo.attr_names | {'groups'}
+    _attrs_no_copy = BaseColumnInfo._attrs_no_copy | {'groups'}
+    attrs_from_parent = attr_names
     _supports_indexing = True
 
     def new_like(self, cols, length, metadata_conflicts='warn', name=None):
diff --git a/astropy/table/groups.py b/astropy/table/groups.py
--- a/astropy/table/groups.py
+++ b/astropy/table/groups.py
@@ -214,7 +214,7 @@ def __len__(self):
 class ColumnGroups(BaseGroups):
     def __init__(self, parent_column, indices=None, keys=None):
         self.parent_column = parent_column  # parent Column
-        self.parent_table = parent_column.parent_table
+        self.parent_table = parent_column.info.parent_table
         self._indices = indices
         self._keys = keys
 
@@ -238,7 +238,8 @@ def keys(self):
             return self._keys
 
     def aggregate(self, func):
-        from .column import MaskedColumn
+        from .column import MaskedColumn, Column
+        from astropy.utils.compat import NUMPY_LT_1_20
 
         i0s, i1s = self.indices[:-1], self.indices[1:]
         par_col = self.parent_column
@@ -248,6 +249,15 @@ def aggregate(self, func):
         mean_case = func is np.mean
         try:
             if not masked and (reduceat or sum_case or mean_case):
+                # For numpy < 1.20 there is a bug where reduceat will fail to
+                # raise an exception for mixin columns that do not support the
+                # operation. For details see:
+                # https://github.com/astropy/astropy/pull/12825#issuecomment-1082412447
+                # Instead we try the function directly with a 2-element version
+                # of the column
+                if NUMPY_LT_1_20 and not isinstance(par_col, Column) and len(par_col) > 0:
+                    func(par_col[[0, 0]])
+
                 if mean_case:
                     vals = np.add.reduceat(par_col, i0s) / np.diff(self.indices)
                 else:
@@ -256,17 +266,18 @@ def aggregate(self, func):
                     vals = func.reduceat(par_col, i0s)
             else:
                 vals = np.array([func(par_col[i0: i1]) for i0, i1 in zip(i0s, i1s)])
+            out = par_col.__class__(vals)
         except Exception as err:
-            raise TypeError("Cannot aggregate column '{}' with type '{}'"
-                            .format(par_col.info.name,
-                                    par_col.info.dtype)) from err
-
-        out = par_col.__class__(data=vals,
-                                name=par_col.info.name,
-                                description=par_col.info.description,
-                                unit=par_col.info.unit,
-                                format=par_col.info.format,
-                                meta=par_col.info.meta)
+            raise TypeError("Cannot aggregate column '{}' with type '{}': {}"
+                            .format(par_col.info.name, par_col.info.dtype, err)) from err
+
+        out_info = out.info
+        for attr in ('name', 'unit', 'format', 'description', 'meta'):
+            try:
+                setattr(out_info, attr, getattr(par_col.info, attr))
+            except AttributeError:
+                pass
+
         return out
 
     def filter(self, func):
@@ -354,7 +365,7 @@ def aggregate(self, func):
                 new_col = col.take(i0s)
             else:
                 try:
-                    new_col = col.groups.aggregate(func)
+                    new_col = col.info.groups.aggregate(func)
                 except TypeError as err:
                     warnings.warn(str(err), AstropyUserWarning)
                     continue
diff --git a/astropy/utils/data_info.py b/astropy/utils/data_info.py
--- a/astropy/utils/data_info.py
+++ b/astropy/utils/data_info.py
@@ -511,7 +511,7 @@ class BaseColumnInfo(DataInfo):
     Note that this class is defined here so that mixins can use it
     without importing the table package.
     """
-    attr_names = DataInfo.attr_names.union(['parent_table', 'indices'])
+    attr_names = DataInfo.attr_names | {'parent_table', 'indices'}
     _attrs_no_copy = set(['parent_table', 'indices'])
 
     # Context for serialization.  This can be set temporarily via
@@ -752,6 +752,15 @@ def name(self, name):
 
         self._attrs['name'] = name
 
+    @property
+    def groups(self):
+        # This implementation for mixin columns essentially matches the Column
+        # property definition.  `groups` is a read-only property here and
+        # depends on the parent table of the column having `groups`. This will
+        # allow aggregating mixins as long as they support those operations.
+        from astropy.table import groups
+        return self._attrs.setdefault('groups', groups.ColumnGroups(self._parent))
+
 
 class ParentDtypeInfo(MixinInfo):
     """Mixin that gets info.dtype from parent"""I am ready to help! Please provide the instructions that need to be compressed.
