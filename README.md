# gamry-parser

[![PyPI](https://img.shields.io/pypi/v/gamry-parser.svg)](https://pypi.org/project/gamry-parser/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gamry-parser.svg)
[![PyPI - License](https://img.shields.io/pypi/l/gamry-parser.svg)](./LICENSE)

Python package for parsing the contents of Gamry EXPLAIN data (DTA) files

## Installation

### Requirements
* pandas

### Package from PyPi
```bash
$ pip install gamry-parser
```

### Local Installation

1. Check out the latest code:
```bash
$ git clone git@github.com:bcliang/gamry-parser.git
```
2. Use setuptools to install the package and register it with pip
```bash
$ python setup.py install
```

## Usage

### Generic GamryParser Example

The following snippet loads a DTA file and prints to screen: (1) experiment type, (2) # of curves, and (3) a random curve in the form of a pandas DataFrame.

```python
import gamry_parser as parser
import random

file = '/enter/the/file/path.dta'
gp = parser.GamryParser()
gp.load(filename=file)

print("experiment type: {}".format(gp.get_experiment_type()))
print("loaded curves: {}".format(gp.get_curve_count()))

curve_index = random.randint(1,gp.get_curve_count())
print("showing curve #{}".format(curve_index))
print(gp.get_curve_data(curve_index))
```

### ChronoAmperometry Example

The `ChronoAmperometry` class is a subclass of `GamryParser`. Executing the method `get_curve_data()` will return a DataFrame with three columns: (1) `T`, (2) `Vf`, and (3) `Im`

In the example, the file is expected to be a simple chronoamperometry experiment (single step, no preconditioning); there will only be a single curve of data contained within the file. In addition, note the use of the `to_timestamp` property, which allows the user to request `get_curve_data` to return a DataFrame with a `T` column containing DateTime objects (as opposed to the default: float seconds since start).

```python
import gamry_parser as parser
import random

file = '/enter/the/file/path.dta'
ca = parser.ChronoAmperometry(to_timestamp=True)
ca.load(filename=file)
print(ca.get_curve_data())
```

Similar procedure should be followed for using the `gamry_parser.CyclicVoltammetry()` and `gamry_parser.Impedance()` parser subclasses. Take a look in `tests/` for some additional usage examples.

## Development

### Roadmap

This package is meant to convert flat-file EXPLAIN data into pandas DataFrames for easy analysis and visualization.

Documentation. Loading of data is straightforward, and hopefully the examples provided in this README provide enough context for any of the subclasses to be used/extended.

In the future, it would be nice to add support for things like equivalent circuit modeling, though at the moment there are some other projects focused specifically on the models and fitting of EIS data (e.g. [kbknudsen/PyEIS](https://github.com/kbknudsen/PyEIS), [ECSHackWeek/impedance.py](https://github.com/ECSHackWeek/impedance.py)).

### Tests

Tests extending `unittest.TestCase` may be found in `/tests/`.

```bash
$ python setup.py test
$ coverage run --source=gamry_parser/ setup.py test
$ coverage report -m
```

Latest output:

```bash
$ coverage report -m
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
gamry_parser\ChronoAmperometry.py      16      0   100%
gamry_parser\CyclicVoltammetry.py      11      0   100%
gamry_parser\GamryParser.py            96      1    99%   63
gamry_parser\Impedance.py               6      0   100%
gamry_parser\__init__.py                1      0   100%
gamry_parser\version.py                 1      0   100%
-----------------------------------------------------------------
TOTAL                                 131      1    99%
```

### Publishing

Use setuptools to build, twine to publish to pypi.

```bash
$ rm -rf dist
$ python setup.py sdist
$ twine upload dist/*
```
