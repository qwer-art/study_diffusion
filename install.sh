### numpy和torch的版本冲突
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.0 as it may crash.
```bash
pip uninstall numpy
pip install numpy==1.22.0
```
