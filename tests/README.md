# Tests

These tests use synthetic fixtures; no bee photographs or model weights
are needed. With NumPy and Pillow installed, run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The private archive also has integration checks that bind the full frozen
splits and training artifacts. Those files and checks are not included here.
Passing these tests is not evidence of biological identification accuracy.
