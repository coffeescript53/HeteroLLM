# Archived C3 analysis source

These two files are byte-identical copies of the accepted October 30 C3 analysis and synthetic tests. They are included for inspection of interval unions, copy matching and dependency-ordering logic.

Run synthetic CPU tests from the export root:

```bash
PYTHONPATH=. python3 -m unittest discover -s analysis/archived/c3 -p 'test_*.py' -v
```

The full analyzer main function requires the original private trace tree. Do not invoke it without that archive; its existing main function writes derived reports beside itself. C4 did not rerun it or alter accepted evidence. The public numerical projection is in evidence/portfolio.
