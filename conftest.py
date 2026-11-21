"""Optional: run the Stage 1/2 tests against another implementation file.

    SUBMISSION_MODULE=submission_stage4 python -m pytest test_stage1.py test_stage2.py

Without the variable, tests use submission.py as usual.
"""
import importlib
import os
import sys

_name = os.environ.get("SUBMISSION_MODULE")
if _name and _name != "submission":
    sys.modules["submission"] = importlib.import_module(_name)

# Strict mode for every test run: a failed internal verification raises
# instead of silently dropping the candidate, so bugs surface as failures.
os.environ.setdefault("QUBLITZ_STRICT", "1")

# Bulk tests use the reduced "fast" beam so the suites finish in minutes;
# test_stage4.py::test_D_full_beam_guarantees runs the production "full" beam.
os.environ.setdefault("QUBLITZ_BEAM", "fast")
