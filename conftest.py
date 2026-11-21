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
