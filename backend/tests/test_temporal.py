"""
Tests for temporal parsing and fiscal period helpers.
"""

from datetime import datetime

from app.utils.temporal import (
    extract_temporal_constraints,
    build_temporal_filters,
    derive_fiscal_metadata,
)


def test_extract_temporal_constraints_quarter_year():
    constraints = extract_temporal_constraints("What are NVDA Q4 FY2025 risk factors?")
    assert constraints.fiscal_year == 2025
    assert constraints.fiscal_quarter == 4
    assert constraints.fiscal_period == "Q4 FY2025"


def test_extract_temporal_constraints_year_only():
    constraints = extract_temporal_constraints("FY2024 revenue guidance")
    assert constraints.fiscal_year == 2024
    assert constraints.fiscal_quarter is None
    assert constraints.fiscal_period == "FY2024"


def test_extract_temporal_constraints_worded_quarter():
    constraints = extract_temporal_constraints("fourth quarter 2025 results")
    assert constraints.fiscal_year == 2025
    assert constraints.fiscal_quarter == 4
    assert constraints.fiscal_period == "Q4 FY2025"


def test_build_temporal_filters_prefers_period():
    constraints = extract_temporal_constraints("Q1 FY2026 earnings")
    filters = build_temporal_filters(constraints)
    assert filters == {"fiscal_period": "Q1 FY2026"}


def test_derive_fiscal_metadata_quarter_with_year_end():
    report_date = datetime(2025, 10, 26)
    derived = derive_fiscal_metadata(report_date, "0125", "10-Q")
    assert derived.fiscal_year == 2026
    assert derived.fiscal_quarter == 3
    assert derived.fiscal_period == "Q3 FY2026"


def test_derive_fiscal_metadata_annual_with_tolerance():
    report_date = datetime(2025, 1, 26)
    derived = derive_fiscal_metadata(report_date, "0125", "10-K")
    assert derived.fiscal_year == 2025
    assert derived.fiscal_quarter is None
    assert derived.fiscal_period == "FY2025"


def test_derive_fiscal_metadata_without_year_end():
    report_date = datetime(2024, 3, 15)
    derived = derive_fiscal_metadata(report_date, None, "10-Q")
    assert derived.fiscal_year == 2024
    assert derived.fiscal_quarter == 1
    assert derived.fiscal_period == "Q1 FY2024"
