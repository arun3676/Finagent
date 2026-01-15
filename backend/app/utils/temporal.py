"""
Temporal parsing and fiscal period helpers.

Provides:
- Query parsing for fiscal period constraints
- Fiscal year/quarter derivation from SEC metadata
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import calendar
import re
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class TemporalConstraints:
    """Temporal constraints derived from a query."""
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    fiscal_period: Optional[str] = None


@dataclass(frozen=True)
class DerivedFiscalMetadata:
    """Fiscal metadata derived from SEC filing metadata."""
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    fiscal_period: Optional[str] = None
    period_end_date: Optional[datetime] = None


_QUARTER_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
}


def extract_temporal_constraints(query: str) -> TemporalConstraints:
    """
    Extract fiscal year/quarter constraints from a query.

    Examples:
        "Q4 FY2025 risk factors" -> Q4 FY2025
        "FY2024 revenue" -> FY2024
        "fourth quarter 2025" -> Q4 FY2025
    """
    if not query:
        return TemporalConstraints()

    quarter = None
    year = None

    # Pattern: Q4 FY2025 / Q4 2025
    match = re.search(r"\bQ([1-4])\s*(?:FY|FISCAL)?\s*(\d{2,4})?\b", query, re.IGNORECASE)
    if match:
        quarter = int(match.group(1))
        year = _normalize_year(match.group(2))

    # Pattern: FY2025 Q4 / 2025 Q4
    if quarter is None:
        match = re.search(r"\b(?:FY|FISCAL)?\s*(\d{2,4})\s*Q([1-4])\b", query, re.IGNORECASE)
        if match:
            year = _normalize_year(match.group(1))
            quarter = int(match.group(2))

    # Pattern: "fourth quarter 2025"
    if quarter is None:
        match = re.search(
            r"\b(first|second|third|fourth)\s+quarter(?:\s+(?:of|in))?\s*(?:FY|FISCAL)?\s*(\d{2,4})?\b",
            query,
            re.IGNORECASE,
        )
        if match:
            quarter = _QUARTER_WORDS.get(match.group(1).lower())
            year = _normalize_year(match.group(2))

    # If we found a quarter but no year, try to infer year from query
    if quarter is not None and year is None:
        year = _extract_year_from_query(query)

    # Pattern: FY2025 / fiscal year 2025
    if year is None:
        match = re.search(r"\b(?:FY|FISCAL\s+YEAR|FISCAL)\s*(\d{2,4})\b", query, re.IGNORECASE)
        if match:
            year = _normalize_year(match.group(1))

    fiscal_period = None
    if year and quarter:
        fiscal_period = f"Q{quarter} FY{year}"
    elif year:
        fiscal_period = f"FY{year}"

    return TemporalConstraints(
        fiscal_year=year,
        fiscal_quarter=quarter if year and quarter else None,
        fiscal_period=fiscal_period,
    )


def build_temporal_filters(constraints: TemporalConstraints) -> Dict[str, Any]:
    """
    Build retrieval filters from temporal constraints.
    """
    if constraints.fiscal_period:
        return {"fiscal_period": constraints.fiscal_period}

    filters: Dict[str, Any] = {}
    if constraints.fiscal_year:
        filters["fiscal_year"] = constraints.fiscal_year
    if constraints.fiscal_quarter:
        filters["fiscal_quarter"] = constraints.fiscal_quarter
    return filters


def merge_temporal_filters(
    filters: Optional[Dict[str, Any]],
    constraints: TemporalConstraints
) -> Dict[str, Any]:
    """
    Merge temporal filters into an existing filter dict.
    """
    merged = dict(filters or {})
    for key, value in build_temporal_filters(constraints).items():
        merged.setdefault(key, value)
    return merged


def derive_fiscal_metadata(
    report_date: Optional[date],
    fiscal_year_end_mmdd: Optional[str],
    document_type: Optional[str],
) -> DerivedFiscalMetadata:
    """
    Derive fiscal year/quarter/period from SEC report date and fiscal year end.
    """
    if not report_date:
        return DerivedFiscalMetadata()

    if isinstance(report_date, datetime):
        report_date_dt = report_date
        report_date = report_date.date()
    else:
        report_date_dt = datetime.combine(report_date, datetime.min.time())

    doc_type = _normalize_doc_type(document_type)

    period_end_date = report_date_dt
    fiscal_year_end = _resolve_fiscal_year_end_date(report_date, fiscal_year_end_mmdd)

    fiscal_year = None
    fiscal_quarter = None
    fiscal_period = None

    if fiscal_year_end:
        fiscal_year = fiscal_year_end.year

    if doc_type == "10-K":
        if fiscal_year is None:
            fiscal_year = report_date.year
        fiscal_period = f"FY{fiscal_year}" if fiscal_year else None
    elif doc_type == "10-Q":
        if fiscal_year_end:
            fiscal_quarter = _compute_fiscal_quarter(report_date, fiscal_year_end)
            if fiscal_year is None:
                fiscal_year = fiscal_year_end.year
        else:
            fiscal_year = report_date.year
            fiscal_quarter = (report_date.month - 1) // 3 + 1
        if fiscal_year and fiscal_quarter:
            fiscal_period = f"Q{fiscal_quarter} FY{fiscal_year}"

    return DerivedFiscalMetadata(
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        fiscal_period=fiscal_period,
        period_end_date=period_end_date,
    )


def _normalize_year(year_str: Optional[str]) -> Optional[int]:
    if not year_str:
        return None
    try:
        year = int(year_str)
    except ValueError:
        return None
    if year < 100:
        year += 2000
    return year


def _extract_year_from_query(query: str) -> Optional[int]:
    match = re.search(r"\b(20\d{2})\b", query)
    if match:
        return _normalize_year(match.group(1))
    return None


def _normalize_doc_type(document_type: Optional[str]) -> str:
    if not document_type:
        return ""
    if hasattr(document_type, "value"):
        return str(document_type.value)
    return str(document_type)


def _resolve_fiscal_year_end_date(
    report_date: date,
    fiscal_year_end_mmdd: Optional[str],
    tolerance_days: int = 30,
) -> Optional[date]:
    if not fiscal_year_end_mmdd:
        return None

    fiscal_year_end = _date_from_mmdd(report_date.year, fiscal_year_end_mmdd)
    if fiscal_year_end is None:
        return None

    if report_date >= fiscal_year_end:
        diff = (report_date - fiscal_year_end).days
        if diff <= tolerance_days:
            return fiscal_year_end
        return _date_from_mmdd(report_date.year + 1, fiscal_year_end_mmdd)

    return fiscal_year_end


def _date_from_mmdd(year: int, mmdd: str) -> Optional[date]:
    if not mmdd or len(mmdd) != 4 or not mmdd.isdigit():
        return None
    month = int(mmdd[:2])
    day = int(mmdd[2:])
    if month < 1 or month > 12:
        return None
    last_day = calendar.monthrange(year, month)[1]
    day = min(day, last_day)
    return date(year, month, day)


def _shift_year(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # Handle Feb 29 for non-leap years by clamping to Feb 28.
        return value.replace(year=value.year + years, day=28)


def _compute_fiscal_quarter(report_date: date, fiscal_year_end: date) -> int:
    fiscal_year_start = _shift_year(fiscal_year_end, -1) + timedelta(days=1)
    days_in_year = (fiscal_year_end - fiscal_year_start).days + 1
    if days_in_year <= 0:
        return 4
    quarter_length = days_in_year / 4.0
    days_into_year = (report_date - fiscal_year_start).days
    quarter = int(days_into_year / quarter_length) + 1
    return max(1, min(quarter, 4))
