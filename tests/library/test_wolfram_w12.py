"""Differential tests for Wolfram batch W12 (dates and times, Dataset/Query, Tabular, relational databases).

Expected values are Wolfram Language results from the reference pages (reference.wolfram.com/language/ref/<Name>)
or results verified independently (Python's datetime/zoneinfo, published Julian-date and ephemeris values, SQLite).
Every case that depends on the time zone fixes it with Block[{$TimeZone = z}, ...] or an explicit TimeZone, so the
results do not depend on the machine running the tests.
"""
import pytest

from _harness import Pred, Raises, check, ids, ev


def T(src):
    """The expression must evaluate to True."""
    return (src, True)


def F(src):
    return (src, False)


def tz0(src):
    return f"Block[{{$TimeZone = 0}}, {src}]"


def _num(x):
    from _harness import to_complex
    return to_complex(x).real


def near(x, tol=1e-6):
    return Pred(lambda g: abs(_num(g) - x) <= tol * max(1.0, abs(x)), f"~{x}")


DATEOBJECT = [
    ("DateObject[{2022, 2, 31}]", 'DateObject[{2022, 3, 3}, "Day"]'),
    ("DateObject[{2025, 0, 0}]", 'DateObject[{2024, 11, 30}, "Day"]'),
    ("DateObject[{2022}]", 'DateObject[{2022}, "Year"]'),
    ("DateObject[{2022, 12}]", 'DateObject[{2022, 12}, "Month"]'),
    ('DateObject[{2022, 8, 4}, "Week"]', 'DateObject[{2022, 8, 1}, "Week"]'),
    ('DateObject[DateObject[{2022, 8, 1}, "Week"], "Month"]', 'DateObject[{2022, 8}, "Month"]'),
    ('DateObject[{2016}, "Decade"]', 'DateObject[{2010}, "Decade"]'),
    ('DateObject[{2022, 4, 3}, "Year"]', 'DateObject[{2022}, "Year"]'),
    ('DateObject[{2017, 1, 18}, "WeekBeginningSunday"]', 'DateObject[{2017, 1, 15}, "WeekBeginningSunday"]'),
    ('DateObject[{2017, 5, 18}, "Quarter"]', 'DateObject[{2017, 4}, "Quarter"]'),
    ('DateObject[{2001, 5}, "CenturyBeginning01"]', 'DateObject[{2001}, "CenturyBeginning01"]'),
    ('DateObject[{2000, 5}, "CenturyBeginning01"]', 'DateObject[{1901}, "CenturyBeginning01"]'),
    ('DateObject[{1999}, "Millennium"]', 'DateObject[{1000}, "Millennium"]'),
    ("DateObject[{2020, 1, 1, 10, 30}, TimeZone -> 0]", 'DateObject[{2020, 1, 1, 10, 30}, "Minute", "Gregorian", 0.]'),
    ("DateObject[{2020, 1, 1, 10, 30, 15}, TimeZone -> -5]",
     'DateObject[{2020, 1, 1, 10, 30, 15}, "Instant", "Gregorian", -5.]'),
    ("DateObject[DateObject[{2020, 1, 1, 10, 30, 15}, TimeZone -> -5], TimeZone -> 0]",
     'DateObject[{2020, 1, 1, 15, 30, 15}, "Instant", "Gregorian", 0.]'),
    ("DateObject[3865673600, TimeZone -> 0]", 'DateObject[{2022, 7, 1, 14, 13, 20}, "Instant", "Gregorian", 0.]'),
    (tz0("DateObject[3865673600]"), 'DateObject[{2022, 7, 1, 14, 13, 20}, "Instant", "Gregorian", 0.]'),
    ('DateObject["2022 12-15"]', 'DateObject[{2022, 12, 15}, "Day"]'),
    ('DateObject["Jan 5, 2021"]', 'DateObject[{2021, 1, 5}, "Day"]'),
    ('DateObject["2021-03-04T10:20:30+01:00", TimeZone -> 0]',
     'DateObject[{2021, 3, 4, 9, 20, 30}, "Second", "Gregorian", 0.]'),
    ('DateObject[{"03/04/2021", {"Day", "/", "Month", "/", "Year"}}]', 'DateObject[{2021, 4, 3}, "Day"]'),
    ('DateObject[{2020, 1, 1}, "MidpointInstant", TimeZone -> 0]',
     'DateObject[{2020, 1, 1, 12, 0, 0}, "Instant", "Gregorian", 0.]'),
    ('DateObject[{2020, 1, 1}, "Day", CalendarType -> "Julian"]', 'DateObject[{2019, 12, 19}, "Day", "Julian", None]'),
    ('DateObject[{2022, 1, 1}, "Day"]', 'DateObject[{2022, 1, 1}, "Day"]'),
    ('DateObject[{2016, 12, 31, 12, 0, 0}, "Instant", "Gregorian", 0.]',
     'DateObject[{2016, 12, 31, 12, 0, 0}, "Instant", "Gregorian", 0.]'),
    ('DateObject["foo"]', Raises()),
    ('DateObject[{2020, 1, 1}, "Fortnight"]', Raises()),
    ('DateObject[{0, 1, 1}]', Raises()),
    ('DateObject[{2020, 5, 17}]["DayName"]', "Sunday"),
    ('DateObject[{2020, 5, 17}, "Year"]["Year"]', "2020"),
    ('DateObject[{2020, 5, 17}]["Granularity"]', '"Day"'),
    ('DateObject[{2020, 5, 17, 3, 4, 5}, TimeZone -> 2]["TimeZone"]', "2."),
]
DATEOBJECTQ = [
    T('DateObjectQ[DateObject[{2020, 1, 1}, "Day"]]'),
    F('DateObjectQ[{2020, 1, 1}]'),
    F('DateObjectQ[Unevaluated[DateObject[{2020, 1, 1, 5}, "Day"]]]'),
    F('DateObjectQ[DateObject]'),
]
DATELIST = [
    ("DateList[{2020, 1, 32}]", "{2020, 2, 1, 0, 0, 0.}"),
    ("DateList[3155673600]", "{2000, 1, 1, 0, 0, 0.}"),
    ("DateList[{2014, 3, 1.5}]", "{2014, 3, 1, 12, 0, 0.}"),
    ("DateList[{2014, 14}]", "{2015, 2, 1, 0, 0, 0.}"),
    ('DateList["2021-10-08"]', "{2021, 10, 8, 0, 0, 0.}"),
    ('DateList[{"03/04/2021", {"Day", "Month", "Year"}}]', "{2021, 4, 3, 0, 0, 0.}"),
    ('DateList[{"10:15 PM, May 2 2010", {"Hour12", ":", "Minute", " ", "AMPM", ", ", "MonthNameShort", " ", "Day", " ", "Year"}}]',
     "{2010, 5, 2, 22, 15, 0.}"),
    ('DateList[DateObject[{2020, 3, 4, 5, 6, 7.5}, TimeZone -> 3]]', "{2020, 3, 4, 5, 6, 7.5}"),
    ('DateList[DateObject[{2020, 3}]]', "{2020, 3, 1, 0, 0, 0.}"),
    ("DateList[{2020, 1, 1}, 3]", Raises()),
    ('DateList["not a date"]', Raises()),
    (tz0("Length[DateList[]]"), "6"),
]
ABSOLUTETIME = [
    ("AbsoluteTime[{2000, 1, 1}]", "3155673600"),
    ("AbsoluteTime[{1900, 1, 1, 0, 0, 0}]", "0"),
    ("AbsoluteTime[{1970, 1, 1}]", "2208988800"),
    ("AbsoluteTime[{2000, 1, 1, 0, 0, 0.5}]", "3155673600.5"),
    ('Block[{$TimeZone = -5}, AbsoluteTime[DateObject[{2000, 1, 1, 0, 0, 0}, "Instant", "Gregorian", 0.]]]',
     "3155655600"),
    ('Block[{$TimeZone = 3}, AbsoluteTime[DateObject[{2000, 1, 1, 0, 0, 0}, "Instant", "Gregorian", 0.]]]',
     "3155684400"),
    ('AbsoluteTime["2000-01-01T00:00:00Z", TimeZone -> 1]', "3155677200"),
    ("AbsoluteTime[3155673600]", "3155673600"),
    (tz0("NumberQ[AbsoluteTime[]]"), "True"),
    ("AbsoluteTime[{2000, 1, x}]", "AbsoluteTime[{2000, 1, x}]"),
    ("AbsoluteTime[{2000.5}]", Raises()),
]
TIMEOBJECT = [
    ("TimeObject[{13, 0, 2}]", 'TimeObject[{13, 0, 2}, "Second"]'),
    ("TimeObject[{25, 61}]", 'TimeObject[{2, 1}, "Minute"]'),
    ("TimeObject[{10}]", 'TimeObject[{10}, "Hour"]'),
    ("TimeObject[14]", 'TimeObject[{14}, "Hour"]'),
    ("TimeObject[{10, 4, 5.5}]", 'TimeObject[{10, 4, 5.5}, "Instant"]'),
    ('TimeObject[{10, 4, 30}, "Minute"]', 'TimeObject[{10, 4}, "Minute"]'),
    ("TimeObject[DateObject[{2020, 1, 1, 10, 30}, TimeZone -> 0]]", 'TimeObject[{10, 30}, "Minute", TimeZone -> 0.]'),
    ('TimeObject[{10, 4, 30}]["Hour"]', "10"),
    ('TimeObject[{10, 4, 30}]["Time"]', '"10:04:30"'),
    ('TimeObject[DateObject[{2020, 1, 1}]]', Raises()),
    ('TimeObject[{1, 2, 3, 4}]', Raises()),
]
TIMEOBJECTQ = [
    T('TimeObjectQ[TimeObject[{13, 0, 2}]]'),
    F('TimeObjectQ[{13, 0, 2}]'),
    F('TimeObjectQ[DateObject[{2020}]]'),
    T('TimeObjectQ[TimeObject[{1, 2}, "Minute", TimeZone -> 0.]]'),
]
DATESTRING = [
    ('DateString[DateObject[{2020, 1, 2}]]', '"Thu 2 Jan 2020"'),
    ("DateString[{2020, 1, 2, 3, 4, 5}]", '"Thu 2 Jan 2020 03:04:05"'),
    ('DateString[{2020, 1, 2}, "ISODate"]', '"2020-01-02"'),
    ('DateString[{2019, 4, 4, 15, 58, 57}, "ISODateTime"]', '"2019-04-04T15:58:57"'),
    ('DateString[{2019, 4, 4}, "ISOWeekDate"]', '"2019-W14-4"'),
    ('DateString[{2019, 4, 4}, "ISOOrdinalDate"]', '"2019-094"'),
    ('DateString[{2019, 4, 4}, "Date"]', '"Thursday 4 April 2019"'),
    ('DateString[{2019, 4, 4, 15, 4, 0}, {"Hour12", ":", "Minute", " ", "AMPM"}]', '"03:04 PM"'),
    ('Block[{$DateStringFormat = "ISODate"}, DateString[{2019, 4, 4, 1, 2, 3}]]', '"2019-04-04"'),
    ('DateString[DateObject[{2019, 4, 4}, DateFormat -> {"Day", "/", "Month", "/", "Year"}]]', '"04/04/2019"'),
    ('DateString[{2019, 4, 4}, 7]', Raises()),
]
DATESTRINGFORMAT = [
    ('$DateStringFormat', '"DateTimeShort"'),
    ('Block[{$DateStringFormat = {"Year", ".", "Month"}}, DateString[{2001, 2, 3}]]', '"2001.02"'),
    ('$DateStringFormat = 5', Raises()),
]
TIMEZONE = [
    ('Block[{$TimeZone = -5}, $TimeZone]', "-5"),
    ('Block[{$TimeZone = -5}, $TimeZone = 2; $TimeZone]', "2"),
    ('Block[{$TimeZone = 5.5}, DateObject[{2020, 1, 1, 0, 0, 0}]]',
     'DateObject[{2020, 1, 1, 0, 0, 0}, "Instant", "Gregorian", 5.5]'),
    ('Block[{$TimeZone = "Europe/London"}, AbsoluteTime[DateObject[{2020, 7, 1, 12, 0, 0}, TimeZone -> 0]]] - '
     'AbsoluteTime[{2020, 7, 1, 12, 0, 0}]', "3600"),
    ('$TimeZone = "Mars/Olympus"', Raises()),
]

GROUPS = [v for k, v in list(globals().items()) if k.isupper() and k != "GROUPS" and isinstance(v, list)
          and v and isinstance(v[0], tuple)]
CASES = [c for g in GROUPS for c in g]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w12(case):
    check(case)
