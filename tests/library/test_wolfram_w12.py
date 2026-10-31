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

DI = 'DateInterval[{{2020, 1, 1}, {2020, 3, 1}}]'
DATEINTERVAL = [
    (DI, 'DateInterval[{{{2020, 1, 1}, {2020, 3, 1}}}, "Day"]'),
    ("DateInterval[{DateObject[{2020}], DateObject[{2021}]}]", 'DateInterval[{{{2020}, {2021}}}, "Year"]'),
    ("DateInterval[{{{2020, 1, 1}, {2020, 2, 1}}, {{2020, 1, 15}, {2020, 3, 1}}}]",
     'DateInterval[{{{2020, 1, 1}, {2020, 3, 1}}}, "Day"]'),
    (DI + '["DateBounds"]', '{DateObject[{2020, 1, 1}, "Day"], DateObject[{2020, 3, 1}, "Day"]}'),
]
DS3 = "{DateObject[{2020, 5, 1}], DateObject[{2019, 5, 1}], DateObject[{2021, 1, 1}]}"
MINDATE = [
    (f"MinDate[{DS3}]", 'DateObject[{2019, 5, 1}, "Day"]'),
    ("MinDate[{DateObject[{2020, 5, 1}], DateObject[{2020, 5, 2}]}]", 'DateObject[{2020, 5, 1}, "Day"]'),
    (f"MinDate[{DI}]", 'DateObject[{2020, 1, 1}, "Day"]'),
]
MAXDATE = [
    (f"MaxDate[{DS3}]", 'DateObject[{2021, 1, 1}, "Day"]'),
    ("MaxDate[{DateObject[{2020, 5, 1}], DateObject[{2019, 5, 1}]}]", 'DateObject[{2020, 5, 1}, "Day"]'),
    (f"MaxDate[{DI}]", 'DateObject[{2020, 3, 1}, "Day"]'),
]
DATEBOUNDS = [
    (f"DateBounds[{DS3}]", '{DateObject[{2019, 5, 1}, "Day"], DateObject[{2021, 1, 1}, "Day"]}'),
    ("DateBounds[{DateObject[{2020, 5, 1}]}]", '{DateObject[{2020, 5, 1}, "Day"], DateObject[{2020, 5, 1}, "Day"]}'),
    (f"DateBounds[{DI}]", '{DateObject[{2020, 1, 1}, "Day"], DateObject[{2020, 3, 1}, "Day"]}'),
]
DATEWITHINQ = [
    T("DateWithinQ[DateObject[{2020}], DateObject[{2020, 5, 3}]]"),
    F("DateWithinQ[DateObject[{2020}], DateObject[{2021, 5, 3}]]"),
    T(f"DateWithinQ[{DI}, DateObject[{{2020, 2, 3}}]]"),
    F(f"DateWithinQ[{DI}, DateObject[{{2020, 4, 3}}]]"),
]
DATEOVERLAPSQ = [
    T("DateOverlapsQ[DateObject[{2020}], DateObject[{2020, 5}]]"),
    F("DateOverlapsQ[DateObject[{2020}], DateObject[{2021, 5}]]"),
    T(f"DateOverlapsQ[{DI}, DateInterval[{{{{2020, 2, 1}}, {{2020, 5, 1}}}}]]"),
]
DATEDIFFERENCE = [
    ("DateDifference[{2017, 1, 1}, {2017, 7, 4}]", 'Quantity[184, "Days"]'),
    ("DateDifference[{2017, 7, 4}, {2017, 1, 1}]", 'Quantity[-184, "Days"]'),
    ('DateDifference[{2017, 1, 1}, {2017, 7, 4}, "Week"][[1]]', near(184 / 7)),
    ('DateDifference[{2017, 1, 1}, {2017, 6, 1}, "Hour"]', 'Quantity[3624, "Hours"]'),
    ('DateDifference[{2017, 1, 1}, {2017, 7, 4}, {"Year", "Month", "Day"}]',
     'Quantity[MixedMagnitude[{0, 6, 3}], MixedUnit[{"Years", "Months", "Days"}]]'),
    ('DateDifference[{2016, 1, 1}, {2017, 3, 5}, {"Month", "Day"}]',
     'Quantity[MixedMagnitude[{14, 4}], MixedUnit[{"Months", "Days"}]]'),
]
DAYCOUNTCONVENTION = [
    ('DateDifference[{2017, 1, 1}, {2017, 7, 4}, DayCountConvention -> "Thirty360"]', 'Quantity[183, "Days"]'),
    ('DateDifference[{2017, 1, 1}, {2017, 7, 4}, DayCountConvention -> "ActualActual"]', 'Quantity[184, "Days"]'),
    ('DateDifference[{2016, 1, 30}, {2017, 3, 1}, "Year", DayCountConvention -> "Thirty360"][[1]]',
     near(391 / 360)),
    ('DateDifference[{2017, 1, 1}, {2017, 7, 4}, DayCountConvention -> "Bogus"]', Raises()),
]
TIMESYSTEMCONVERT = [
    ('TimeSystemConvert[DateObject[{2021, 8, 1, 0, 0, 0}, TimeZone -> 0], "TAI"]',
     'DateObject[{2021, 8, 1, 0, 0, 37.}, "Instant", "Gregorian", 0., "TAI"]'),
    ('TimeSystemConvert[DateObject[{2021, 8, 1, 0, 0, 0}, TimeZone -> 0], "GPS"]',
     'DateObject[{2021, 8, 1, 0, 0, 18.}, "Instant", "Gregorian", 0., "GPS"]'),
    ('TimeSystemConvert[DateObject[{2021, 8, 1, 0, 0, 0}, TimeZone -> 0], "TT"]',
     'DateObject[{2021, 8, 1, 0, 1, 9.184}, "Instant", "Gregorian", 0., "TT"]'),
    ('TimeSystemConvert[DateObject[{2021, 8, 1, 0, 0, 0}, TimeZone -> 0], "Bogus"]', Raises()),
]
JULIANDATE = [
    ("JulianDate[DateObject[{2000, 1, 1, 12, 0, 0}, TimeZone -> 0]]", near(2451545.0)),
    ('JulianDate["Modified", DateObject[{2000, 1, 1, 0, 0, 0}, TimeZone -> 0]]', near(51544.0)),
    ('JulianDate["Dublin", DateObject[{2021, 2, 19, 11, 0, 0}, TimeZone -> 0]]', near(44244.958333333)),
]
FROMJULIANDATE = [
    ("FromJulianDate[2457754, TimeZone -> 0]", 'DateObject[{2016, 12, 31, 12, 0, 0}, "Instant", "Gregorian", 0.]'),
    ("FromJulianDate[0, TimeZone -> 0]", 'DateObject[{-4714, 11, 24, 12, 0, 0}, "Instant", "Gregorian", 0.]'),
    ('FromJulianDate["Modified", 0, TimeZone -> 0]', 'DateObject[{1858, 11, 17, 0, 0, 0}, "Instant", "Gregorian", 0.]'),
    ('FromJulianDate["Truncated", 0, TimeZone -> 0]',
     'DateObject[{1968, 5, 24, 0, 0, 0}, "Instant", "Gregorian", 0.]'),
]
LUNATIONNUMBER = [
    ("Block[{$TimeZone = -6}, LunationNumber[DateObject[{2020, 1, 1}]]]", near(247.19, 3e-5)),
    ("Block[{$TimeZone = -5}, LunationNumber[DateObject[{1988, 6, 23}]]]", near(-142.726, 5e-6)),
    ('Block[{$TimeZone = -5}, LunationNumber["BrownLunationNumber", DateObject[{1988, 6, 23}]]]', near(810.274, 1e-6)),
    ("LunationNumber[DateObject[{2023, 9, 28, 12, 0, 0}, TimeZone -> 0]]", near(293.465, 3e-6)),
    ('LunationNumber["NoSuchScheme", DateObject[{2020, 1, 1}]]', Raises()),
]
FROMLUNATIONNUMBER = [
    T("Abs[AbsoluteTime[FromLunationNumber[0]] - AbsoluteTime[DateObject[{2000, 1, 6, 18, 13, 38}, TimeZone -> 0]]] < 60"),
    T("Abs[AbsoluteTime[FromLunationNumber[300]] - AbsoluteTime[DateObject[{2024, 4, 8, 18, 20, 51}, TimeZone -> 0]]] "
      "< 60"),
    T('Abs[AbsoluteTime[FromLunationNumber["IslamicLunationNumber", 1]] - '
      'AbsoluteTime[DateObject[{622, 7, 17, 5, 29, 13}, TimeZone -> 0]]] < 600'),
    T("Abs[LunationNumber[FromLunationNumber[123.25]] - 123.25] < 10^-6"),
]


def _sid(secs, tol):
    return Pred(lambda g: abs(_num(g) - secs) <= tol, f"~{secs}s")


def _sidsecs(src):
    return f"Module[{{q = {src}}}, q[[1, 1]] . {{3600, 60, 1}}]"


SIDEREALTIME = [
    (_sidsecs('SiderealTime[GeoPosition[{50.3, -80.1}], DateObject[{2025, 3, 17, 15, 3, 6.49081}, "Instant", '
              '"Gregorian", -5.]]'), _sid(2 * 3600 + 25 * 60 + 17.7664, 0.1)),
    (_sidsecs('SiderealTime["MeanTime", {0, 0}, DateObject[{1987, 4, 10, 0, 0, 0}, TimeZone -> 0]]'),
     _sid(13 * 3600 + 10 * 60 + 46.3668, 0.002)),
    (_sidsecs('SiderealTime["ApparentTime", {0, 0}, DateObject[{1987, 4, 10, 0, 0, 0}, TimeZone -> 0]]'),
     _sid(13 * 3600 + 10 * 60 + 46.1351, 0.01)),
    (_sidsecs('SiderealTime["MeanTime", {10, 0}, DateObject[{2000, 1, 1, 12, 0, 0}, TimeZone -> 0]]'),
     _sid(18 * 3600 + 41 * 60 + 50.548, 0.002)),
    ('SiderealTime[{0, 0}, DateObject[{2000}]][[2]]',
     'MixedUnit[{"HoursOfRightAscension", "MinutesOfRightAscension", "SecondsOfRightAscension"}]'),
    ("SiderealTime[{100, 0}, DateObject[{2000}]]", Raises()),
]
RANDOMDATE = [
    T('DateWithinQ[DateObject[{1999}, "Year"], RandomDate[DateObject[{1999}, "Year"], TimeZone -> 0]]'),
    ('Length[RandomDate[DateObject[{1999}, "Year"], 5]]', "5"),
    ('Dimensions[RandomDate[DateObject[{1999}, "Year"], {2, 3}]]', "{2, 3}"),
    T('SeedRandom[3]; w12a = RandomDate[DateObject[{1999}, "Year"]]; SeedRandom[3]; '
      'w12a === RandomDate[DateObject[{1999}, "Year"]]'),
    T('Module[{d = RandomDate[DateObject[{1985, 8}, "Month"], DateGranularity -> "Day"]}, '
      'd[[2]] === "Day" && d[[1, 1 ;; 2]] === {1985, 8}]'),
    ("RandomDate[DateObject[{2000, 1, 1, 0, 0, 0}]]", Raises()),
]
RANDOMTIME = [
    T("TimeObjectQ[RandomTime[]]"),
    T('Module[{t = RandomTime[TimeObject[{3}, "Hour"]]}, t[[1, 1]] == 3]'),
    ('Length[RandomTime[TimeObject[{3}, "Hour"], 4]]', "4"),
    T('Module[{t = RandomTime[{TimeObject[{6}, "Hour"], TimeObject[{15}, "Hour"]}]}, 6 <= t[[1, 1]] <= 15]'),
]
DATED = [
    ('Dated["Population", 1985]', 'Dated["Population", 1985]'),
    ('Dated["Population", All]', 'Dated["Population", All]'),
    ('Dated["Population", "x"]', Raises()),
]
CLOCK = [
    ("Clock[]", "0"),
    ("Clock[{1, 3}]", "1"),
    ("Clock[{5, 7}, 2]", "5"),
    ("Clock[{1, 3}, -2]", Raises()),
]
DATESCALE = [
    ("DateScale[]", "DateScale[]"),
    ('DateScale[DateTicksFormat -> {"Year"}]', 'DateScale[DateTicksFormat -> {"Year"}]'),
    ("DateScale[Foo -> 1]", Raises()),
]
DATEPATTERN = [
    T('StringMatchQ["2021-03-04", DatePattern[{"Year", "Month", "Day"}]]'),
    F('StringMatchQ["2021-13-04", DatePattern[{"Year", "Month", "Day"}]]'),
    ('StringCases["on 12/25/2020 and 1/2/2021", DatePattern[{"Month", "Day", "Year"}]]',
     '{"12/25/2020", "1/2/2021"}'),
    T('StringMatchQ["Mar 2021", DatePattern[{"MonthName", " ", "Year"}]]'),
    T('StringMatchQ["2021 03", DatePattern[{"Year", "Month"}, " "]]'),
]
DATEFORMAT = [
    ('DateString[DateObject[{2019, 4, 4}, DateFormat -> {"Year", "-", "Month"}]]', '"2019-04"'),
    ('DateObject[{2019, 4, 4}, DateFormat -> "ISODate"][[-1]]', 'DateFormat -> "ISODate"'),
    ('DateObject[{2019, 4, 4}, DateFormat -> 17]', Raises()),
]
DATEGRANULARITY = [
    ('RandomDate[DateObject[{1985, 8}, "Month"], DateGranularity -> "Month"]', 'DateObject[{1985, 8}, "Month"]'),
    ('RandomDate[DateObject[{1985}, "Year"], DateGranularity -> "Year"]', 'DateObject[{1985}, "Year"]'),
    ('RandomDate[DateObject[{1985}, "Year"], DateGranularity -> "Fortnight"]', Raises()),
]


NS3 = '<|"CategoryList" -> {"a", "b", "c"}, "LabelList" -> {"a", "b", "c"}, "ScoreList" -> {1, 2, 3}|>'
NOMINAL = [
    ('Nominal[{"a", "b", "c"}]', f"Nominal[{NS3}]"),
    ('Nominal[{"a", "b", "c"}, "a"]', f"Nominal[{NS3}, 1]"),
    ('Nominal[{"a", "b", "c"}] /@ {"c", "b"}', f"{{Nominal[{NS3}, 3], Nominal[{NS3}, 2]}}"),
    ('Nominal[NominalScale[{"a", "b", "c"}]]["b"]', f"Nominal[{NS3}, 2]"),
    ('CategoricalValue[Sort[Nominal[{"b", "c", "a"}] /@ {"b", "c", "a"}]]', '{"a", "b", "c"}'),
    ('Head[Min[Nominal[{"b", "a"}] /@ {"a", "b"}]]', "Min"),
    ('Nominal[{"a", "b"}, "z"]', Raises()),
]
ORDINAL = [
    ('Ordinal[{"a", "b", "c"}, "b"]', f"Ordinal[{NS3}, 2]"),
    ('Ordinal[OrdinalScale[{"a", "b", "c"}], "b"]', f"Ordinal[{NS3}, 2]"),
    ('CategoricalValue[Sort[Ordinal[{"b", "c", "a"}] /@ {"a", "b", "c"}]]', '{"b", "c", "a"}'),
    ('CategoricalValue[Min[Ordinal[{"b", "c", "a"}] /@ {"a", "b", "c"}]]', '"b"'),
    T('Less @@ {Ordinal[{w12a, w12b}, w12a], Ordinal[{w12a, w12b}, w12b]}'),
    F('Greater @@ {Ordinal[{w12a, w12b}, w12a], Ordinal[{w12a, w12b}, w12b]}'),
    ('Ordinal[<|"CategoryList" -> {"x", "y"}, "ScoreList" -> {1, "q"}|>]', Raises()),
]
HT = '<|"CategoryList" -> {"Heads", "Tails"}, "LabelList" -> {"Heads", "Tails"}, "ScoreList" -> {1, 2}|>'
SUIT = ('<|"CategoryList" -> {"NS", "S3", "S2", "S1"}, "LabelList" -> {"Not Suitable", "Marginally Suitable", '
        '"Moderately Suitable", "Highly Suitable"}, "ScoreList" -> {0, 0.5, 0.75, 1}|>')
CATEGORICALVALUE = [
    (f"CategoricalValue[Nominal[{HT}, 2]]", '"Tails"'),
    (f'CategoricalValue[Nominal[{HT}, 2], "Index"]', "2"),
    (f'CategoricalValue[Ordinal[{SUIT}], "Score"]', "{0, 0.5, 0.75, 1}"),
    (f'CategoricalValue[Ordinal[{SUIT}], "Label"]',
     '{"Not Suitable", "Marginally Suitable", "Moderately Suitable", "Highly Suitable"}'),
    (f'CategoricalValue[Ordinal[{SUIT}], "Scale"]', f"Ordinal[{SUIT}]"),
    ('CategoricalValue[Nominal[{"male", "female"}, "male"], "Scale"]',
     'Nominal[<|"CategoryList" -> {"male", "female"}, "LabelList" -> {"male", "female"}, "ScoreList" -> {1, 2}|>]'),
    ('Normal[CategoricalValue[Nominal[{"male", "female"}, "male"], "Scale"]]', '{"male", "female"}'),
    ('SeedRandom[1]; Union[CategoricalValue[RandomChoice[Nominal[{"Heads", "Tails"}], 20]]]', '{"Heads", "Tails"}'),
    (f'CategoricalValue[Nominal[{HT}, 2], "Bogus"]', Raises()),
]
NOMINALSCALE = [
    ('NominalScale[{"x", "y"}]', 'NominalScale[{"x", "y"}]'),
    ('NominalScale[<|"a" -> "A", "b" -> "B"|>]["LabelList"]', '{"A", "B"}'),
    ('NominalScale[{"a", "b"}, <|"b" -> "Bee"|>]["Labels", {"a", "b"}]', '{"a", "Bee"}'),
    ('NominalScale[{"a", "b"}]["CategoryList"]', '{"a", "b"}'),
    ('NominalScale[{"a", "b"}, {"x"}]', Raises()),
]
OS5 = 'OrdinalScale[<|"A" -> 5, "B" -> 4, "C" -> 3, "D" -> 2, "F" -> 1|>, <|"F" -> "Failed"|>]'
ORDINALSCALE = [
    (OS5 + '["LabelList"]', '{"A", "B", "C", "D", "Failed"}'),
    (OS5 + '["Value", "C"]', "3"),
    (OS5 + '["Categories", {5, 1}]', '{"A", "F"}'),
    ('OrdinalScale[{"x", "y"}, {-1, 3}]["CategoryAssociation"]', '<|-1 -> "x", 3 -> "y"|>'),
    ('OrdinalScale[{"x", "y"}]["ValueList"]', "{1, 2}"),
    ('OrdinalScale[{"x", "y"}, {1, "a"}]', Raises()),
]
CAST = [
    ('Cast[2, "Real64"]', "2."),
    ('Cast[{1, 2, 3}, TypeSpecifier["PackedArray"]["Real64", 1]]', "{1., 2., 3.}"),
    ('Cast["test", TypeSpecifier["PackedArray"]["Real64", 1]]', Raises()),
    ('Cast[100, "Integer32", "CCast"]', "100"),
    ('Cast[2^32, "Integer32", "CCast"]', "0"),
    ('Cast[2^32, "Integer32"]', Raises()),
    ('Cast[-1, "UnsignedInteger8", "CCast"]', "255"),
    ('Cast[1.5, "Integer64", "BitCast"]', "4609434218613702656"),
    ('Cast[4609434218613702656, "Real64", "BitCast"]', "1.5"),
]


GROUPS = [v for k, v in list(globals().items()) if k.isupper() and k != "GROUPS" and isinstance(v, list)
          and v and isinstance(v[0], tuple)]
CASES = [c for g in GROUPS for c in g]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w12(case):
    check(case)
