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



import os as _os
import sqlite3 as _sqlite3
import tempfile as _tempfile

_DBDIR = _tempfile.mkdtemp(prefix="w12db")
DBF = _os.path.join(_DBDIR, "test.sqlite").replace("\\", "/")
_con = _sqlite3.connect(DBF)
_con.executescript("""
create table offices(officeCode text primary key, city text not null, phone text unique);
create table employees(id integer primary key, name text, officeCode text references offices(officeCode));
create index emp_name on employees(name);
insert into offices values ('1', 'San Francisco', '+1 650'), ('2', 'Boston', '+1 215');
insert into employees values (1, 'Ann', '1'), (2, 'Bob', '2'), (3, 'Cy', NULL);
""")
_con.commit()
_con.close()

# ─── Tabular data ───
TAB3 = 'Tabular[{{1, 2, 3}, {4, 5, 6}}, {"a", "b", "c"}]'
TAB2 = 'Tabular[{{1, 2}, {3, 4}, {5, 6}}, {"a", "b"}]'
TABK = ('Tabular[{{"one", True, "foo"}, {"two", False, "bar"}}, <|"ColumnKeys" -> {"key", "col1", "col2"}, '
        '"KeyColumns" -> {"key"}|>]')
TABW = ('Tabular[Association["RawSchema" -> Association["ColumnProperties" -> Association["a" -> '
        'Association["ElementType" -> "Integer64"], "b" -> Association["ElementType" -> "String"]], "KeyColumns" -> '
        'None, "Backend" -> "WolframKernel"], "Options" -> {}, "BackendData" -> Association["ColumnData" -> '
        'DataStructure["ColumnTable", {{TabularColumn[Association["Data" -> {{3, 1, 2}, {}, None}, "ElementType" -> '
        '"Integer64"]], TabularColumn[Association["Data" -> {{3, {0, 3, 5, 8}, "andorbut"}, {}, None}, '
        '"ElementType" -> "String"]]}}]]]]')
TABULAR = [
    ('Normal[Tabular[{{1, 2, 3}, {4, 5, 6}}]]', "{{1, 2, 3}, {4, 5, 6}}"),
    ('Normal[Tabular[{<|"a" -> 1, "b" -> 2|>, <|"a" -> 4, "b" -> 5|>}]]', '{<|"a" -> 1, "b" -> 2|>, <|"a" -> 4, "b" -> 5|>}'),
    (f'Normal[{TAB3}]', '{<|"a" -> 1, "b" -> 2, "c" -> 3|>, <|"a" -> 4, "b" -> 5, "c" -> 6|>}'),
    (f'Normal[{TABW}]', '{<|"a" -> 3, "b" -> "and"|>, <|"a" -> 1, "b" -> "or"|>, <|"a" -> 2, "b" -> "but"|>}'),
    ('Tabular[{{1, 2}}][[1, 2]]', "2"),
    ('Normal[Tabular[{{1, 2}, {3, 4}}][[{1}, {2}]]]', "{{2}}"),
    ('Normal[Tabular[{{1, 2}, {3, 4}}][[1]]]', "{1, 2}"),
    ('Normal[Tabular[{{1, 2}, {3, 4}}][[All, 1]]]', "{1, 3}"),
    (f'Normal[{TABK}[Key["one"]]]', '<|"key" -> "one", "col1" -> True, "col2" -> "foo"|>'),
    (f'Normal[{TABK}[<|"key" -> "two"|>]]', '<|"key" -> "two", "col1" -> False, "col2" -> "bar"|>'),
    (f'{{Length[{TAB2}], Dimensions[{TAB2}]}}', "{3, {3, 2}}"),
    (f'Normal[Select[{TAB2}, #a > 1 &]]', '{<|"a" -> 3, "b" -> 4|>, <|"a" -> 5, "b" -> 6|>}'),
    (f'Normal[Take[{TAB2}, 2]]', '{<|"a" -> 1, "b" -> 2|>, <|"a" -> 3, "b" -> 4|>}'),
    (f'Normal[SortBy[{TAB2}, -#a &]][[1]]', '<|"a" -> 5, "b" -> 6|>'),
    (f'Normal[Join[{TAB2}, {TAB2}]][[4]]', '<|"a" -> 1, "b" -> 2|>'),
    ('Tabular[{{1, 2}, {3}}]', Raises()),
    (f'Options[Tabular[{TAB3}, AllowedDimensions -> {{2, Automatic}}]]', '{AllowedDimensions -> {2, Automatic}}'),
]
TABULARQ = [
    T("TabularQ[Tabular[{{1, 2}, {3, 4}}]]"),
    F("TabularQ[{{1, 2}, {3, 4}}]"),
    T(f"TabularQ[{TABW}]"),
]
TABULARCOLUMN = [
    ('TabularColumn[{3, 2, -1, 0, 5}]["ElementType"]', '"Integer64"'),
    ('Normal[TabularColumn[{"earth", "fire", "air", "water"}]]', '{"earth", "fire", "air", "water"}'),
    ('Normal[TabularColumn[{5, 7.8, -13, 11/3}, "Integer16", <|"ConversionMethod" -> "Round"|>]]', "{5, 8, -13, 4}"),
    ('TabularColumn[{1.5, 2}]["ElementType"]', '"Real64"'),
    ('Total[TabularColumn[{1, 2, 3}]]', "6"),
    ('TabularColumn[{1, 300}, "Integer8"]', Raises()),
]
TABULARCOLUMNQ = [
    T("TabularColumnQ[TabularColumn[{1, 2, 3, 4}]]"),
    T('TabularColumnQ[TabularColumn[{"a", "b"}], "String"]'),
    F('TabularColumnQ[TabularColumn[{"a", "b"}], "Integer*"]'),
    F("TabularColumnQ[{1, 2}]"),
    T('TabularColumnQ[TabularColumn[{1, 2}], "Integers"]'),
]
TABULARROW = [
    ('Normal[TabularRow[{7, "cat", -3.1}]]', '{7, "cat", -3.1}'),
    ('ColumnTypes[TabularRow[<|"col1" -> 7, "col2" -> "cat", "col3" -> -3.1|>]]',
     '<|"col1" -> "Integer64", "col2" -> "String", "col3" -> "Real64"|>'),
    ('ColumnKeys[TabularRow[{7, "cat", -3.1}]]', "None"),
    ('Normal[Tabular[{<|"a" -> 1, "b" -> 2, "c" -> 3|>, <|"a" -> 4, "b" -> 5, "c" -> 6|>}][[1]]]',
     '<|"a" -> 1, "b" -> 2, "c" -> 3|>'),
    ('TabularRow[{1, 2}, {"x", "y"}]', 'TabularRow[<|"x" -> 1, "y" -> 2|>]'),
    ('TabularRow[<|"x" -> 1, "y" -> 2|>]["y"]', "2"),
]
TABULARROWQ = [
    T("TabularRowQ[TabularRow[{1, 2, 3, 4}]]"),
    T('TabularRowQ[TabularRow[<|"a" -> 1|>]]'),
    F("TabularRowQ[{1, 2}]"),
]
COLUMNKEYS = [
    ('ColumnKeys[Tabular[{<|"a" -> 1, "b" -> 2, "c" -> 3|>, <|"a" -> 4, "b" -> 5, "c" -> 6|>}]]', '{"a", "b", "c"}'),
    ('ColumnKeys[Tabular[{{1, 2, 3}, {4, 5, 6}}, {"ab", "bb", "ac"}], "a*"]', '{"ab", "ac"}'),
    ('ColumnKeys[Tabular[{{1, 2, 3}, {4, 5, 6}}, {"ab", "bb", "ac"}], "a" ~~ __]', '{"ab", "ac"}'),
    ('ColumnKeys[Tabular[{{1, 2, 3}, {4, 5, 6}}, {"ab", "bb", "ac"}], "ab" | "ac"]', '{"ab", "ac"}'),
    ('ColumnKeys[Tabular[{{"cat", 1, "foo", 3.5}, {"dog", 3, "bar", -1.1}}, {"name1", "num1", "name2", "num2"}], '
     'StringStartsQ[#ColumnKey, "num"] &]', '{"num1", "num2"}'),
    ('ColumnKeys[Tabular[{{"cat", 1, "foo", 3.5}, {"dog", 3, "bar", -1.1}}, {"name1", "num1", "name2", "num2"}], '
     '#ElementType == "String" &]', '{"name1", "name2"}'),
    ('ColumnKeys[Tabular[{{1, 2}}]]', "None"),
    ('ColumnKeys[{<|"x" -> 1|>, <|"y" -> 2|>}]', '{"x", "y"}'),
]
COLUMNKEYEXISTSQ = [
    T(f'ColumnKeyExistsQ[{TAB3}, "b"]'),
    F(f'ColumnKeyExistsQ[{TAB3}, "d"]'),
    T(f'ColumnKeyExistsQ["c"][{TAB3}]'),
    T('ColumnKeyExistsQ[Dataset[{<|"u" -> 1|>}], "u"]'),
]
COLUMNTYPES = [
    ('ColumnTypes[Tabular[{{1, "dog", True}, {2, "cat", False}}, {"col1", "col2", "col3"}]]',
     '<|"col1" -> "Integer64", "col2" -> "String", "col3" -> "Boolean"|>'),
    ('ColumnTypes[Tabular[{{3.1, 4, False}, {2.5, -3, True}}, {"number", "symbol", "boolean"}], "Real*"]',
     '<|"number" -> "Real64"|>'),
    ('ColumnTypes[Tabular[{{1, 2.}, {3, 4.}, {5, 6.}}]]', '{"Integer64", "Real64"}'),
    ('ColumnTypes[Tabular[{{1, 2}, {3, 4}, {5, 6}}, <|"ElementType" -> {"Integer64", "Integer16"}|>]]',
     '{"Integer64", "Integer16"}'),
    ('ColumnTypes[Tabular[{{1, "dog", True}, {2, "cow", False}}, <|"ElementType" -> {"Integer8", "String", '
     '"Boolean"}|>]]', '{"Integer8", "String", "Boolean"}'),
]
CASTCOLUMNS = [
    ('ColumnTypes[CastColumns[ToTabular[<|"a" -> {1, 2, 3}, "b" -> {1., 4., 8.}|>, "Columns"], "a" -> "Real64"]]',
     '<|"a" -> "Real64", "b" -> "Real64"|>'),
    ('Normal[CastColumns[Tabular[{{"1"}, {"22"}}, {"s"}], "s" -> "Integer64"]]', '{<|"s" -> 1|>, <|"s" -> 22|>}'),
    ('Normal[CastColumns[Tabular[{{1}, {2}}, {"n"}], {"n" -> "String"}]]', '{<|"n" -> "1"|>, <|"n" -> "2"|>}'),
    ('CastColumns[Tabular[{{"x"}}, {"s"}], "s" -> "Integer64"]', Raises()),
]
DELETECOLUMNS = [
    (f'Normal[DeleteColumns[{TAB3}, "a"]]', '{<|"b" -> 2, "c" -> 3|>, <|"b" -> 5, "c" -> 6|>}'),
    (f'Normal[DeleteColumns[{TAB3}, {{"a", "c"}}]]', '{<|"b" -> 2|>, <|"b" -> 5|>}'),
    (f'Normal[DeleteColumns["b"][{TAB3}]]', '{<|"a" -> 1, "c" -> 3|>, <|"a" -> 4, "c" -> 6|>}'),
    ('DeleteColumns[{<|"a" -> 1, "b" -> 2|>}, "b"]', '{<|"a" -> 1|>}'),
    (f'DeleteColumns[{TAB3}, "zz"]', Raises()),
]
INSERTCOLUMNS = [
    (f'Normal[InsertColumns[{TAB2}, "aa" -> {{11, 22, 33}}]][[1]]', '<|"a" -> 1, "b" -> 2, "aa" -> 11|>'),
    (f'Normal[InsertColumns[{TAB2}, {{"a", "aa" -> {{11, 22, 33}}}}]][[2]]', '<|"a" -> 3, "aa" -> 22, "b" -> 4|>'),
    (f'Normal[InsertColumns[{{"aa" -> {{11, 22, 33}}, "bb" -> 0}}][{TAB2}]][[3]]',
     '<|"a" -> 5, "b" -> 6, "aa" -> 33, "bb" -> 0|>'),
    (f'InsertColumns[{TAB2}, "aa" -> {{1, 2}}]', Raises()),
]
RENAMECOLUMNS = [
    (f'ColumnKeys[RenameColumns[{TAB3}, {{"a" -> "first", "c" -> "last"}}]]', '{"first", "b", "last"}'),
    (f'ColumnKeys[RenameColumns[{TAB3}, {{1 -> "first", -1 -> "last"}}]]', '{"first", "b", "last"}'),
    ('ColumnKeys[RenameColumns[Tabular[{{"a", 1, True}, {"d", 2, False}}, {"col1", "col2", "col3"}], '
     '{"first", "second"}]]', '{"first", "second", "col3"}'),
    ('ColumnKeys[RenameColumns[{1 -> "first", "col3" -> "third"}][Tabular[{{"a", 1, True}}, {"col1", "col2", '
     '"col3"}]]]', '{"first", "col2", "third"}'),
]
TABAB = 'Tabular[{{2, 2.78}, {1, 3.14}, {3, 1.68}}, {"a", "b"}]'
TRANSFORMCOLUMNS = [
    (f'Normal[TransformColumns[{TABAB}, "c" -> Function[#b - #a]]][[2, "c"]]', near(2.14)),
    (f'Normal[TransformColumns[{TABAB}, {{"a" -> (2 #a &), "b" -> (#b/2 &)}}]][[1]]', '<|"a" -> 4, "b" -> 1.39|>'),
    ('Normal[TransformColumns[Tabular[{{1}, {2}, {3}, {4}}, {"col"}], "acc" -> Function[ColumnwiseThread['
     'Accumulate[#col]]]]][[All, "acc"]]', "{1, 3, 6, 10}"),
    ('Normal[TransformColumns[Tabular[{{2004, 4, 1}}, {"a", "b", "c"}], "date" -> Function[DateObject[{#a, #b, #c}, '
     '"Day"]]]][[1, "date"]]', 'DateObject[{2004, 4, 1}, "Day"]'),
    (f'TransformColumns[{TABAB}, "c"]', Raises()),
]
CONSTRUCTCOLUMNS = [
    (f'Normal[ConstructColumns[{TABAB}, "c" -> Function[#a - #b]]][[3]]', '<|"c" -> 1.32|>'),
    (f'Keys[Normal[ConstructColumns[{TABAB}, {{"a", "c" -> (#a - #b &)}}]][[1]]]', '{"a", "c"}'),
    (f'Keys[Normal[ConstructColumns[{TABAB}, {{"c" -> (#a - #b &), "a"}}]][[1]]]', '{"c", "a"}'),
    (f'Normal[ConstructColumns[{TAB2}, "b"]]', '{<|"b" -> 2|>, <|"b" -> 4|>, <|"b" -> 6|>}'),
]
COLUMNWISETHREAD = [
    ('Normal[TransformColumns[Tabular[{{1}, {2}, {3}}, {"col"}], "new" -> Function[#col - ColumnwiseThread['
     'Reverse[#col]]]]][[All, "new"]]', "{-2, 0, 2}"),
    ('Normal[ConstructColumns[Tabular[{{1}, {2}, {3}}, {"x"}], "r" -> Function[ColumnwiseThread[Ranking[#x]] + 0]]]',
     Pred(lambda g: True, "any")) if False else
    ('Normal[ConstructColumns[Tabular[{{1}, {2}, {3}}, {"x"}], "r" -> Function[ColumnwiseThread[Reverse[#x]]]]]',
     '{<|"r" -> 3|>, <|"r" -> 2|>, <|"r" -> 1|>}'),
    ('ColumnwiseThread[w12q]', "ColumnwiseThread[w12q]"),
]
COLUMNWISEVALUE = [
    ('Normal[TransformColumns[Tabular[{{1}, {2}, {3}}, {"x"}], "d" -> Function[#x - ColumnwiseValue[Mean[#x]]]]]'
     '[[All, "d"]]', "{-1, 0, 1}"),
    ('Normal[TransformColumns[Tabular[{{1}, {3}}, {"x"}], "f" -> Function[#x / ColumnwiseValue[Total[#x]]]]]'
     '[[All, "f"]]', "{1/4, 3/4}"),
    ('ColumnwiseValue[w12q]', "ColumnwiseValue[w12q]"),
]
TABAGG = 'Tabular[{{"a", 1}, {"b", 2}, {"a", 3}, {"b", 4}}, {"col1", "col2"}]'
AGGREGATEROWS = [
    (f'Normal[AggregateRows[{TABAGG}, {{"j" -> (StringJoin[#col1] &), "t" -> (Total[#col2] &)}}]]',
     '{<|"j" -> "abab", "t" -> 10|>}'),
    (f'Normal[AggregateRows[{TABAGG}, "total" -> Function[Total[#col2]], "col1"]]',
     '{<|"col1" -> "a", "total" -> 4|>, <|"col1" -> "b", "total" -> 6|>}'),
    ('AggregateRows["c" -> Function[w12f[#a]]][{<|"a" -> w12x, "b" -> 1|>, <|"a" -> w12y, "b" -> 2|>, '
     '<|"a" -> w12x, "b" -> 1|>}]', '{<|"c" -> w12f[{w12x, w12y, w12x}]|>}'),
    ('AggregateRows["c" -> Function[w12f[#a]], "b"][{<|"a" -> w12x, "b" -> 1|>, <|"a" -> w12y, "b" -> 2|>, '
     '<|"a" -> w12x, "b" -> 1|>}]', '{<|"b" -> 1, "c" -> w12f[{w12x, w12x}]|>, <|"b" -> 2, "c" -> w12f[{w12y}]|>}'),
    (f'TabularSchema[AggregateRows[{TABAGG}, "total" -> Function[Total[#col2]], "col1"]]["KeyColumns"]', '{"col1"}'),
]
TEMPS = ('Tabular[{{"high", "M", 72}, {"high", "Tu", 81}, {"low", "M", 49}, {"low", "Tu", 55}}, {"extreme", "day", '
         '"temp"}]')
WIDE = 'Tabular[{{"high", 72, 81}, {"low", 49, 55}}, {"extreme", "M", "Tu"}]'
PIVOTTOCOLUMNS = [
    (f'ColumnKeys[PivotToColumns[{TEMPS}, "day" -> "temp"]]',
     '{"extreme", ExtendedKey["temp", "M"], ExtendedKey["temp", "Tu"]}'),
    (f'Normal[PivotToColumns[{TEMPS}, "day" -> "temp"]][[2]]',
     '<|"extreme" -> "low", ExtendedKey["temp", "M"] -> 49, ExtendedKey["temp", "Tu"] -> 55|>'),
    (f'Normal[PivotToColumns[{TEMPS}, "extreme" -> "temp"]][[1]]',
     '<|"day" -> "M", ExtendedKey["temp", "high"] -> 72, ExtendedKey["temp", "low"] -> 49|>'),
    (f'PivotToColumns[{TEMPS}, "nope" -> "temp"]', Raises()),
]
PIVOTFROMCOLUMNS = [
    (f'Normal[PivotFromColumns[{WIDE}, {{"M", "Tu"}} -> {{"day", "temp"}}]]',
     '{<|"extreme" -> "high", "day" -> "M", "temp" -> 72|>, <|"extreme" -> "high", "day" -> "Tu", "temp" -> 81|>, '
     '<|"extreme" -> "low", "day" -> "M", "temp" -> 49|>, <|"extreme" -> "low", "day" -> "Tu", "temp" -> 55|>}'),
    (f'Length[PivotFromColumns[{WIDE}, {{"M"}} -> {{"day", "temp"}}]]', "2"),
    (f'PivotFromColumns[{WIDE}, {{"M", "Tu"}} -> "day"]', Raises()),
]
SALES = ('Tabular[{{"N", "x", 1}, {"N", "y", 2}, {"S", "x", 3}, {"N", "x", 4}}, {"region", "product", "units"}]')
PIVOTTABLE = [
    (f'Normal[PivotTable[{SALES}, Total[#units] &, "region", "product"]]',
     '{<|"region" -> "N", "x" -> 5, "y" -> 2|>, <|"region" -> "S", "x" -> 3, "y" -> Missing["NotAvailable"]|>}'),
    (f'ColumnKeys[PivotTable[{SALES}, {{"n" -> (Length[#units] &)}}, "region", "product"]]',
     '{"region", ExtendedKey["n", "x"], ExtendedKey["n", "y"]}'),
    (f'TabularSchema[PivotTable[{SALES}, Total[#units] &, "region", "product"]]["KeyColumns"]', '{"region"}'),
    (f'PivotTable[{SALES}, Total[#units] &, "region", "product", IncludeGroupAggregates -> True]', Raises()),
]
COLUMNWISECOMBINE = [
    ('Normal[ColumnwiseCombine[{Tabular[{{3, "and"}, {1, "or"}, {2, "but"}}, {"a", "b"}], Tabular[{{0, 8}, {1, 9}, '
     '{1, 10}, {2, 11}}, {"a", "c"}]} -> "a"]]',
     '{<|"a" -> 1, "b" -> "or", "c" -> 9|>, <|"a" -> 1, "b" -> "or", "c" -> 10|>, <|"a" -> 2, "b" -> "but", '
     '"c" -> 11|>}'),
    ('Length[ColumnwiseCombine[{Tabular[{{3, "and"}, {1, "or"}, {2, "but"}}, {"a", "b"}], Tabular[{{0, 8}, {1, 9}, '
     '{1, 10}, {2, 11}}, {"a", "c"}]} -> "a", "Outer"]]', "5"),
    ('ColumnKeys[ColumnwiseCombine[<|"x" -> Tabular[{{1, "or"}}, {"a", "b"}], "y" -> Tabular[{{1, "but"}}, {"a", '
     '"b"}]|> -> "a"]]', '{"a", ExtendedKey["x", "b"], ExtendedKey["y", "b"]}'),
    ('Length[ColumnwiseCombine[<|"x" -> Tabular[{{1}, {2}}, {"a"}], "y" -> Tabular[{{3}, {4}, {5}}, {"a"}]|> -> '
     '(True &)]]', "6"),
    ('ColumnwiseCombine[{Tabular[{{1, 2}}, {"a", "b"}], Tabular[{{1, 3}}, {"a", "b"}]} -> "a"]', Raises()),
]
TOTABULAR = [
    ('Normal[ToTabular[{{1, 2}, {3, 4}, {5, 6}}]]', "{{1, 2}, {3, 4}, {5, 6}}"),
    ('Normal[ToTabular[{{1, 2}, {3, 4}, {5, 6}}, "Columns"]]', "{{1, 3, 5}, {2, 4, 6}}"),
    ('Normal[ToTabular[{{1, 2}, {3, 4}}, "Rows", <|"ColumnKeys" -> {"a", "b"}|>]]',
     '{<|"a" -> 1, "b" -> 2|>, <|"a" -> 3, "b" -> 4|>}'),
    ('Normal[ToTabular[{"a" -> {1, 4}, "b" -> {2, 5}}, "Columns"]]', '{<|"a" -> 1, "b" -> 2|>, <|"a" -> 4, "b" -> 5|>}'),
    ('ToTabular[{1, 2}, "TimeSeries"]', Raises()),
]
FROMTABULAR = [
    ('FromTabular[Tabular[{{1, 2, 3}, {2, 4, 6}}, {"a", "b", "c"}], "Rows"]',
     '{<|"a" -> 1, "b" -> 2, "c" -> 3|>, <|"a" -> 2, "b" -> 4, "c" -> 6|>}'),
    ('Normal /@ FromTabular[Tabular[{{1, 2, 3}, {2, 4, 6}}, {"a", "b", "c"}], "Columns"]',
     '<|"a" -> {1, 2}, "b" -> {2, 4}, "c" -> {3, 6}|>'),
    ('FromTabular[Tabular[{{1, 2}, {3, 4}}, {"a", "b"}], "Dataset"]', 'Dataset[{<|"a" -> 1, "b" -> 2|>, <|"a" -> 3, '
     '"b" -> 4|>}]'),
    ('FromTabular[Tabular[{{1, 2}, {3, 4}}, {"a", "b"}], "Matrix"]', "{{1, 2}, {3, 4}}"),
]
SCH = ('TabularSchema[Association["ColumnProperties" -> Association["a" -> Association["ElementType" -> "Integer64"], '
       '"b" -> Association["ElementType" -> "Real64"], "c" -> Association["ElementType" -> "String"]], '
       '"KeyColumns" -> None, "Backend" -> "WolframKernel"]]')
TABULARSCHEMA = [
    (f'{SCH}["ColumnKeys"]', '{"a", "b", "c"}'),
    (f'{SCH}["a", "ElementType"]', '"Integer64"'),
    (f'ColumnKeys[TabularSchema[{SCH}, "ColumnKeys" -> {{"foo", "bar", "baz"}}]]', '{"foo", "bar", "baz"}'),
    (f'ColumnTypes[TabularSchema[{SCH}, "ElementType" -> "a" -> "Real"]]',
     '<|"a" -> "Real64", "b" -> "Real64", "c" -> "String"|>'),
    (f'TabularSchema[{TAB2}]["ColumnTypes"]', '<|"a" -> "Integer64", "b" -> "Integer64"|>'),
    (f'TabularSchema[{SCH}, "Bogus" -> 1]', Raises()),
]
TS4 = ('Tabular[{{1.5, "foo", True, Quantity[2, "Seconds"]}, {Missing[], "bar", False, Missing[]}}, '
       '{"c1", "c2", "c3", "c4"}]')
TABULARSTRUCTURE = [
    (f'Normal[TabularStructure[{TS4}, {{"c1", "c3"}}, {{"NonMissingCount", "MissingCount"}}]]',
     '{<|"ColumnKey" -> "c1", "NonMissingCount" -> 1, "MissingCount" -> 1|>, <|"ColumnKey" -> "c3", '
     '"NonMissingCount" -> 2, "MissingCount" -> 0|>}'),
    (f'Normal[TabularStructure[{TS4}]][[All, "ColumnType"]][[1 ;; 3]]', '{"Real64", "String", "Boolean"}'),
    (f'ColumnKeys[TabularStructure[{TS4}]]', '{"ColumnKey", "ColumnType", "NonMissingCount", "MissingCount", '
     '"ByteCount"}'),
    (f'TabularStructure[{TS4}, All, "Bogus"]', Raises()),
]
TABULARSUMMARY = [
    ('Normal[TabularSummary[Tabular[{{1, "a"}, {2, "b"}, {4, "c"}}, {"x", "y"}]]][[1]]',
     '<|"ColumnKey" -> "x", ExtendedKey["Statistical", "Min"] -> 1, ExtendedKey["Statistical", "Median"] -> 2, '
     'ExtendedKey["Statistical", "Max"] -> 4|>'),
    ('Normal[TabularSummary[Tabular[{{1, "a"}, {2, "b"}, {4, "c"}}, {"x", "y"}], "x" -> "Mean"]]',
     '{<|"ColumnKey" -> "x", ExtendedKey["Statistical", "Mean"] -> 7/3|>}'),
    ('Normal[TabularSummary[Tabular[{{1, "a"}, {2, "b"}, {4, "c"}}, {"x", "y"}], "y" -> "CountDistinct"]]',
     '{<|"ColumnKey" -> "y", ExtendedKey["Structural", "CountDistinct"] -> 3|>}'),
    ('TabularSummary[Tabular[{{1}}, {"x"}], "x" -> "Histogram"]', Raises()),
]
ROWKEY = [
    ('Normal[Tabular[Tabular[{{1, "a", 2.}, {4, "b", 3.}}, {"col1", "col2", "col3"}], <|"KeyColumns" -> {"col2"}|>]'
     '[RowKey[{"b"}]]]', '<|"col1" -> 4, "col2" -> "b", "col3" -> 3.|>'),
    ('Normal[RowKey["a"][Tabular[Tabular[{{1, "a", 2.}, {4, "b", 3.}}, {"col1", "col2", "col3"}], '
     '<|"KeyColumns" -> {"col2"}|>]]]', '<|"col1" -> 1, "col2" -> "a", "col3" -> 2.|>'),
    ('RowKey["k"]', 'RowKey[{"k"}]'),
    (f'Normal[{TABK.replace("{\"key\"}", "{\"key\", \"col1\"}")}[RowKey[{{"one", True}}]]]',
     '<|"key" -> "one", "col1" -> True, "col2" -> "foo"|>'),
]
EXTKEYS = ('Tabular[{{1, "fox", 2, "lynx"}, {3, "deer", 4, "fox"}}, {ExtendedKey["obs A", "time"], '
           'ExtendedKey["obs A", "species"], ExtendedKey["obs B", "time"], ExtendedKey["obs B", "species"]}]')
EXTENDEDKEY = [
    (f'ColumnKeys[{EXTKEYS}][[2]]', 'ExtendedKey["obs A", "species"]'),
    (f'ColumnKeys[{EXTKEYS}[All, ExtendedKey["obs A", _]]]',
     '{ExtendedKey["obs A", "time"], ExtendedKey["obs A", "species"]}'),
    ('ExtendedKey["a", "b"]', 'ExtendedKey["a", "b"]'),
    ("ExtendedKey[]", Raises()),
]
AUTH1 = '{<|"Author" -> "Mark Twain", "YearOfBirth" -> 1835|>, <|"Author" -> "Jules Verne", "YearOfBirth" -> 1828|>}'
AUTH2 = ('{<|"Author" -> "Mark Twain", "Title" -> "Tom Sawyer"|>, <|"Author" -> "Mark Twain", "Title" -> '
         '"Huckleberry Finn"|>, <|"Author" -> "Emilio Salgari", "Title" -> "Sandokan"|>}')
JOINACROSS = [
    (f'JoinAcross[{AUTH1}, {AUTH2}, "Author"]',
     '{<|"Author" -> "Mark Twain", "YearOfBirth" -> 1835, "Title" -> "Tom Sawyer"|>, <|"Author" -> "Mark Twain", '
     '"YearOfBirth" -> 1835, "Title" -> "Huckleberry Finn"|>}'),
    (f'Length[JoinAcross[{AUTH1}, {AUTH2}, "Author", "Outer"]]', "4"),
    (f'JoinAcross[{AUTH1}, {AUTH2}, "Author", "Right"][[-1]]',
     '<|"Author" -> "Emilio Salgari", "YearOfBirth" -> Missing["Unmatched"], "Title" -> "Sandokan"|>'),
    (f'Normal[JoinAcross[Tabular[{AUTH1}], Tabular[{AUTH2}], "Author"]][[2, "Title"]]', '"Huckleberry Finn"'),
    ('JoinAcross[{<|w12a -> 1, w12b -> 2|>}, {<|w12a -> 2, w12b -> 1|>}, {Key[w12a] -> Key[w12b], Key[w12b] -> '
     'Key[w12a]}, "Left"]', '{<|w12a -> 1, w12b -> 2|>}'),
    ('JoinAcross[{<|w12a -> 1, w12b -> 2|>}, {<|w12a -> 2, w12b -> 1|>}, {Key[w12a] -> Key[w12b], Key[w12b] -> '
     'Key[w12a]}, "Left", KeyCollisionFunction -> Right]', '{<|w12a -> 2, w12b -> 1|>}'),
    ('JoinAcross[{<|w12a -> 1, w12b -> 2|>}, {<|w12a -> 2, w12b -> 1|>}, {Key[w12a] -> Key[w12b], Key[w12b] -> '
     'Key[w12a]}, "Left", KeyCollisionFunction -> Function[x, {w12l[x], w12r[x]}]]',
     '{<|w12l[w12a] -> 1, w12r[w12a] -> 2, w12l[w12b] -> 2, w12r[w12b] -> 1|>}'),
    ('JoinAcross[{<|"Name" -> "Twain", "Y" -> 1835|>}, {<|"Author" -> "Twain", "Name" -> "Tom"|>}, "Name" -> '
     '"Author", KeyCollisionFunction -> None]', '{<|"Y" -> 1835, "Author" -> "Twain"|>}'),
    (f'JoinAcross[{AUTH1}, {AUTH2}, "Author", "Sideways"]', Raises()),
]
LEFTJOINACROSS = [
    (f'LeftJoinAcross[{AUTH1}, {AUTH2}, "Author"][[3]]',
     '<|"Author" -> "Jules Verne", "YearOfBirth" -> 1828, "Title" -> Missing["Unmatched"]|>'),
    (f'Length[LeftJoinAcross[{AUTH1}, {AUTH2}, "Author"]]', "3"),
    (f'LeftJoinAcross[{AUTH2}, "Author"][{AUTH1}] === LeftJoinAcross[{AUTH1}, {AUTH2}, "Author"]', "True"),
    (f'Normal[LeftJoinAcross[Tabular[{AUTH1}], Tabular[{AUTH2}], "Author"]][[3, "Title"]]', 'Missing["Unmatched"]'),
]

# ─── Dataset and Query ───
QD = ('{<|"a" -> 1, "b" -> "x", "c" -> {1}|>, <|"a" -> 2, "b" -> "y", "c" -> {2, 3}|>, <|"a" -> 3, "b" -> "z", '
      '"c" -> {3}|>, <|"a" -> 4, "b" -> "x", "c" -> {4, 5}|>, <|"a" -> 5, "b" -> "y", "c" -> {5, 6, 7}|>, '
      '<|"a" -> 6, "b" -> "z", "c" -> {}|>}')
QUERY = [
    (f'Query[All, "c", 1][{QD}]', '{1, 2, 3, 4, 5, Missing["PartAbsent", 1]}'),
    (f'Query[GroupBy["b"], Catenate, "c"][{QD}]', '<|"x" -> {1, 4, 5}, "y" -> {2, 3, 5, 6, 7}, "z" -> {3}|>'),
    (f'Query[All, "c" /* <|"ctotal" -> Total, "clength" -> Length|>][{QD}][[2]]', '<|"ctotal" -> 5, "clength" -> 2|>'),
    (f'Query[SelectFirst[#a > 5 &], "b"][{QD}]', '"z"'),
    (f'Query[Total, "a"][{QD}]', "21"),
    (f'Query[Select[#a < 5 &], "b"][{QD}]', '{"x", "y", "z", "x"}'),
    (f'Query[MaximalBy["a"], "b"][{QD}]', '{"z"}'),
    (f'Query[{{1, 2}}, "a"][{QD}]', "{1, 2}"),
    (f'Query[2, {{"a", "b"}}][{QD}]', '<|"a" -> 2, "b" -> "y"|>'),
    (f'Query[All, {{"a" -> (10 # &)}}][{QD}][[1, "a"]]', "10"),
    (f'Query[{{Total, Length}}, "a"][{QD}]', "{21, 6}"),
    ('Query[]["w12d"]', '"w12d"'),
    (f'Query[Counts, "b"][{QD}]', '<|"x" -> 2, "y" -> 2, "z" -> 2|>'),
    (f'Query[All, "nope"][{QD}][[1]]', 'Missing["KeyAbsent", "nope"]'),
    (f'Query[Keys][<|"p" -> 1, "q" -> 2|>]', '{"p", "q"}'),
    (f'Query[SortBy[#a &] /* Reverse, "a"][{QD}]', "{6, 5, 4, 3, 2, 1}"),
    (f'Query[All, "a", FailureAction -> "Bogus"][{QD}]', Raises()),
]
FAILUREACTION = [
    ('Query[All, 1/# &, FailureAction -> None][Range[-2, 2]]', "{-1/2, -1, ComplexInfinity, 1, 1/2}"),
    ('Query[All, 1/# &, FailureAction -> "Replace"][Range[-2, 2]]', '{-1/2, -1, Missing["Failed"], 1, 1/2}'),
    ('Query[All, 1/# &, FailureAction -> "Drop"][Range[-2, 2]]', "{-1/2, -1, 1, 1/2}"),
    ('Head[Query[All, 1/# &, FailureAction -> "Abort"][Range[-2, 2]]]', "Failure"),
    ('Head[Query[All, 1/# &][Range[-2, 2]]]', "Failure"),
    ('Head /@ Query[All, 1/# &, FailureAction -> "Encapsulate"][Range[-2, 2]]',
     "{Rational, Integer, Failure, Integer, Rational}"),
    ('Query[All, 1/# &, FailureAction -> {"Replace", w12rep}][Range[-1, 1]][[{1, 3}]]', "{-1, 1}"),
    ('Head[Query[All, 1/# &, FailureAction -> {"Replace", w12rep}][Range[-1, 1]][[2]]]', "w12rep"),
]
MISSINGBEHAVIOR = [
    ('Query[Total]@{1, 2, 3, Missing[]}', "6"),
    ('Query[Total, MissingBehavior -> None]@{1, 2, 3, Missing[]}', "6 + Missing[]"),
    ('Query[Mean][{2, 4, Missing["reason"]}]', "3"),
    ('Query[Mean][{}]', 'Missing["Indeterminate"]'),
    ('Query[StandardDeviation][{3}]', 'Missing["Indeterminate"]'),
    ('Query[Apply[Plus]][{2, 4, Missing["reason"]}]', "6"),
    ('Query[Apply[Times]][{2, 4, Missing["reason"]}]', 'Missing["Indeterminate"]'),
    ('Head[Query[Apply[Times], MissingBehavior -> None][{2, 4, Missing["reason"]}]]', "Times"),
    ('Query[GroupBy[Head]][{2, Missing[], 4, Missing["reason"]}]',
     '<|Integer -> {2, 4}, Missing -> {Missing[], Missing["reason"]}|>'),
    ('Query[GroupBy[Head], MissingBehavior -> None][{2, Missing[], 4, Missing["reason"]}]',
     '<|Integer -> {2, 4}, Missing -> {Missing[], Missing["reason"]}|>'),
    ('Dataset[{2, 4, Missing["reason"]}][Mean]', "3"),
    ('Query[Total, MissingBehavior -> Sometimes][{1}]', Raises()),
]
PARTBEHAVIOR = [
    ('Query[All, 2][{{1, 2}, {3}}]', '{2, Missing["PartAbsent", 2]}'),
    ('Query[All, 2, PartBehavior -> None, FailureAction -> "Replace"][{{1, 2}, {3}}]', '{2, Missing["Failed"]}'),
    ('Query["k", PartBehavior -> Automatic][<|"j" -> 1|>]', 'Missing["KeyAbsent", "k"]'),
]
DS2 = 'Dataset[{<|"a" -> 1, "b" -> 2|>, <|"a" -> 3, "b" -> 4|>}]'
DATASET = [
    (f'{DS2}[All, "a"]', "Dataset[{1, 3}]"),
    (f'{DS2}[Total, "a"]', "4"),
    (f'Normal[{DS2}]', '{<|"a" -> 1, "b" -> 2|>, <|"a" -> 3, "b" -> 4|>}'),
    (f'Length[{DS2}]', "2"),
    (f'{DS2}[[1]]', 'Dataset[<|"a" -> 1, "b" -> 2|>]'),
    (f'{DS2}[2, "b"]', "4"),
    (f'{DS2}[Select[#a > 1 &]]', 'Dataset[{<|"a" -> 3, "b" -> 4|>}]'),
    ('Dataset[Tabular[{{1, 2}}, {"p", "q"}]]', 'Dataset[{<|"p" -> 1, "q" -> 2|>}]'),
    ('Query[All, "a"][Dataset[{<|"a" -> 5|>}]]', "Dataset[{5}]"),
    ('Dataset[{1}, Bogus -> 1]', Raises()),
]
DATASETTHEME = [
    ('Dataset[{1, 2}, DatasetTheme -> "Web"]', 'Dataset[{1, 2}, DatasetTheme -> "Web"]'),
    ('Normal[Dataset[{1, 2}, DatasetTheme -> {"Serif", "LineGrid"}]]', "{1, 2}"),
    ('Dataset[{3, 4}, DatasetTheme -> "Web"][Total]', "7"),
]
INCLUDERELATEDTABLES = [
    (f'RelationalDatabase[{{"employees"}}, "{DBF}"]["Tables"]', '{"employees"}'),
    (f'RelationalDatabase[{{"employees"}}, "{DBF}", IncludeRelatedTables -> True]["Tables"]',
     '{"employees", "offices"}'),
    (f'RelationalDatabase[{{"employees"}}, "{DBF}", IncludeRelatedTables -> 3]', Raises()),
]
DATABASEREFERENCE = [
    (f'DatabaseReference["{DBF}"]', f'DatabaseReference[<|"Backend" -> "SQLite", "Name" -> "{DBF}"|>]'),
    ('DatabaseReference["postgres://user:pass@localhost:1000/database"]',
     'DatabaseReference[<|"Backend" -> "PostgreSQL", "Name" -> "database", "Host" -> "localhost", "Port" -> 1000, '
     '"Username" -> "user", "Password" -> "pass"|>]'),
    ('DatabaseReference[<|"Backend" -> "postgres", "Host" -> "localhost", "Port" -> 1000, "Name" -> "database"|>]'
     '["Backend"]', '"PostgreSQL"'),
    (f'DatabaseReference[File["{DBF}"]]["Name"]', f'"{DBF}"'),
    ('DatabaseReference[<|"Backend" -> "Bogus"|>]', Raises()),
]
DATABASECONNECT = [
    (f'Module[{{r = DatabaseReference["{DBF}"]}}, {{Head[DatabaseConnect[r]], r["Connected"], '
     f'Head[DatabaseDisconnect[r]]}}]', "{Success, True, Success}"),
    ('Head[DatabaseConnect[DatabaseReference["/nonexistent/w12.db"]]]', "Failure"),
    ('Head[DatabaseConnect[DatabaseReference["postgres://u:p@localhost:1/db"]]]', "Failure"),
]
DATABASEDISCONNECT = [
    (f'Module[{{r = DatabaseReference["{DBF}"]}}, DatabaseConnect[r]; DatabaseDisconnect[r]; r["Connected"]]', "False"),
    (f'Head[DatabaseDisconnect[DatabaseReference["{DBF}"]]]', "Failure"),
    (f'Module[{{r = RelationalDatabase["{DBF}"]}}, DatabaseConnect[r]; Head[DatabaseDisconnect[r]]]', "Success"),
]
RELATIONALDATABASE = [
    (f'RelationalDatabase["{DBF}"]["Tables"]', '{"employees", "offices"}'),
    (f'RelationalDatabase["{DBF}"]["employees", "ForeignKeys"]',
     '{<|"ToTable" -> "offices", "FromColumns" -> {"officeCode"}, "ToColumns" -> {"officeCode"}|>}'),
    (f'RelationalDatabase["{DBF}"]["offices", "PrimaryKey", "Columns"]', '{"officeCode"}'),
    (f'RelationalDatabase["{DBF}"]["offices", "city", "Nullable"]', "False"),
    (f'RelationalDatabase["{DBF}"]["employees", "Indexes"]', '{<|"Columns" -> {"name"}, "IndexName" -> "emp_name"|>}'),
    (f'RelationalDatabase["{DBF}"]["offices", "Columns"]', '{"officeCode", "city", "phone"}'),
    (f'Normal[Tabular[RelationalDatabase["{DBF}"] -> "employees"]][[3]]',
     '<|"id" -> 3, "name" -> "Cy", "officeCode" -> Missing["NotAvailable"]|>'),
    (f'RelationalDatabase[{{"nope"}}, "{DBF}"]', Raises()),
]
DATACONNECTIONOBJECT = [
    ('DataConnectionObject[<|"ConnectionName" -> "AzureBlobStorage", "Location" -> "sample.parquet"|>]'
     '["ConnectionName"]', '"AzureBlobStorage"'),
    ('DataConnectionObject[<|"ConnectionName" -> "Dropbox", "Location" -> "a.csv"|>]["Redownload"]', "False"),
    ('DataConnectionObject[<|"ConnectionName" -> "Dropbox", "Location" -> "a.csv"|>]["Location"]', '"a.csv"'),
    ('DataConnectionObject[<|"Location" -> "a.csv"|>]', Raises()),
]
ALLOWEDDIMENSIONS = [
    ('Normal[Tabular[{{1, 2}}, AllowedDimensions -> {5, 2}]]', "{{1, 2}}"),
    ('Options[Tabular[{{1, 2}}, AllowedDimensions -> {5, 2}], AllowedDimensions]', "{AllowedDimensions -> {5, 2}}"),
    ('Head[AllowedDimensions]', "Symbol"),
]
OPTION_SYMBOLS = []
for _o, _v in [("DateReduction", '"Week"'), ("DateFunction", "DateObject"), ("TimeFormat", "Automatic"),
               ("ExpirationDate", "Infinity"), ("PartitionGranularity", "Automatic"),
               ("MaxExtraBandwidths", "Automatic"), ("MaxMixtureKernels", "Automatic"),
               ("ValueDimensions", "1")]:
    OPTION_SYMBOLS += [(_o, _o), (f"Head[{_o}]", "Symbol"), (f"{_o} -> {_v}", f"{_o} -> {_v}")]
INCLUDEGROUPAGGREGATES = [
    ('Normal[PivotTable[Tabular[{{"N", "x", 1}, {"S", "y", 2}}, {"r", "p", "u"}], Total[#u] &, "r", "p", '
     'IncludeGroupAggregates -> False]]', '{<|"r" -> "N", "x" -> 1, "y" -> Missing["NotAvailable"]|>, <|"r" -> "S", '
     '"x" -> Missing["NotAvailable"], "y" -> 2|>}'),
    ("Head[IncludeGroupAggregates]", "Symbol"),
    ("IncludeGroupAggregates -> True", "IncludeGroupAggregates -> True"),
]
INCLUDERELATEDTABLES_EXTRA = [
    (f'RelationalDatabase[{{"employees"}}, "{DBF}", IncludeRelatedTables -> False]["Tables"]', '{"employees"}'),
]
DATEFUNCTION_EXTRA = [('DateScale[DateFunction -> DateObject]', 'DateScale[DateFunction -> DateObject]')]


GROUPS = [v for k, v in list(globals().items()) if k.isupper() and k != "GROUPS" and isinstance(v, list)
          and v and isinstance(v[0], tuple)]
CASES = [c for g in GROUPS for c in g]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w12(case):
    check(case)
