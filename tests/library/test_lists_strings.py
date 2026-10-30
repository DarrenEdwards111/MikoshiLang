import base64
import hashlib
import time

import pytest

from _harness import Raises, Pred, check, ids
from mikoshilang.expr import Symbol

H = lambda f, s: '"' + f(s.encode()).hexdigest() + '"'  # noqa: E731

LISTS = [
    ("Union[{3, 1}, {2, 1}]", "{1, 2, 3}"), ("Union[{b, a, b}]", "{a, b}"), ("Union[{1}, {2}, {1, 3}]", "{1, 2, 3}"), ("Union[]", "{}"),
    ("Intersection[{1, 2, 3}, {2, 3, 4}]", "{2, 3}"), ("Intersection[{c, a, b}, {b, c}]", "{b, c}"), ("Intersection[{1}, {2}]", "{}"),
    ("Intersection[{1, 2}, {2, 3}, {2}]", "{2}"),
    ("Complement[{1, 2, 3}, {2}]", "{1, 3}"), ("Complement[{3, 1, 1}, {}]", "{1, 3}"), ("Complement[{1, 2, 3, 4}, {1}, {4}]", "{2, 3}"),
    ("SetDifference[{1, 2, 3}, {2}]", "{1, 3}"), ("SetDifference[{a, b}, {a, b}]", "{}"), ("SetDifference[{2, 1, 2}, {}]", "{1, 2}"),
    ("SymmetricDifference[{1, 2}, {2, 3}]", "{1, 3}"), ("SymmetricDifference[{1}, {1}]", "{}"), ("SymmetricDifference[{a}, {b}]", "{a, b}"),
    ("Cardinality[{1, 1, 2}]", "2"), ("Cardinality[{}]", "0"), ("Cardinality[{a, b, c}]", "3"),
    ("ElementQ[2, {1, 2}]", True), ("ElementQ[5, {1, 2}]", False), ("ElementQ[1, {1.0}]", False),
    ("SubsetQ[{1, 2, 3}, {1, 2}]", True), ("SubsetQ[{1, 2}, {1, 2, 3}]", False), ("SubsetQ[{a}, {}]", True),
    ("DeleteDuplicates[{1, 2, 1, 3}]", "{1, 2, 3}"), ("DeleteDuplicates[{}]", "{}"), ("DeleteDuplicates[{b, a, b}]", "{b, a}"),
    ("Riffle[{a, b, c}, x]", "{a, x, b, x, c}"), ("Riffle[{a, b, c}, {x, y}]", "{a, x, b, y, c}"), ("Riffle[{a}, x]", "{a}"),
    ("Interleave[{1, 2}, {a, b}]", "{1, a, 2, b}"), ("Interleave[{1}, {2}, {3}]", "{1, 2, 3}"), ("Interleave[{1, 2, 3}, {a}]", Raises()),
    ("Partition[{1, 2, 3, 4, 5}, 2]", "{{1, 2}, {3, 4}}"), ("Partition[{1, 2, 3, 4, 5}, 3, 1]", "{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}"),
    ("Partition[{1, 2, 3, 4}, 2, 3]", "{{1, 2}}"), ("Partition[{1, 2}, 0]", Raises()),
    ("Chunk[{1, 2, 3, 4, 5}, 2]", "{{1, 2}, {3, 4}, {5}}"), ("Chunk[{}, 2]", "{}"), ("Chunk[{1, 2}, 5]", "{{1, 2}}"),
    ("SlidingWindow[{1, 2, 3, 4}, 2]", "{{1, 2}, {2, 3}, {3, 4}}"), ("SlidingWindow[{1, 2}, 3]", "{}"), ("SlidingWindow[{1, 2, 3}, 3]", "{{1, 2, 3}}"),
    ("RotateLeft[{1, 2, 3}]", "{2, 3, 1}"), ("RotateLeft[{1, 2, 3}, 2]", "{3, 1, 2}"), ("RotateLeft[{1, 2, 3}, -1]", "{3, 1, 2}"),
    ("RotateRight[{1, 2, 3}]", "{3, 1, 2}"), ("RotateRight[{1, 2, 3}, 2]", "{2, 3, 1}"), ("RotateRight[{}, 2]", "{}"),
    ("Accumulate[{1, 2, 3}]", "{1, 3, 6}"), ("Accumulate[{1/2, 1/3}]", "{1/2, 5/6}"), ("Accumulate[{}]", "{}"),
    ("Differences[{1, 4, 9}]", "{3, 5}"), ("Differences[{1, 4, 9, 16}, 2]", "{2, 2}"), ("Differences[{5}]", "{}"),
    ("GroupBy[{1, 2, 3, 4}, EvenQ]", "Association[False -> {1, 3}, True -> {2, 4}]"),
    ("GroupBy[{a, b, a}, Length]", Pred(lambda r: r.head == "Association", "association")),
    ("GroupBy[{}, EvenQ]", "Association[]"),
    ("Tally[{a, b, a}]", "{{a, 2}, {b, 1}}"), ("Tally[{1, 1.0}]", "{{1, 1}, {1.0, 1}}"), ("Tally[{}]", "{}"),
]

STRINGS = [
    ("StringTake[\"abcdef\", 2]", "\"ab\""), ("StringTake[\"abcdef\", -2]", "\"ef\""), ("StringTake[\"abcdef\", {2, 4}]", "\"bcd\""),
    ("StringTake[\"abc\", 5]", Raises()),
    ("StringDrop[\"abcdef\", 2]", "\"cdef\""), ("StringDrop[\"abcdef\", -2]", "\"abcd\""), ("StringDrop[\"abcdef\", {2, 3}]", "\"adef\""),
    ("StringPart[\"abc\", 2]", "\"b\""), ("StringPart[\"abc\", -1]", "\"c\""), ("StringPart[\"abc\", 4]", Raises()),
    ("StringInsert[\"abc\", \"X\", 2]", "\"aXbc\""), ("StringInsert[\"abc\", \"X\", -1]", "\"abcX\""), ("StringInsert[\"abc\", \"X\", 1]", "\"Xabc\""),
    ("StringDelete[\"hello\", \"l\"]", "\"heo\""), ("StringDelete[\"aaa\", \"aa\"]", "\"a\""), ("StringDelete[\"abc\", \"z\"]", "\"abc\""),
    ("StringReplace[\"abc\", \"b\" -> \"x\"]", "\"axc\""), ("StringReplace[\"abab\", {\"a\" -> \"b\", \"b\" -> \"a\"}]", "\"baba\""),
    ("StringReplace[\"abc\", \"b\", \"x\"]", Raises()),
    ("StringPadLeft[\"abc\", 5]", "\"  abc\""), ("StringPadLeft[\"abcdef\", 3]", "\"def\""), ("StringPadLeft[\"7\", 3, \"0\"]", "\"007\""),
    ("StringPadRight[\"abc\", 5, \"*\"]", "\"abc**\""), ("StringPadRight[\"abcdef\", 3]", "\"abc\""), ("StringPadRight[\"a\", 2, \"xy\"]", Raises()),
    ("StringCenter[\"ab\", 6]", "\"  ab  \""), ("StringCenter[\"abc\", 3]", "\"abc\""), ("StringCenter[\"a\", 4, \"*\"]", Pred(lambda r: len(r) == 4 and "a" in r, "width 4")),
    ("StringRepeat[\"ab\", 3]", "\"ababab\""), ("StringRepeat[\"x\", 0]", Raises()), ("StringRepeat[\"\", 2]", "\"\""), ("StringRepeat[\"x\", -1]", Raises()),
    ("StringReverse[\"abc\"]", "\"cba\""), ("StringReverse[\"\"]", "\"\""), ("StringReverse[\"ab c\"]", "\"c ba\""),
    ("StringToUpperCase[\"abC\"]", "\"ABC\""), ("StringToUpperCase[\"1a\"]", "\"1A\""), ("StringToUpperCase[\"\"]", "\"\""),
    ("StringToLowerCase[\"AbC\"]", "\"abc\""), ("StringToLowerCase[\"1A\"]", "\"1a\""), ("StringToLowerCase[\"\"]", "\"\""),
    ("StringCapitalize[\"hELLO\"]", "\"HELLO\""), ("StringCapitalize[\"abc def\"]", "\"Abc def\""), ("StringCapitalize[\"\"]", "\"\""),
    ("StringToTitleCase[\"hello wORLD\"]", "\"Hello WORLD\""), ("StringToTitleCase[\"a b\"]", "\"A B\""), ("StringToTitleCase[\"\"]", "\"\""),
    ("StringTrim[\"  a  \"]", "\"a\""), ("StringTrim[\"a\"]", "\"a\""), ("StringTrim[\"\\n x \\t\"]", "\"x\""),
    ("StringCount[\"abcabc\", \"bc\"]", "2"), ("StringCount[\"aaaa\", \"aa\"]", "2"), ("StringCount[\"abc\", \"z\"]", "0"),
    ("StringContainsQ[\"abc\", \"b\"]", True), ("StringContainsQ[\"abc\", \"d\"]", False), ("StringContainsQ[\"abc\", \"\"]", True),
    ("StringStartsQ[\"abc\", \"ab\"]", True), ("StringStartsQ[\"abc\", \"b\"]", False), ("StringStartsQ[\"\", \"\"]", True),
    ("StringEndsQ[\"abc\", \"bc\"]", True), ("StringEndsQ[\"abc\", \"b\"]", False), ("StringEndsQ[\"abc\", \"\"]", True),
    ("Characters[\"ab\"]", "{\"a\", \"b\"}"), ("Characters[\"\"]", "{}"), ("Characters[\"a b\"]", "{\"a\", \" \", \"b\"}"),
    ("ToCharacterCode[\"ab\"]", "{97, 98}"), ("ToCharacterCode[\"\"]", "{}"), ("ToCharacterCode[\"\\u00e9\"]", "{233}"),
    ("FromCharacterCode[{97, 98}]", "\"ab\""), ("FromCharacterCode[65]", "\"A\""), ("FromCharacterCode[-1]", Raises()),
    ("AlphabetPosition[\"abc\"]", "{1, 2, 3}"), ("AlphabetPosition[\"Z\"]", "26"), ("AlphabetPosition[\"a1\"]", "{1, 0}"),
    ("Base64Encode[\"hi\"]", '"' + base64.b64encode(b"hi").decode() + '"'), ("Base64Encode[\"\"]", "\"\""),
    ("Base64Encode[\"hello world\"]", '"' + base64.b64encode(b"hello world").decode() + '"'),
    ("Base64Decode[\"aGk=\"]", "\"hi\""), ("Base64Decode[\"\"]", "\"\""), ("Base64Decode[\"@@@\"]", Raises()),
    ("HexEncode[\"hi\"]", "\"6869\""), ("HexEncode[\"\"]", "\"\""), ("HexEncode[\"A\"]", "\"41\""),
    ("HexDecode[\"6869\"]", "\"hi\""), ("HexDecode[\"41\"]", "\"A\""), ("HexDecode[\"zz\"]", Raises()),
    ("MD5[\"abc\"]", H(hashlib.md5, "abc")), ("MD5[\"\"]", H(hashlib.md5, "")), ("MD5[\"The quick\"]", H(hashlib.md5, "The quick")),
    ("SHA1[\"abc\"]", H(hashlib.sha1, "abc")), ("SHA1[\"\"]", H(hashlib.sha1, "")), ("SHA1[\"xyz\"]", H(hashlib.sha1, "xyz")),
    ("SHA256[\"abc\"]", H(hashlib.sha256, "abc")), ("SHA256[\"\"]", H(hashlib.sha256, "")), ("SHA256[\"xyz\"]", H(hashlib.sha256, "xyz")),
    ("SHA512[\"abc\"]", H(hashlib.sha512, "abc")), ("SHA512[\"\"]", H(hashlib.sha512, "")), ("SHA512[\"xyz\"]", H(hashlib.sha512, "xyz")),
    ("ROT13[\"Hello\"]", "\"Uryyb\""), ("ROT13[\"Uryyb\"]", "\"Hello\""), ("ROT13[\"123\"]", "\"123\""),
    ("CaesarCipher[\"abc\", 1]", "\"bcd\""), ("CaesarCipher[\"xyz\", 3]", "\"abc\""), ("CaesarCipher[\"Ab!\", -1]", "\"Za!\""),
    ("XORCipher[\"ab\", \"k\"]", "\"0a09\""), ("XORCipher[\"\", \"k\"]", "\"\""), ("XORCipher[\"a\", \"\"]", Raises()),
    ("DayOfWeek[2026, 10, 30]", Pred(lambda r: r == Symbol("Friday"), "Friday")), ("DayOfWeek[{2000, 1, 1}]", Pred(lambda r: r == Symbol("Saturday"), "Saturday")),
    ("DayOfWeek[2023, 2, 29]", Raises()),
    ("DaysInMonth[2024, 2]", "29"), ("DaysInMonth[2023, 2]", "28"), ("DaysInMonth[2023, 13]", Raises()),
    ("LeapYearQ[2024]", True), ("LeapYearQ[1900]", False), ("LeapYearQ[2000]", True),
    ("DateString[{2026, 10, 30, 14, 5, 9}]", "\"Fri 30 Oct 2026 14:05:09\""), ("DateString[{2000, 1, 1}]", "\"Sat 1 Jan 2000 00:00:00\""),
    ("DateString[]", Pred(lambda r: isinstance(r, str) and len(r.split()) == 5, "current date")),
    ("TimeString[{2026, 10, 30, 14, 5, 9}]", "\"14:05:09\""), ("TimeString[{2026, 1, 1}]", "\"00:00:00\""),
    ("TimeString[]", Pred(lambda r: isinstance(r, str) and r.count(":") == 2, "current time")),
    ("DateTimeString[{2026, 10, 30, 14, 5, 9}]", "\"2026-10-30T14:05:09\""), ("DateTimeString[{2026, 1, 2}]", "\"2026-01-02T00:00:00\""),
    ("DateTimeString[]", Pred(lambda r: isinstance(r, str) and "T" in r, "current timestamp")),
    ("UnixTime[{1970, 1, 2, 0, 0, 0}]", "86400"), ("UnixTime[{2000, 1, 1}]", "946684800"),
    ("UnixTime[]", Pred(lambda r: abs(r - time.time()) < 5, "now")),
]

CASES = LISTS + STRINGS


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_lists_strings(case):
    check(case)
