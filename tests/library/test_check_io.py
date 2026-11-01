"""Real input/output cases for file, stream, URL, system, parallel, compiler and raw-memory heads that had no
case of their own. Everything runs in a fresh temporary directory, and nothing touches the network.

Sources of the expected values (never the implementation's own output):

* The Python standard library, as an independent reference: ``os.path``/``pathlib`` for file names and file
  types, files written and read back with ``open`` for streams and sizes, ``urllib.parse`` for URL encoding,
  ``os.environ`` for environment variables, ``subprocess`` conventions for exit codes and output.
* Wolfram Language reference pages for return conventions: DeleteFile returns Null, FileType gives File,
  Directory or None, FileSize gives Quantity[n, "Bytes"], ParallelEvaluate gives one result per kernel,
  DirectoryName keeps the trailing separator, IfCompiled[c, u] gives u outside compiled code, Compile returns
  machine reals, and FromLunationNumber[0] is the new moon of 6 January 2000 (about 18:14 UTC).

Symbols used as state are prefixed ``cbk`` so they cannot collide with other test modules.
"""
import os
import tempfile
import urllib.parse

import pytest

from _harness import Pred, Via, check, ev, ids
from mikoshilang.expr import Expr

T = tempfile.mkdtemp(prefix="cbkio")
A = os.path.join(T, "a.txt")
with open(A, "w") as f:
    f.write("hello\nworld\n")
with open(os.path.join(T, "nums.txt"), "w") as f:
    f.write("1 2 3\n")
os.makedirs(os.path.join(T, "d1"))
with open(os.path.join(T, "d1", "n.txt"), "w") as f:
    f.write("abc")


def q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def P(*parts):
    return os.path.join(T, *parts)


def Eq(x, what=None):
    return Pred(lambda v: v == x, what or f"== {x!r}")


def _head(h):
    return Pred(lambda v: isinstance(v, Expr) and v.head == h, f"an expression with head {h}")


def _exists(path, want=True):
    return Pred(lambda v: os.path.exists(path) is want, f"{path} {'exists' if want else 'is gone'} afterwards")


def _file_text(path, text):
    return Pred(lambda v: open(path).read() == text, f"{path} contains {text!r}")


FILES = [
    ('FileNameSplit["a/b/c.txt"]', '{"a", "b", "c.txt"}'),
    ('FileNameSplit["x/y"]', '{"x", "y"}'),
    ('FileNameSplit["/a/b/c.txt"]', '{"", "a", "b", "c.txt"}'),
    ('FileNameJoin[{"a", "b", "c.txt"}]', q(os.path.join("a", "b", "c.txt"))),
    ('FileNameJoin[{"x", "y"}]', q(os.path.join("x", "y"))),
    ('FileNameJoin[{"/", "a"}]', q(os.path.join("/", "a"))),
    ('FileNameTake["a/b/c.txt"]', '"c.txt"'),
    ('FileNameTake["a/b/c.txt", 2]', '"a/b"'),
    ('FileNameTake["a/b/c.txt", -2]', '"b/c.txt"'),
    ('FileNameDrop["a/b/c.txt"]', '"a/b"'),
    ('FileNameDrop["a/b/c.txt", 1]', '"b/c.txt"'),
    ('FileNameDrop["a/b/c.txt", -2]', '"a"'),
    ('FileNameDepth["a/b/c.txt"]', "3"),
    ('FileNameDepth["a"]', "1"),
    ('FileNameDepth["a/b"]', "2"),
    ('FileExtension["x/y.tar.gz"]', '"gz"'),
    ('FileExtension["c.txt"]', '"txt"'),
    ('FileExtension["noext"]', '""'),
    ('FileBaseName["x/y.tar.gz"]', '"y.tar"'),
    ('FileBaseName["c.txt"]', '"c"'),
    ('FileBaseName["noext"]', '"noext"'),
    ('DirectoryName["/a/b/c.txt"]', '"/a/b/"'),
    ('DirectoryName["a/c.txt"]', '"a/"'),
    ('DirectoryName["c.txt"]', '""'),
    (f'ExpandFileName[{q(T + "/./a.txt")}]', q(A)),
    (f'ExpandFileName[{q(T + "/d1/../a.txt")}]', q(A)),
    (f'ExpandFileName[{q(T)}]', q(os.path.abspath(T))),
    (f'AbsoluteFileName[{q(A)}]', q(os.path.realpath(A))),
    (f'AbsoluteFileName[{q(P("d1", "n.txt"))}]', q(os.path.realpath(P("d1", "n.txt")))),
    (f'AbsoluteFileName[{q(P("d1"))}]', q(os.path.realpath(P("d1")))),
    ('ParentDirectory["/a/b/c"]', '"/a/b"'),
    ('ParentDirectory["/a/b"]', '"/a"'),
    (f'ParentDirectory[{q(P("d1"))}]', q(T)),
    (f'FileExistsQ[{q(A)}]', True),
    (f'FileExistsQ[{q(P("nope"))}]', False),
    (f'FileExistsQ[{q(T)}]', True),
    (f'DirectoryQ[{q(T)}]', True),
    (f'DirectoryQ[{q(A)}]', False),
    (f'DirectoryQ[{q(P("nope"))}]', False),
    (f'FileType[{q(A)}]', "File"),
    (f'FileType[{q(T)}]', "Directory"),
    (f'FileType[{q(P("nope"))}]', "None"),
    (f'FileByteCount[{q(A)}]', str(os.path.getsize(A))),
    (f'FileByteCount[{q(P("d1", "n.txt"))}]', "3"),
    (f'FileByteCount[{q(P("nums.txt"))}]', str(os.path.getsize(P("nums.txt")))),
    (f'FileSize[{q(A)}]', f'Quantity[{os.path.getsize(A)}, "Bytes"]'),
    (f'FileSize[{q(P("d1", "n.txt"))}]', 'Quantity[3, "Bytes"]'),
    (f'FileSize[{q(P("nums.txt"))}]', 'Quantity[6, "Bytes"]'),
    (f'FileNames["*.txt", {q(T)}]', "{" + ", ".join(q(P(n)) for n in sorted(["a.txt", "nums.txt"])) + "}"),
    (f'FileNames["n*", {q(P("d1"))}]', "{" + q(P("d1", "n.txt")) + "}"),
    (f'FileNames["*.none", {q(T)}]', "{}"),
    (f'FindFile[{q(A)}]', q(A)),
    (f'FindFile[{q(P("d1", "n.txt"))}]', q(P("d1", "n.txt"))),
    (f'FindFile[{q(P("nope"))}]', "$Failed"),
    # creating, copying, renaming and deleting (each checked against the file system afterwards)
    (f'CreateDirectory[{q(P("c1"))}]', _exists(P("c1"))),
    (f'CreateDirectory[{q(P("c2"))}]', q(P("c2"))),
    (f'CreateDirectory[{q(P("c1", "sub"))}]', _exists(P("c1", "sub"))),
    (f'CreateFile[{q(P("c1", "f.txt"))}]', _exists(P("c1", "f.txt"))),
    (f'CreateFile[{q(P("c2", "g.txt"))}]', q(P("c2", "g.txt"))),
    (f'CreateFile[{q(P("c2", "h.txt"))}]', _file_text(P("c2", "h.txt"), "")),
    (f'CopyFile[{q(A)}, {q(P("b.txt"))}]', _file_text(P("b.txt"), "hello\nworld\n")),
    (f'CopyFile[{q(A)}, {q(P("b2.txt"))}]', q(P("b2.txt"))),
    (f'CopyFile[{q(P("d1", "n.txt"))}, {q(P("n2.txt"))}]', _file_text(P("n2.txt"), "abc")),
    (f'RenameFile[{q(P("b.txt"))}, {q(P("r.txt"))}]', _exists(P("b.txt"), False)),
    (f'RenameFile[{q(P("b2.txt"))}, {q(P("r2.txt"))}]', q(P("r2.txt"))),
    (f'RenameFile[{q(P("n2.txt"))}, {q(P("r3.txt"))}]', _file_text(P("r3.txt"), "abc")),
    (f'DeleteFile[{q(P("r.txt"))}]', "Null"),
    (f'DeleteFile[{q(P("r2.txt"))}]', _exists(P("r2.txt"), False)),
    (f'DeleteFile[{{{q(P("r3.txt"))}, {q(P("c2", "g.txt"))}}}]', _exists(P("r3.txt"), False)),
    (f'CopyDirectory[{q(P("d1"))}, {q(P("d2"))}]', _file_text(P("d2", "n.txt"), "abc")),
    (f'CopyDirectory[{q(P("d1"))}, {q(P("d4"))}]', q(P("d4"))),
    (f'CopyDirectory[{q(P("c1"))}, {q(P("c3"))}]', _exists(P("c3", "sub"))),
    (f'RenameDirectory[{q(P("d2"))}, {q(P("d3"))}]', _file_text(P("d3", "n.txt"), "abc")),
    (f'RenameDirectory[{q(P("d4"))}, {q(P("d5"))}]', q(P("d5"))),
    (f'RenameDirectory[{q(P("c3"))}, {q(P("c4"))}]', _exists(P("c3"), False)),
    (f'DeleteDirectory[{q(P("d3"))}, DeleteContents -> True]', _exists(P("d3"), False)),
    (f'DeleteDirectory[{q(P("d5"))}, DeleteContents -> True]', "Null"),
    (f'DeleteDirectory[{q(P("c4", "sub"))}]', _exists(P("c4", "sub"), False)),
    # file dates
    (f'SetFileDate[{q(P("nums.txt"))}, {{2020, 1, 2, 3, 4, 5}}]; FileDate[{q(P("nums.txt"))}]',
     "DateObject[{2020, 1, 2, 3, 4, 5.}, \"Instant\", \"Gregorian\", 0.]"),
    (f'SetFileDate[{q(P("d1", "n.txt"))}, {{2001, 2, 3, 4, 5, 6}}]; FileDate[{q(P("d1", "n.txt"))}]',
     "DateObject[{2001, 2, 3, 4, 5, 6.}, \"Instant\", \"Gregorian\", 0.]"),
    (f'SetFileDate[{q(A)}, {{2010, 5, 6, 7, 8, 9}}]',
     Pred(lambda v: abs(os.path.getmtime(A) - 1273129689) < 1, "mtime set to 2010-05-06T07:08:09Z")),
    (f'FileDate[{q(A)}]', _head("DateObject")),
    (f'FileDate[{q(P("d1"))}]', _head("DateObject")),
    (f'FileDate[{q(T)}]', _head("DateObject")),
    (f'FileSystemMap[StringLength, {q(P("d1"))}]', '<|"n.txt" -> ' + str(len(P("d1", "n.txt"))) + "|>"),
    (f'FileSystemMap[FileBaseName, {q(P("d1"))}]', '<|"n.txt" -> "n"|>'),
    (f'FileSystemMap[FileExtension, {q(P("d1"))}]', '<|"n.txt" -> "txt"|>'),
    (Via("FileSystemScan", f'cbkseen = {{}}; FileSystemScan[AppendTo[cbkseen, #] &, {q(P("d1"))}]; cbkseen'),
     "{" + q(P("d1", "n.txt")) + "}"),
    (f'FileSystemScan[Identity, {q(P("d1"))}]', "Null"),
    (Via("FileSystemScan", f'cbkn = 0; FileSystemScan[cbkn++ &, {q(T)}]; cbkn > 0'), True),
    # directories (each case restores the working directory)
    (f'SetDirectory[{q(T)}]; cbkd = Directory[]; ResetDirectory[]; cbkd', q(os.path.realpath(T))),
    (f'SetDirectory[{q(P("d1"))}]; cbkd = Directory[]; ResetDirectory[]; cbkd', q(os.path.realpath(P("d1")))),
    (f'SetDirectory[{q(T)}]; cbkd = FileExistsQ["a.txt"]; ResetDirectory[]; cbkd', True),
    ("Directory[]", Pred(lambda v: v == os.getcwd(), "the process working directory")),
    ("Directory[] === Directory[]", True),
    (Via("ResetDirectory", f'SetDirectory[{q(T)}]; cbkd = ResetDirectory[]; cbkd === Directory[]'), True),
    (Via("ResetDirectory", f'SetDirectory[{q(T)}]; SetDirectory[{q(P("d1"))}]; ResetDirectory[]; cbkd = Directory[]; '
                           'ResetDirectory[]; cbkd'), q(os.path.realpath(T))),
    ('StringQ[Directory[]]', True),
    (Via("ResetDirectory", f'SetDirectory[{q(T)}]; ResetDirectory[]'), Pred(lambda v: v == os.getcwd(), "the previous directory")),
    (f'SetDirectory[{q(T)}]; SetDirectory[{q(P("d1"))}]; ResetDirectory[]; cbkd = Directory[]; ResetDirectory[]; cbkd',
     q(os.path.realpath(T))),
    (Via("DirectoryStack", f'cbkd0 = Length[DirectoryStack[]]; SetDirectory[{q(T)}]; cbkd = Length[DirectoryStack[]] - cbkd0; '
     'ResetDirectory[]; cbkd'), "1"),
    (Via("DirectoryStack", f'cbkd0 = Length[DirectoryStack[]]; SetDirectory[{q(T)}]; SetDirectory[{q(T)}]; '
     'cbkd = Length[DirectoryStack[]] - cbkd0; ResetDirectory[]; ResetDirectory[]; cbkd'), "2"),
    (Via("DirectoryStack", "Head[DirectoryStack[]]"), "List"),
]

STREAMS = [
    (f'ReadList[{q(A)}, String]', '{"hello", "world"}'),
    (f'ReadList[{q(P("nums.txt"))}, Number]', "{1, 2, 3}"),
    (f'ReadList[{q(A)}, Word]', '{"hello", "world"}'),
    (f'ReadString[{q(A)}]', q(open(A).read())),
    (f'ReadString[{q(P("nums.txt"))}]', '"1 2 3\\n"'),
    (f'ReadString[{q(P("d1", "n.txt"))}]', '"abc"'),
    (Via("OpenRead", f'cbks = OpenRead[{q(A)}]; ReadLine[cbks]'), '"hello"'),
    ("ReadLine[cbks]", '"world"'),
    ("ReadLine[cbks]", "EndOfFile"),
    ("Close[cbks]", q(A)),
    (Via("OpenRead", f'cbks = OpenRead[{q(P("nums.txt"))}]; Read[cbks, Number]'), "1"),
    ("Read[cbks, Number]", "2"),
    ("Read[cbks, {Number}]", "{3}"),
    ("Close[cbks]", q(P("nums.txt"))),
    (Via("OpenRead", f'cbks = OpenRead[{q(A)}]; Skip[cbks, Character, 2]; StreamPosition[cbks]'), "2"),
    ("SetStreamPosition[cbks, 6]; Read[cbks, Word]", '"world"'),
    ("StreamPosition[cbks]", "11"),
    ("SetStreamPosition[cbks, 0]", "0"),
    ("Skip[cbks, Word]; Read[cbks, Word]", '"world"'),
    ("Close[cbks]", q(A)),
    # each write case starts from a known file state, so the expectation depends on the stream actually opened
    (Via("OpenWrite", f'If[FileExistsQ[{q(P("w.txt"))}], DeleteFile[{q(P("w.txt"))}]]; cbkw = OpenWrite[{q(P("w.txt"))}]; WriteString[cbkw, "ab"]; '
                      f'WriteLine[cbkw, "cd"]; Close[cbkw]; ReadString[{q(P("w.txt"))}]'), '"abcd\\n"'),
    (Via("OpenAppend", f'WriteString[{q(P("w.txt"))}, "abcd\\n"]; Close[{q(P("w.txt"))}]; '
                       f'cbkw = OpenAppend[{q(P("w.txt"))}]; WriteString[cbkw, "z"]; Close[cbkw]; '
                       f'ReadString[{q(P("w.txt"))}]'), '"abcd\\nz"'),
    (Via("OpenWrite", f'If[FileExistsQ[{q(P("w2.txt"))}], DeleteFile[{q(P("w2.txt"))}]]; cbkw = OpenWrite[{q(P("w2.txt"))}]; WriteLine[cbkw, "x"]; '
                      f'WriteLine[cbkw, "y"]; Close[cbkw]; ReadString[{q(P("w2.txt"))}]'), '"x\\ny\\n"'),
    (Via("OpenAppend", f'WriteString[{q(P("w2.txt"))}, "x\\ny\\n"]; Close[{q(P("w2.txt"))}]; '
                       f'cbkw = OpenAppend[{q(P("w2.txt"))}]; WriteLine[cbkw, "z"]; Close[cbkw]; '
                       f'ReadString[{q(P("w2.txt"))}]'), '"x\\ny\\nz\\n"'),
    (Via("OpenAppend", f'If[FileExistsQ[{q(P("w3.txt"))}], DeleteFile[{q(P("w3.txt"))}]]; cbkw = OpenAppend[{q(P("w3.txt"))}]; '
                       f'WriteString[cbkw, "new"]; Close[cbkw]; ReadString[{q(P("w3.txt"))}]'), '"new"'),
    (Via("OpenWrite", f'If[FileExistsQ[{q(P("w4.txt"))}], DeleteFile[{q(P("w4.txt"))}]]; cbkw = OpenWrite[{q(P("w4.txt"))}]; Close[cbkw]; '
                      f'FileExistsQ[{q(P("w4.txt"))}]'), True),
    (f'WriteString[{q(P("ws.txt"))}, "p", "q"]; Close[{q(P("ws.txt"))}]', _file_text(P("ws.txt"), "pq")),
    (f'WriteLine[{q(P("wl.txt"))}, "line"]; Close[{q(P("wl.txt"))}]', _file_text(P("wl.txt"), "line\n")),
    (f'Write[{q(P("e.m"))}, x^2]; Close[{q(P("e.m"))}]', _file_text(P("e.m"), "x^2\n")),
    (f'Write[{q(P("e2.m"))}, {{1, 2}}]; Close[{q(P("e2.m"))}]', _file_text(P("e2.m"), "{1, 2}\n")),
    (f'Write[{q(P("e3.m"))}, "s"]; Close[{q(P("e3.m"))}]', _file_text(P("e3.m"), '"s"\n')),
    (f'Read[{q(P("e.m"))}]', "x^2"),
    (f'Read[{q(P("e2.m"))}]', "{1, 2}"),
    (f'Read[{q(P("e3.m"))}]', '"s"'),
    (f'BinaryWrite[{q(P("bin"))}, {{1, 2, 255}}]; Close[{q(P("bin"))}]',
     Pred(lambda v: open(P("bin"), "rb").read() == bytes([1, 2, 255]), "bytes 01 02 FF")),
    (f'BinaryWrite[{q(P("bin2"))}, 65]; Close[{q(P("bin2"))}]',
     Pred(lambda v: open(P("bin2"), "rb").read() == b"A", "byte 41")),
    (f'BinaryWrite[{q(P("bin3"))}, {{258}}, "UnsignedInteger16"]; Close[{q(P("bin3"))}]',
     Pred(lambda v: open(P("bin3"), "rb").read() == (258).to_bytes(2, "little"), "258 as a little-endian uint16")),
    (f'BinaryReadList[{q(P("bin"))}]', "{1, 2, 255}"),
    (f'BinaryReadList[{q(P("bin2"))}]', "{65}"),
    (f'BinaryReadList[{q(P("bin3"))}, "UnsignedInteger16"]', "{258}"),
    (Via("OpenRead", f'cbkb = OpenRead[{q(P("bin"))}, BinaryFormat -> True]; BinaryRead[cbkb]'), "1"),
    ("BinaryRead[cbkb]", "2"),
    ("BinaryRead[cbkb]; Close[cbkb]", q(P("bin"))),
    (Via("ReadByteArray", f'Normal[ReadByteArray[{q(P("bin"))}]]'), "{1, 2, 255}"),
    (Via("ReadByteArray", f'Normal[ReadByteArray[{q(P("bin2"))}]]'), "{65}"),
    (Via("ReadByteArray", f'Normal[ReadByteArray[{q(P("d1", "n.txt"))}]]'), "{97, 98, 99}"),
    (Via("StringToStream", 'ReadList[StringToStream["1 2 3"], Number]'), "{1, 2, 3}"),
    (Via("StringToStream", 'ReadLine[StringToStream["ab\\ncd"]]'), '"ab"'),
    (Via("StringToStream", 'ReadList[StringToStream["x y"], Word]'), '{"x", "y"}'),
    ('Find[StringToStream["a\\nbcd\\nb"], "c"]', '"bcd"'),
    (f'Find[{q(A)}, "or"]', '"world"'),
    (f'Find[{q(A)}, "zz"]', "EndOfFile"),
    (Via("OpenRead", f'cbks = OpenRead[{q(A)}]; cbkin = MemberQ[Streams[], cbks]; Close[cbks]; cbkin'), True),
    (Via("Streams", 'MemberQ[Streams[], OutputStream["stdout", 1]]'), True),
    (Via("Streams", 'Head[Streams[]]'), "List"),
]

URL = [
    ('URLEncode["a b&c"]', q(urllib.parse.quote("a b&c", safe=""))),
    ('URLEncode["x/y?z"]', q(urllib.parse.quote("x/y?z", safe=""))),
    ('URLEncode["plain"]', '"plain"'),
    ('URLDecode["a%20b%26c"]', '"a b&c"'),
    ('URLDecode["x%2Fy"]', '"x/y"'),
    ('URLDecode["plain"]', '"plain"'),
    ('URLQueryEncode[{"a" -> 1, "b" -> "x y"}]', q(urllib.parse.urlencode({"a": 1, "b": "x y"}))),
    ('URLQueryEncode[<|"k" -> "v"|>]', '"k=v"'),
    ('URLQueryEncode[{"q" -> "a&b"}]', q(urllib.parse.urlencode({"q": "a&b"}))),
    ('URLQueryDecode["a=1&b=x+y"]', '{"a" -> "1", "b" -> "x y"}'),
    ('URLQueryDecode["k=v"]', '{"k" -> "v"}'),
    ('URLQueryDecode["q=a%26b"]', '{"q" -> "a&b"}'),
    ('URLParse["https://u@ex.com:8080/p/q?a=1#f"]["Domain"]', '"ex.com"'),
    ('URLParse["https://u@ex.com:8080/p/q?a=1#f"]["Port"]', "8080"),
    ('URLParse["https://u@ex.com:8080/p/q?a=1#f"]["Fragment"]', '"f"'),
    ('URLParse["https://u@ex.com:8080/p/q?a=1#f", "Scheme"]', '"https"'),
    ('URLBuild["https://ex.com/p", {"a" -> 1}]', '"https://ex.com/p?a=1"'),
    ('URLBuild["http://h/x", {"a" -> 1, "b" -> 2}]', '"http://h/x?a=1&b=2"'),
    ('HostLookup["127.0.0.1"]', Pred(lambda v: isinstance(v, str) and v, "a host name")),
    ('IPAddress["127.0.0.1"]', 'IPAddress["127.0.0.1"]'),
    ('IPAddress["::1"]', 'IPAddress["::1"]'),
    ('Head[IPAddress["10.0.0.1"]]', "IPAddress"),
]

SYSTEM = [
    ('Environment["CBK_SURELY_UNSET_VAR"]', "$Failed"),
    ('Environment["PATH"]', q(os.environ.get("PATH", ""))),
    ('Environment["HOME"]', q(os.environ.get("HOME", ""))),
    ('SetEnvironment["CBKX" -> "42"]; Environment["CBKX"]', '"42"'),
    ('SetEnvironment["CBKY" -> "a b"]', Pred(lambda v: os.environ.get("CBKY") == "a b", "os.environ updated")),
    ('SetEnvironment["CBKY" -> None]', Pred(lambda v: "CBKY" not in os.environ, "variable removed")),
    ('GetEnvironment["CBKX"]', '"CBKX" -> "42"'),
    ('GetEnvironment["PATH"]', '"PATH" -> ' + q(os.environ.get("PATH", ""))),
    ('GetEnvironment["CBK_SURELY_UNSET_VAR"]', '"CBK_SURELY_UNSET_VAR" -> None'),
    ('BlockRandom[SeedRandom[1]; RandomInteger[100]] === BlockRandom[SeedRandom[1]; RandomInteger[100]]', True),
    ('BlockRandom[RandomReal[]] < 1', True),
    ('SeedRandom[7]; cbkr1 = RandomReal[]; SeedRandom[7]; BlockRandom[RandomReal[]]; RandomReal[] == cbkr1', True),
    ('Run["true"]', "0"),
    ('Run["exit 3"]', Pred(lambda v: v != 0, "a non-zero status")),
    (f'Run["echo hi > {P("run.txt")}"]', _file_text(P("run.txt"), "hi\n")),
    ('RunProcess[{"echo", "hi"}, "StandardOutput"]', '"hi\\n"'),
    ('RunProcess[{"sh", "-c", "exit 3"}, "ExitCode"]', "3"),
    ('RunProcess[{"sh", "-c", "echo e 1>&2"}, "StandardError"]', '"e\\n"'),
    ('ParallelMap[#^2 &, {1, 2, 3}]', "{1, 4, 9}"),
    ('ParallelMap[f, {a, b}]', "{f[a], f[b]}"),
    ('ParallelMap[Length, {{1}, {1, 2}}]', "{1, 2}"),
    ("ParallelTable[i^2, {i, 4}]", "{1, 4, 9, 16}"),
    ("ParallelTable[i + j, {i, 2}, {j, 2}]", "{{2, 3}, {3, 4}}"),
    ("ParallelTable[x, 3]", "{x, x, x}"),
    ("ParallelSum[i, {i, 10}]", "55"),
    ("ParallelSum[i^2, {i, 3}]", "14"),
    ("ParallelSum[1/i, {i, 2}]", "3/2"),
    ("ParallelProduct[i, {i, 5}]", "120"),
    ("ParallelProduct[2, {i, 3}]", "8"),
    ("ParallelProduct[x, {i, 2}]", "x^2"),
    ("ParallelArray[#^2 &, 3]", "{1, 4, 9}"),
    ("ParallelArray[f, 2]", "{f[1], f[2]}"),
    ("ParallelArray[#1 + #2 &, {2, 2}]", "{{2, 3}, {3, 4}}"),
    ("ParallelCases[{1, a, 2, b}, _Integer]", "{1, 2}"),
    ("ParallelCases[{1, a, 2, b}, _Symbol]", "{a, b}"),
    ("ParallelCases[{x^2, y}, _^2]", "{x^2}"),
    ("ParallelSelect[{1, 2, 3, 4}, EvenQ]", "{2, 4}"),
    ("ParallelSelect[{1, 2, 3, 4}, # > 2 &]", "{3, 4}"),
    ("ParallelSelect[{a, 1}, NumberQ]", "{1}"),
    ("ParallelEvaluate[1 + 1]", Pred(lambda v: isinstance(v, Expr) and v.head == "List" and v.args and
                                     all(a == 2 for a in v.args), "one 2 per kernel")),
    ("Length[ParallelEvaluate[0]] == Length[Kernels[]]", True),
    ("Union[ParallelEvaluate[x]]", "{x}"),
    ("ParallelCombine[Map[#^2 &, #] &, {1, 2, 3}]", "{1, 4, 9}"),
    ("ParallelCombine[Select[#, EvenQ] &, {1, 2, 3, 4}]", "{2, 4}"),
    ("ParallelCombine[Map[f, #] &, {a, b}]", "{f[a], f[b]}"),
    ("ParallelTry[#^2 &, {3}]", "9"),
    ("ParallelTry[If[# > 1, #, $Failed] &, {2}]", "2"),
    ("ParallelTry[f, {a}]", "f[a]"),
    ("Parallelize[Map[#^2 &, {1, 2}]]", "{1, 4}"),
    ("Parallelize[Table[i, {i, 3}]]", "{1, 2, 3}"),
    ("Parallelize[Total[{1, 2, 3}]]", "6"),
    (Via("ParallelSubmit", "WaitAll[ParallelSubmit[1 + 2]]"), "3"),
    (Via("ParallelSubmit", "WaitAll[{ParallelSubmit[1], ParallelSubmit[2]}]"), "{1, 2}"),
    (Via("ParallelSubmit", "WaitAll[ParallelSubmit[x^2]]"), "x^2"),
    ("CriticalSection[{cbklk}, 5]", "5"),
    ("CriticalSection[{cbklk}, 1 + 1]", "2"),
    ("CriticalSection[{cbklk, cbklk2}, x]", "x"),
    (f'WithLock[File[{q(P("lock"))}], 7]', "7"),
    (f'WithLock[File[{q(P("lock"))}], x + 1]', "1 + x"),
    (f'WithLock[File[{q(P("lock2"))}], {{1}}]', "{1}"),
    ("LocalEvaluate[2 + 3]", "5"),
    ("LocalEvaluate[x + x]", "2 x"),
    ("LocalEvaluate[{1, 2}]", "{1, 2}"),
    ("TaskWait[SessionSubmit[cbkz = 9]]; cbkz", "9"),
    (Via("SessionSubmit", "Head[SessionSubmit[1]]"), "TaskObject"),
]

COMPILE = [
    ('FunctionCompile[Function[Typed[x, "MachineInteger"], x + 1]][41]', "42"),
    ('FunctionCompile[Function[Typed[x, "MachineInteger"], x^2]][7]', "49"),
    ('FunctionCompile[Function[Typed[x, "Real64"], x/2]][3.]', 1.5),
    ("Compile[{x}, x^2][3]", 9.0),
    ("Compile[{{x, _Real}}, Sin[x]][1.0]", 0.8414709848078965),
    ("Compile[{x, y}, x + y][1, 2]", 3.0),
    ("IfCompiled[1, 2]", "2"),
    ("IfCompiled[a, b]", "b"),
    ("IfCompiled[x, 1 + 1]", "2"),
]



def _near_date(y, mo, d, h, mi, tol_min=30):
    import datetime as dt
    want = dt.datetime(y, mo, d, h, mi)

    def f(v):
        if not (isinstance(v, Expr) and v.head == "DateObject"):
            return False
        a = [float(x) for x in v.args[0].args]
        got = dt.datetime(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4])) + dt.timedelta(seconds=a[5])
        return abs((got - want).total_seconds()) <= tol_min * 60
    return Pred(f, f"a DateObject within {tol_min} min of {want:%Y-%m-%d %H:%M} UTC")


# New moons from Meeus, Astronomical Algorithms (ch. 49): lunation 0 is 2000-01-06 18:14 UTC
DATES = [
    ("FromLunationNumber[0]", _near_date(2000, 1, 6, 18, 14)),
    ("FromLunationNumber[1]", _near_date(2000, 2, 5, 13, 3)),
    ("FromLunationNumber[12]", _near_date(2000, 12, 25, 17, 22)),
]

CASES = FILES + STREAMS + URL + SYSTEM + COMPILE + DATES


@pytest.fixture(scope="module", autouse=True)
def cbkio_module():
    """Remove the tasks and close the parallel kernels this module starts, so later modules see a clean session."""
    from mikoshilang.library import wl_system as W
    kernels = set(W._KERNELS)
    tasks = set(W._TASKS)
    yield
    for uuid, t in list(W._TASKS.items()):
        if uuid not in tasks:
            t.remove()
    ev("Unset[$DefaultParallelKernels]")
    for kid in list(W._KERNELS):
        if kid not in kernels:
            ev("CloseKernels[%d]" % kid)
    assert set(W._KERNELS) == kernels, "parallel kernels left running"
    assert set(W._TASKS) <= tasks, "tasks left behind"


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_check_io(case):
    check(case)
