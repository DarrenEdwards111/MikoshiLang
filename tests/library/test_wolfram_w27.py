"""Differential tests for Wolfram batch W27 (files, directories and streams; module ``wl_files``).

Every test runs in its own temporary directory, created by the ``w27env`` fixture (tempfile.mkdtemp) and removed
afterwards. The fixture makes it the working directory, the home directory ($HomeDirectory) and the parent of
$TemporaryDirectory, and binds the symbol ``w27dir`` to its absolute name, so no test touches a file outside it and
no random path appears in a test id. Streams opened by a test, string streams, stream methods, the directory stack
and the working directory are restored afterwards.

Reference values (written independently of the implementation):

* Wolfram documentation pages on reference.wolfram.com/language/ref/ (cited per case as "doc: Name"): file-name
  examples (FileNameSplit, FileNameDepth, FileNameDrop, DirectoryName, FileExtension, FileBaseName), the Read,
  Number and ReadList examples ("abcdefg 123456", "123 abc 456", "12.3 456 789", "aαℵ" character codes), the
  DefineInputStreamMethod "StringName"/"TextName" examples, and the documented semantics of DeleteDirectory,
  CreateFile, CopyFile, RenameFile, SetDirectory/DirectoryStack, Close, Streams and $StandardOutputStream;
* IEEE 754-2008 binary interchange formats (binary32 0.1 = 0x3DCCCCCD, binary64 1.0 = 0x3FF0000000000000, binary128
  1.0 = 0x3FFF0000000000000000000000000000, infinities 0x7FF0..., 0x7F80...) and IntegerDigits[n, 256] byte order;
* Python's os, os.path, tempfile and sys modules (byte counts, os.sep, sys.byteorder) for system-dependent values;
* the byte content of the sample files written by the fixture (counted by hand in comments).
"""
import os
import shutil
import sys
import tempfile

import pytest

from _harness import Pred, Raises, check, ev, ids
from mikoshilang.expr import Expr, Symbol

SAMPLE = {
    "a.txt": b"1 2 3\nhello world\nx+y\n",          # 6 + 12 + 4 = 22 bytes
    "bin.dat": bytes([1, 2, 3, 255]),
    "sub/b.txt": b"B\n",                              # 2 bytes
    "sub/inner/c.m": b"1+1\n",                        # 4 bytes
    "Pkg/init.m": b"42\n",
    "infile.m": b"$InputFileName\n",
    "crlf.txt": b"a\r\nb\r\n",
    "nums.txt": b"1 2.5 -3e2\n4*^2 .5\n",
    "empty.txt": b"",
}
EMPTY_DIRS = ("emptydir", "tmp")


def _is_list(x, n=None):
    return isinstance(x, Expr) and x.head == "List" and (n is None or len(x.args) == n)


def _same_path(a, b):
    return isinstance(a, str) and isinstance(b, str) and os.path.normcase(os.path.realpath(a)) == \
        os.path.normcase(os.path.realpath(b))


def _sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def _ev(src):
    return ev(src)


# ───────────────────────────────────────────── file names ─────────────────────────────────────────────

NAMES = [
    # FileNameSplit
    ('FileNameSplit["a/b/c", OperatingSystem -> "Unix"]', '{"a", "b", "c"}'),                  # doc: FileNameDepth "a/b/c" has 3 parts
    ('FileNameSplit["/usr/local/bin", OperatingSystem -> "Unix"]', '{"", "usr", "local", "bin"}'),  # doc: absolute names start with ""
    (r'FileNameSplit["\\\\server\\share\\path\\file", OperatingSystem -> "Windows"]',
     r'{"\\\\server\\share", "path", "file"}'),                                                # doc: share name is one unit
    (r'FileNameSplit["C:\\Users\\bill\\report.nb", OperatingSystem -> "Windows"]',
     '{"C:", "Users", "bill", "report.nb"}'),                                                   # doc: drive is the first part
    ('FileNameSplit[File["a//b/"], OperatingSystem -> "Unix"]', '{"a", "b"}'),                 # doc: File[...] supported
    ('FileNameSplit[w27x]', 'FileNameSplit[w27x]'),                                             # symbolic: unevaluated
    ('FileNameSplit[1]', Raises(TypeError)),
    ('FileNameSplit["a", OperatingSystem -> "Plan9"]', Raises(ValueError)),
    ('FileNameJoin[FileNameSplit["/x/y/z.txt", OperatingSystem -> "Unix"], OperatingSystem -> "Unix"]',
     '"/x/y/z.txt"'),                                                                          # Check: round trip
    # FileNameJoin
    ('FileNameJoin[{"a", "b", "c"}, OperatingSystem -> "Unix"]', '"a/b/c"'),                   # doc: FileNameJoin[{"a","b","c"}]
    ('FileNameJoin[{"", "a", "b"}, OperatingSystem -> "Unix"]', '"/a/b"'),                     # doc: {"", ...} is absolute
    ('FileNameJoin[{"a", "b/c"}, OperatingSystem -> "Unix"]', '"a/b/c"'),                      # doc: names may contain separators
    ('FileNameJoin[{"C:", "Users", "bill"}, OperatingSystem -> "Windows"]', r'"C:\\Users\\bill"'),
    ('FileNameJoin["a//b/", OperatingSystem -> "Unix"]', '"a/b"'),                             # doc: canonicalises a name
    ('FileNameJoin[{"a", "b"}, OperatingSystem -> "Windows"]', r'"a\\b"'),
    ('FileNameJoin[{"a", w27x}]', 'FileNameJoin[{"a", w27x}]'),                                 # symbolic
    ('FileNameJoin[{"a", 1}]', Raises(TypeError)),
    ('FileNameJoin[{"a", "C:"}, OperatingSystem -> "Windows"]', Raises(ValueError)),
    ('FileNameSplit[FileNameJoin[{"p", "q", "r"}]]', '{"p", "q", "r"}'),                       # Check: round trip, native OS
    # FileNameTake
    ('FileNameTake["a/b/c", OperatingSystem -> "Unix"]', '"c"'),                               # doc: last element by default
    ('FileNameTake["a/b/c", 2, OperatingSystem -> "Unix"]', '"a/b"'),
    ('FileNameTake["a/b/c", -2, OperatingSystem -> "Unix"]', '"b/c"'),
    ('FileNameTake["a/b/c", {2, 3}, OperatingSystem -> "Unix"]', '"b/c"'),
    ('FileNameTake["a/b/c", 10, OperatingSystem -> "Unix"]', '"a/b/c"'),                       # doc: more than present: all
    ('FileNameTake["a/b", w27n]', 'FileNameTake["a/b", w27n]'),
    ('FileNameTake["a/b/c", {5, 6}]', Raises(ValueError)),
    ('FileNameTake["a/b/c", "x"]', Raises(TypeError)),
    ('FileNameJoin[{FileNameTake["/a/b/c/d", 2], FileNameDrop["/a/b/c/d", 2]}]', '"/a/b/c/d"'
     if os.sep == "/" else 'FileNameJoin[{"", "a", "b", "c", "d"}]'),                          # Check: Take + Drop = name
    # FileNameDrop
    ('FileNameDrop["a/b/c", OperatingSystem -> "Unix"]', '"a/b"'),                             # doc: drops the last element
    ('FileNameDrop[FileNameJoin[{"a", "b", "c"}], -1]', 'FileNameJoin[{"a", "b"}]'),           # doc: FileNameDrop[%, -1]
    (r'FileNameDrop["\\\\server\\share\\path\\file", 1, OperatingSystem -> "Windows"]', r'"path\\file"'),  # doc
    ('FileNameDrop["a/b/c", {2}, OperatingSystem -> "Unix"]', '"a/c"'),
    ('FileNameDrop["a/b/c", 10, OperatingSystem -> "Unix"]', '""'),                            # doc: drops those present
    ('FileNameDrop[File["a/b/c"], OperatingSystem -> "Unix"]', '"a/b"'),                       # doc: File[...] supported
    ('FileNameDrop["a/b/c", {0, 1}]', Raises(ValueError)),
    ('FileNameDrop["a/b/c", 1.5]', Raises(TypeError)),
    # FileNameDepth
    ('FileNameDepth["a/b/c", OperatingSystem -> "Unix"]', '3'),                                # doc: FileNameDepth["a/b/c"] = 3
    ('FileNameDepth[FileNameJoin[{"a", "b", "c"}]]', '3'),                                     # doc: basic example
    (r'FileNameDepth["\\\\server\\share\\path\\file", OperatingSystem -> "Windows"]', '3'),    # doc: share is one unit
    ('FileNameDepth[File["x/y"], OperatingSystem -> "Unix"]', '2'),
    ('FileNameDepth[w27x]', 'FileNameDepth[w27x]'),
    ('FileNameDepth[{1}]', Raises(TypeError)),
    ('FileNameDepth["/q/r/s"] == Length[FileNameSplit["/q/r/s"]]', 'True'),                    # Check: definition
    # FileExtension, FileBaseName
    ('FileExtension["file.txt"]', '"txt"'),                                                    # doc: basic example
    ('FileExtension["file."]', '""'),                                                          # doc: possible issues
    ('FileExtension["arch.tar.gz"]', '"gz"'),                                                  # doc: only the last ending
    ('FileExtension["dir.d/file", OperatingSystem -> "Unix"]', '""'),                          # doc: ignores directories
    ('FileExtension["a/b.c/", OperatingSystem -> "Unix"]', '""'),                              # doc: directory-form name
    ('FileExtension[File["file.txt"]]', '"txt"'),                                              # doc: scope
    ('FileExtension[w27x]', 'FileExtension[w27x]'),
    ('FileExtension[3]', Raises(TypeError)),
    ('FileBaseName["file.txt"]', '"file"'),                                                    # doc: "name.ext" -> "name"
    ('FileBaseName["file."]', '"file"'),                                                       # doc: possible issues
    ('FileBaseName["dir/x.tar.gz", OperatingSystem -> "Unix"]', '"x.tar"'),                    # doc: drops directories
    ('FileBaseName[File["d/noext"], OperatingSystem -> "Unix"]', '"noext"'),
    ('FileBaseName[w27x]', 'FileBaseName[w27x]'),
    ('FileBaseName[{"a"}]', Raises(TypeError)),
    ('FileBaseName["p/q.r.s"] <> "." <> FileExtension["p/q.r.s"] == FileNameTake["p/q.r.s"]', 'True'),  # Check
    # DirectoryName
    ('DirectoryName["/Users/steve/report.nb", OperatingSystem -> "Unix"]', '"/Users/steve/"'),  # doc
    (r'DirectoryName["C:\\Users\\bill\\report.nb", OperatingSystem -> "Windows"]', r'"C:\\Users\\bill\\"'),  # doc
    (r'DirectoryName["a\\b\\c", 2, OperatingSystem -> "Windows"]', r'"a\\"'),                # doc: applied twice
    ('DirectoryName["file"]', '""'),                                                           # doc: no directory part
    ('DirectoryName[File["a/b/c"], OperatingSystem -> "Unix"]', '"a/b/"'),                     # doc: scope
    ('DirectoryName["/x", OperatingSystem -> "Unix"]', '"/"'),
    ('DirectoryName[w27x]', 'DirectoryName[w27x]'),
    ('DirectoryName["a/b", 0]', Raises(ValueError)),
    ('DirectoryName["a/b", "c"]', Raises(TypeError)),
    ('DirectoryName["u/v/w", OperatingSystem -> "Unix"] <> FileNameTake["u/v/w", OperatingSystem -> "Unix"]',
     '"u/v/w"'),                                                                               # Check: reassembly
]

# ─────────────────────────────── directories, $-variables (temp dir is cwd and home) ───────────────────────────────

DIRS = [
    ('ExpandFileName["a.txt"] == FileNameJoin[{Directory[], "a.txt"}]', 'True'),             # doc: relative to Directory[]
    ('ExpandFileName["sub/../a.txt"] == FileNameJoin[{w27dir, "a.txt"}]', 'True'),           # doc: . and .. expanded
    ('ExpandFileName["~"] == $HomeDirectory', 'True'),
    ('ExpandFileName[File["nothere/x"]] == FileNameJoin[{w27dir, "nothere", "x"}]', 'True'),  # need not exist
    ('ExpandFileName[w27x]', 'ExpandFileName[w27x]'),
    ('ExpandFileName[1]', Raises(TypeError)),
    ('AbsoluteFileName["a.txt"] == FileNameJoin[{w27dir, "a.txt"}]', 'True'),                # doc
    ('AbsoluteFileName["sub/./inner/../b.txt"] == FileNameJoin[{w27dir, "sub", "b.txt"}]', 'True'),
    ('AbsoluteFileName[File["sub"]] == FileNameJoin[{w27dir, "sub"}]', 'True'),
    ('AbsoluteFileName["nothere.txt"]', Raises(ValueError)),                                  # doc: checks the file system
    ('AbsoluteFileName[1]', Raises(TypeError)),
    ('FileExistsQ[AbsoluteFileName["sub/b.txt"]] && AbsoluteFileName["a.txt"] == ExpandFileName["a.txt"]', 'True'),  # Check
    ('ParentDirectory[] == FileNameDrop[w27dir]', 'True'),                                    # doc: parent of Directory[]
    ('ParentDirectory["sub/inner"] == FileNameJoin[{w27dir, "sub"}]', 'True'),
    ('ParentDirectory[FileNameJoin[{w27dir, "sub", "inner"}], 2] == w27dir', 'True'),
    ('ParentDirectory[File["sub"]] == w27dir', 'True'),                                       # doc: File supported
    ('ParentDirectory[2] == FileNameDrop[w27dir, -2]', 'True'),                               # doc: n levels up
    ('ParentDirectory["sub", -1]', Raises(ValueError)),
    ('ParentDirectory[1.5]', Raises(TypeError)),
    ('ParentDirectory["sub"] == DirectoryName[FileNameJoin[{w27dir, "sub"}]] || '
     'ParentDirectory["sub"] <> $PathnameSeparator == DirectoryName[FileNameJoin[{w27dir, "sub"}]]', 'True'),  # Check
    ('Directory[] == w27dir', 'True'),
    ('Module[{d = SetDirectory["sub"]}, {d == FileNameJoin[{w27dir, "sub"}], Directory[] == d}]', '{True, True}'),
    ('Module[{}, SetDirectory["sub"]; SetDirectory["inner"]; Directory[] == FileNameJoin[{w27dir, "sub", "inner"}]]',
     'True'),
    ('SetDirectory[] == $HomeDirectory', 'True'),                                             # doc: SetDirectory[] -> home
    ('SetDirectory["nothere"]', Raises(ValueError)),
    ('SetDirectory["a.txt"]', Raises(ValueError)),
    ('SetDirectory[3]', Raises(TypeError)),
    ('Module[{}, SetDirectory[File["sub"]]; FileExistsQ["b.txt"]]', 'True'),                  # Check: relative names follow
    ('DirectoryStack[]', '{}'),
    ('Module[{}, SetDirectory["sub"]; DirectoryStack[] == {w27dir}]', 'True'),                # doc: prepends previous dir
    ('Module[{}, SetDirectory["sub"]; SetDirectory["inner"]; DirectoryStack[] == {FileNameJoin[{w27dir, "sub"}], w27dir}]',
     'True'),
    ('Module[{}, SetDirectory["sub"]; ResetDirectory[]; DirectoryStack[]]', '{}'),            # Check: push/pop
    ('Module[{}, SetDirectory["sub"]; ResetDirectory[] == w27dir && Directory[] == w27dir]', 'True'),  # doc
    ('Module[{}, SetDirectory["sub"]; SetDirectory["inner"]; {ResetDirectory[], ResetDirectory[]} == '
     '{FileNameJoin[{w27dir, "sub"}], w27dir}]', 'True'),                                     # doc: earlier and earlier
    ('ResetDirectory[]', Raises(ValueError)),                                                 # empty stack
    ('Module[{}, SetDirectory["sub"]; ResetDirectory[]; FileExistsQ["a.txt"]]', 'True'),
    ('$HomeDirectory == w27dir', 'True'),                                                     # fixture sets HOME
    ('StringQ[$HomeDirectory] && DirectoryQ[$HomeDirectory]', 'True'),
    ('Module[{}, SetDirectory[$HomeDirectory]; Directory[] == $HomeDirectory]', 'True'),
    ('$TemporaryDirectory == FileNameJoin[{w27dir, "tmp"}]', 'True'),                        # fixture sets tempfile.tempdir
    ('DirectoryName[CreateFile[]] == DirectoryName[FileNameJoin[{$TemporaryDirectory, "x"}]]', 'True'),
    ('Block[{$TemporaryDirectory = FileNameJoin[{w27dir, "emptydir"}]}, '
     'ParentDirectory[CreateDirectory[]] == FileNameJoin[{w27dir, "emptydir"}]]', 'True'),
    ('$PathnameSeparator', Pred(lambda g: g == os.sep, "os.sep")),                            # Python os.sep
    ('StringLength[$PathnameSeparator]', '1'),
    ('FileNameJoin[{"a", "b"}] == "a" <> $PathnameSeparator <> "b"', 'True'),
    ('$ByteOrdering', Pred(lambda g: g == (1 if sys.byteorder == "big" else -1), "sys.byteorder")),
    ('MemberQ[{1, -1}, $ByteOrdering]', 'True'),                                              # doc: +1 or -1
    ('BinaryReadList[Module[{f = "bo.bin"}, BinaryWrite[f, 1, "Integer16"]; Close[f]; f]] == '
     'If[$ByteOrdering == 1, {0, 1}, {1, 0}]', 'True'),                                       # doc: default ByteOrdering
    ('$RootDirectory', Pred(lambda g: g == "/" if os.name != "nt" else (isinstance(g, str) and g.endswith("\\")),
                            "/ on Unix")),
    ('DirectoryQ[$RootDirectory]', 'True'),
    ('ParentDirectory[$RootDirectory] == $RootDirectory', 'True'),
    ('$InitialDirectory', Pred(lambda g: isinstance(g, str) and os.path.isabs(g), "absolute name")),
    ('DirectoryQ[$InitialDirectory]', 'True'),
    ('$InitialDirectory == ExpandFileName[$InitialDirectory]', 'True'),
    ('$UserBaseDirectory', Pred(lambda g: g == _user_base_ref(), "per-platform user base")),
    ('StringQ[$UserBaseDirectory] && FileNameDepth[$UserBaseDirectory] > FileNameDepth[$HomeDirectory]', 'True'),
    ('MemberQ[$Path, FileNameJoin[{$UserBaseDirectory, "Kernel"}]]', 'True'),                 # doc: subdirectories on $Path
    ('$BaseDirectory', Pred(lambda g: g == _base_ref(), "per-platform base")),
    ('MemberQ[$Path, FileNameJoin[{$BaseDirectory, "Applications"}]]', 'True'),
    ('Position[$Path, FileNameJoin[{$UserBaseDirectory, "Kernel"}]][[1, 1]] < '
     'Position[$Path, FileNameJoin[{$BaseDirectory, "Kernel"}]][[1, 1]]', 'True'),                    # doc: user dirs first
    ('$InstallationDirectory', Pred(lambda g: _same_path(g, _install_ref()), "package parent directory")),
    ('FileExistsQ[FileNameJoin[{$InstallationDirectory, "mikoshilang", "library", "wl_files.py"}]]', 'True'),
    ('DirectoryQ[$InstallationDirectory]', 'True'),
    ('$UserDocumentsDirectory == FileNameJoin[{$HomeDirectory, "Documents"}]', 'True'),       # doc: ~/Documents
    ('StringQ[$UserDocumentsDirectory]', 'True'),
    ('FileNameTake[$UserDocumentsDirectory] == "Documents"', 'True'),
    ('$WolframDocumentsDirectory == FileNameJoin[{$UserDocumentsDirectory, "Wolfram"}]', 'True'),  # doc
    ('FileNameTake[$WolframDocumentsDirectory] == "Wolfram"', 'True'),
    ('StringQ[$WolframDocumentsDirectory]', 'True'),
    ('First[$Path]', '"."'),
    ('Block[{$Path = {"sub"}}, FindFile["b.txt"] == FileNameJoin[{w27dir, "sub", "b.txt"}]]', 'True'),
    ('Block[{$Path = {FileNameJoin[{w27dir, "sub", "inner"}]}}, ReadList["c.m"]]', '{2}'),   # Read functions use $Path
    ('$InputFileName', '""'),                                                                 # doc: "" when not from a file
    ('Get["infile.m"] == FileNameJoin[{w27dir, "infile.m"}]', 'True'),                       # doc: set during <<file
    ('Module[{}, Get["infile.m"]; $InputFileName]', '""'),                                    # Check: restored after Get
    ('$StandardOutputStream', 'OutputStream["stdout", 1]'),                                   # doc
    ('$StandardErrorStream', 'OutputStream["stderr", 2]'),
    ('MemberQ[Streams[], $StandardOutputStream] && MemberQ[Streams[], $StandardErrorStream]', 'True'),  # doc: ContainsAll
    ('Module[{}, WriteString[$StandardOutputStream, ""]; StringQ[First[$StandardOutputStream]]]', 'True'),
    ('First[$StandardErrorStream]', '"stderr"'),
    ('Close[$StandardErrorStream]', Raises(ValueError)),
]


def _user_base_ref():
    h = os.environ["HOME"] if os.name != "nt" else os.environ["USERPROFILE"]
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA") or os.path.join(h, "AppData", "Roaming"), "Mikoshilang")
    if sys.platform == "darwin":
        return os.path.join(h, "Library", "Mikoshilang")
    return os.path.join(h, ".Mikoshilang")


def _base_ref():
    if os.name == "nt":
        return os.path.join(os.environ.get("PROGRAMDATA") or "C:\\ProgramData", "Mikoshilang")
    return "/Library/Mikoshilang" if sys.platform == "darwin" else "/usr/share/Mikoshilang"


def _install_ref():
    import mikoshilang
    return os.path.dirname(os.path.dirname(os.path.abspath(mikoshilang.__file__)))


# ─────────────────────────────── queries and file operations ───────────────────────────────

FILES = [
    ('FileExistsQ["a.txt"]', 'True'),
    ('FileExistsQ["nothere.txt"]', 'False'),
    ('FileExistsQ["sub"]', 'True'),                                                           # directories exist too
    ('FileExistsQ[File["sub/b.txt"]]', 'True'),
    ('FileExistsQ[w27x]', 'FileExistsQ[w27x]'),
    ('FileExistsQ[1]', Raises(TypeError)),
    ('Module[{}, CreateFile["fe.txt"]; {FileExistsQ["fe.txt"], DeleteFile["fe.txt"]; FileExistsQ["fe.txt"]}]',
     '{True, False}'),                                                                        # Check
    ('DirectoryQ["sub"]', 'True'),
    ('DirectoryQ["a.txt"]', 'False'),
    ('DirectoryQ["nothere"]', 'False'),
    ('DirectoryQ[File["emptydir"]]', 'True'),
    ('DirectoryQ[w27x]', 'DirectoryQ[w27x]'),
    ('DirectoryQ[2]', Raises(TypeError)),
    ('FileType["a.txt"]', 'File'),                                                            # doc
    ('FileType["sub"]', 'Directory'),
    ('FileType["nothere"]', 'None'),                                                          # doc: None if missing
    ('FileType[File["bin.dat"]]', 'File'),
    ('FileType[2.5]', Raises(TypeError)),
    ('FileByteCount["a.txt"]', '22'),                                                         # sample: 6 + 12 + 4 bytes
    ('FileByteCount["bin.dat"]', '4'),
    ('FileByteCount[File["empty.txt"]]', '0'),
    ('Module[{}, WriteString["u.txt", "\\:00e9"]; Close["u.txt"]; FileByteCount["u.txt"]]', '2'),  # UTF-8: e9 -> c3 a9
    ('FileByteCount["nothere"]', Raises(ValueError)),
    ('FileByteCount[{}]', Raises(TypeError)),
    ('FileSize["a.txt"]', 'Quantity[22, "Bytes"]'),
    ('FileSize[File["bin.dat"]]', 'Quantity[4, "Bytes"]'),
    ('First[FileSize["sub/b.txt"]] == FileByteCount["sub/b.txt"]', 'True'),                   # Check
    ('FileSize["nothere"]', Raises(ValueError)),
    ('Module[{}, SetFileDate["a.txt", {2020, 1, 2, 3, 4, 5}]; DateList[FileDate["a.txt"]]]',
     '{2020, 1, 2, 3, 4, 5.}'),                                                               # Check: round trip
    ('Module[{}, SetFileDate["a.txt", DateObject[{2019, 6, 7, 8, 9, 10}]]; '
     'AbsoluteTime[FileDate["a.txt", "Modification"]] == AbsoluteTime[DateObject[{2019, 6, 7, 8, 9, 10}]]]', 'True'),
    ('Module[{}, SetFileDate["a.txt", {2018, 3, 4, 5, 6, 7}, "Access"]; DateList[FileDate["a.txt", "Access"]]]',
     '{2018, 3, 4, 5, 6, 7.}'),
    ('Module[{r = FileDate["a.txt", "Rules"]}, {MemberQ[Keys[r], "Modification"], MemberQ[Keys[r], "Access"]}]',
     '{True, True}'),                                                                         # doc: "Rules"
    ('Head[FileDate[File["a.txt"]]]', 'DateObject'),                                          # doc: a DateObject
    ('FileDate["a.txt", "Birthday"]', Raises(ValueError)),
    ('FileDate["nothere"]', Raises(ValueError)),
    ('FileDate["a.txt", 3]', Raises(TypeError)),
    ('SetFileDate["a.txt", {2020, 1, 1, 0, 0, 0}]', Pred(lambda g: g is None, "Null")),
    ('Module[{}, SetFileDate["a.txt", {2021, 5, 6, 7, 8, 9}, "Modification"]; '
     'SetFileDate["a.txt", {2001, 1, 1, 0, 0, 0}, "Access"]; DateList[FileDate["a.txt"]]]',
     '{2021, 5, 6, 7, 8, 9.}'),                                                               # access-only keeps mtime
    ('Module[{t = AbsoluteTime[]}, SetFileDate["a.txt"]; Abs[AbsoluteTime[FileDate["a.txt"]] - t] < 5]', 'True'),
    ('SetFileDate["a.txt", {2020, 1, 1}, "Creation"]', Raises(NotImplementedError)),
    ('SetFileDate["a.txt", "yesterday"]', Raises(TypeError)),
    ('SetFileDate["a.txt", {2020, 1, 1}, "Sometime"]', Raises(ValueError)),
    ('SetFileDate["nothere", {2020, 1, 1}]', Raises(ValueError)),
    ('FilePrint["a.txt"]', Pred(lambda g: g is None, "Null")),
    ('FilePrint["a.txt", -1]', Pred(lambda g: g is None, "Null")),
    ('FilePrint[File["a.txt"], 1 ;; 3 ;; 2]', Pred(lambda g: g is None, "Null")),
    ('FilePrint["nothere"]', Raises(ValueError)),
    ('FilePrint["a.txt", "x"]', Raises(TypeError)),
    ('FilePrint["a.txt", 1 ;; 2 ;; 0]', Raises(ValueError)),
    # FileNames
    ('FileNames["*.txt"]', '{"a.txt", "crlf.txt", "empty.txt", "nums.txt"}'),
    ('FileNames["@.dat"]', '{"bin.dat"}'),                                                    # doc: @ = non-uppercase chars
    ('FileNames["*.TXT", IgnoreCase -> True]', '{"a.txt", "crlf.txt", "empty.txt", "nums.txt"}'),  # doc: IgnoreCase
    ('FileNames["*.TXT", IgnoreCase -> False]', '{}'),
    ('FileNames[{"a.*", "bin.*"}]', '{"a.txt", "bin.dat"}'),                                  # doc: list of forms
    ('FileNames["*", "sub"]', '{FileNameJoin[{"sub", "b.txt"}], FileNameJoin[{"sub", "inner"}]}'),
    ('FileNames["*", "sub", Infinity]',
     '{FileNameJoin[{"sub", "b.txt"}], FileNameJoin[{"sub", "inner"}], FileNameJoin[{"sub", "inner", "c.m"}]}'),
    ('FileNames["*.m", {"sub"}, {2}]', '{FileNameJoin[{"sub", "inner", "c.m"}]}'),            # doc: exactly level n
    ('FileNames[RegularExpression["[ab].*\\\\.(txt|dat)"]]', '{"a.txt", "bin.dat"}'),                   # doc: RegularExpression
    ('FileNames["sub/*.txt"]', '{FileNameJoin[{"sub", "b.txt"}]}'),                          # doc: forms with directories
    ('FileNames[All, File["emptydir"]]', '{}'),
    ('FileNames["*", "nothere"]', '{}'),
    ('Length[FileNames[]] == Length[FileNames[All]] == Length[FileNames["*"]]', 'True'),       # doc: equivalent forms
    ('FileNames["*/x/*.m"]', Raises(NotImplementedError)),
    ('FileNames["*", ".", 0]', Raises(ValueError)),
    ('FileNames[1]', Raises(TypeError)),
    ('FileNames["*", ".", w27n]', 'FileNames["*", ".", w27n]'),
    ('Sort[FileNames["*", "sub", Infinity]] == FileNames["*", "sub", Infinity]', 'True'),     # doc: sorted as by Sort
    # FindFile
    ('FindFile["a.txt"] == AbsoluteFileName["a.txt"]', 'True'),                               # "." is first on $Path
    ('FindFile["b.txt", Path -> {"sub"}] == FileNameJoin[{w27dir, "sub", "b.txt"}]', 'True'),  # doc: Path option
    ('FindFile["c.m", Path -> {{"nothere"}, FileNameJoin[{w27dir, "sub", "inner"}]}] == '
     'FileNameJoin[{w27dir, "sub", "inner", "c.m"}]', 'True'),                                # doc: nested sublists
    ('FindFile["Pkg`"] == FileNameJoin[{w27dir, "Pkg", "init.m"}]', 'True'),                 # doc: dir -> init.m
    ('FindFile["Pkg"] == FileNameJoin[{w27dir, "Pkg", "init.m"}]', 'True'),
    ('FindFile["nothere.m"]', '$Failed'),                                                     # doc: $Failed if not found
    ('FindFile[File[FileNameJoin[{w27dir, "a.txt"}]]] == FileNameJoin[{w27dir, "a.txt"}]', 'True'),
    ('FindFile[1]', Raises(TypeError)),
    ('FindFile[""]', Raises(ValueError)),
    ('FindFile["a.txt", Path -> 3]', Raises(TypeError)),
    ('FileExistsQ[FindFile["sub/inner/c.m"]]', 'True'),                                       # Check
    # FileSystemMap, FileSystemScan
    ('FileSystemMap[FileByteCount, "sub"]', '<|"b.txt" -> 2|>'),                              # sample bytes
    ('FileSystemMap[FileByteCount, "sub", Infinity]', '<|"b.txt" -> 2, "inner" -> <|"c.m" -> 4|>|>'),
    ('FileSystemMap[FileByteCount, "sub", 2, FileNameForms -> "*.m"]', '<|"inner" -> <|"c.m" -> 4|>|>'),  # doc
    ('FileSystemMap[FileExtension, "sub", {2}]', '<|"inner" -> <|"c.m" -> "m"|>|>'),          # doc: only level 2 files
    ('FileSystemMap[FileType, "sub", IncludeDirectories -> True]', '<|"b.txt" -> File, "inner" -> Directory|>'),
    ('FileSystemMap[FileType, "sub", 2, IncludeDirectories -> True]',
     '<|"b.txt" -> File, "inner" -> <|All -> Directory, "c.m" -> File|>|>'),                  # doc: key All
    ('FileSystemMap[FileByteCount, ".", MaxItems -> 1, FileNameForms -> "*.txt"]', '<|"a.txt" -> 22|>'),  # doc: MaxItems
    ('FileSystemMap[FileByteCount, File["emptydir"]]', '<||>'),
    ('FileSystemMap[f, "sub", 2, 1]', Raises(NotImplementedError)),
    ('FileSystemMap[f, "nothere"]', Raises(ValueError)),
    ('FileSystemMap[f, "sub", FileNameForms -> 3]', Raises(TypeError)),
    ('Values[FileSystemMap[FileNameTake, "sub"]] == Keys[FileSystemMap[FileNameTake, "sub"]]', 'True'),  # Check: full names
    ('Reap[FileSystemScan[Sow, "sub", Infinity]][[2, 1]] == '
     '{FileNameJoin[{w27dir, "sub", "b.txt"}], FileNameJoin[{w27dir, "sub", "inner", "c.m"}]}', 'True'),  # Check
    ('FileSystemScan[f, "sub"]', Pred(lambda g: g is None, "Null")),
    ('Reap[FileSystemScan[Sow, "sub", IncludeDirectories -> True]][[2, 1]] == '
     '{FileNameJoin[{w27dir, "sub", "b.txt"}], FileNameJoin[{w27dir, "sub", "inner"}]}', 'True'),
    ('Reap[FileSystemScan[Sow, ".", FileNameForms -> "*.dat"]][[2, 1]] == {FileNameJoin[{w27dir, "bin.dat"}]}', 'True'),
    ('FileSystemScan[f, "a.txt"]', Raises(ValueError)),
    ('FileSystemScan[f, "sub", IncludeDirectories -> 1]', Raises(ValueError)),
    # CreateFile, CreateDirectory
    ('CreateFile["new.txt"] == FileNameJoin[{w27dir, "new.txt"}]', 'True'),                   # doc: full name
    ('Module[{f = CreateFile["n2.txt"]}, {FileExistsQ[f], FileByteCount[f]}]', '{True, 0}'),  # Check
    ('Module[{f = CreateFile[File["x/y/z.txt"]]}, FileExistsQ[f] && DirectoryQ["x/y"]]', 'True'),  # doc: intermediates
    ('CreateFile["a.txt"]', Raises(ValueError)),
    ('Module[{}, CreateFile["a.txt", OverwriteTarget -> True]; FileByteCount["a.txt"]]', '0'),  # doc: truncates
    ('CreateFile["q/r.txt", CreateIntermediateDirectories -> False]', Raises(ValueError)),
    ('CreateFile["sub"]', Raises(ValueError)),
    ('Length[CreateFile[{"l1.txt", "l2.txt"}]]', '2'),                                        # doc: Listable
    ('Module[{f = CreateFile[]}, FileExistsQ[f] && DirectoryName[f] == DirectoryName[FileNameJoin[{$TemporaryDirectory, "x"}]]]',
     'True'),                                                                                 # doc: CreateFile[]
    ('CreateFile["c.txt", OverwriteTarget -> "yes"]', Raises(ValueError)),
    ('CreateFile[1]', Raises(TypeError)),
    ('CreateDirectory["newdir"] == FileNameJoin[{w27dir, "newdir"}]', 'True'),                # doc
    ('Module[{d = CreateDirectory["nd"]}, {DirectoryQ[d], FileNames[All, d]}]', '{True, {}}'),  # Check: empty
    ('Module[{d = CreateDirectory["p/q/r"]}, DirectoryQ["p/q"] && DirectoryQ[d]]', 'True'),
    ('CreateDirectory["s/t", CreateIntermediateDirectories -> False]', Raises(ValueError)),
    ('Module[{}, CreateDirectory["s2", CreateIntermediateDirectories -> False]; DirectoryQ["s2"]]', 'True'),
    ('CreateDirectory["sub"]', Raises(ValueError)),
    ('Module[{d = CreateDirectory[]}, DirectoryQ[d] && ParentDirectory[d] == $TemporaryDirectory]', 'True'),
    ('Length[CreateDirectory[{"m1", "m2", "m3"}]]', '3'),                                     # doc: Listable
    ('CreateDirectory[{1}]', Raises(TypeError)),
    # CopyFile, RenameFile, DeleteFile
    ('CopyFile["a.txt", "copy.txt"] == FileNameJoin[{w27dir, "copy.txt"}]', 'True'),          # doc: full name of dest
    ('Module[{}, CopyFile["a.txt", "copy.txt"]; ReadString["copy.txt"] == ReadString["a.txt"]]', 'True'),  # Check
    ('Module[{}, SetFileDate["a.txt", {2010, 1, 1, 0, 0, 0}]; CopyFile["a.txt", "c2.txt"]; '
     'DateList[FileDate["c2.txt"]]]', '{2010, 1, 1, 0, 0, 0.}'),                              # doc: date of file1
    ('CopyFile["a.txt", "bin.dat"]', Raises(ValueError)),                                     # doc: OverwriteTarget False
    ('Module[{}, CopyFile["a.txt", "bin.dat", OverwriteTarget -> True]; FileByteCount["bin.dat"]]', '22'),
    ('CopyFile["sub", "sub3"]', Raises(ValueError)),
    ('CopyFile["nothere", "x"]', Raises(ValueError)),
    ('CopyFile["a.txt", "a.txt", OverwriteTarget -> True]', Raises(ValueError)),
    ('CopyFile["a.txt", 7]', Raises(TypeError)),
    ('RenameFile["a.txt", "r.txt"] == FileNameJoin[{w27dir, "r.txt"}]', 'True'),              # doc
    ('Module[{}, RenameFile["a.txt", "r.txt"]; {FileExistsQ["a.txt"], FileByteCount["r.txt"]}]', '{False, 22}'),  # Check
    ('RenameFile["a.txt", "bin.dat"]', Raises(ValueError)),                                   # doc: file2 must not exist
    ('Module[{}, RenameFile["a.txt", "bin.dat", OverwriteTarget -> True]; FileByteCount["bin.dat"]]', '22'),
    ('RenameFile["sub", "sub4"]', Raises(ValueError)),
    ('RenameFile["nothere", "x"]', Raises(ValueError)),
    ('RenameFile[File["a.txt"], File["emptydir/a.txt"]] == FileNameJoin[{w27dir, "emptydir", "a.txt"}]', 'True'),
    ('DeleteFile["a.txt"]', Pred(lambda g: g is None, "Null")),                               # doc: Null
    ('Module[{}, DeleteFile[{"a.txt", File["bin.dat"]}]; {FileExistsQ["a.txt"], FileExistsQ["bin.dat"]}]',
     '{False, False}'),                                                                       # Check
    ('DeleteFile["nothere"]', Raises(ValueError)),
    ('{Check[DeleteFile[{"a.txt", "nothere"}], "failed"], FileExistsQ["a.txt"]}', '{"failed", True}'),  # nothing deleted
    ('DeleteFile["sub"]', Raises(ValueError)),
    ('DeleteFile[5]', Raises(TypeError)),
    # DeleteDirectory, CopyDirectory, RenameDirectory
    ('DeleteDirectory["emptydir"]', Pred(lambda g: g is None, "Null")),                       # doc: Null
    ('Module[{}, DeleteDirectory["emptydir"]; DirectoryQ["emptydir"]]', 'False'),             # Check
    ('DeleteDirectory["sub"]', Raises(ValueError)),                                           # doc: only if empty
    ('{Check[DeleteDirectory["sub"], "failed"], DirectoryQ["sub"], FileExistsQ["sub/b.txt"]}',
     '{"failed", True, True}'),                                                               # nothing deleted
    ('Module[{}, DeleteDirectory["sub", DeleteContents -> True]; {DirectoryQ["sub"], FileExistsQ["sub/inner/c.m"]}]',
     '{False, False}'),                                                                       # doc: DeleteContents
    ('Module[{}, DeleteDirectory[File["emptydir"], DeleteContents -> False]; DirectoryQ["emptydir"]]', 'False'),
    ('DeleteDirectory["sub", DeleteContents -> "yes"]', Raises(ValueError)),
    ('DeleteDirectory["a.txt"]', Raises(ValueError)),
    ('DeleteDirectory["nothere"]', Raises(ValueError)),
    ('DeleteDirectory[{"sub"}]', Raises(TypeError)),
    ('CopyDirectory["sub", "subcopy"] == FileNameJoin[{w27dir, "subcopy"}]', 'True'),         # doc: full name
    ('Module[{}, CopyDirectory["sub", "sc"]; {FileNames["*", "sc", Infinity], ReadString["sc/inner/c.m"]}]',
     '{{FileNameJoin[{"sc", "b.txt"}], FileNameJoin[{"sc", "inner"}], FileNameJoin[{"sc", "inner", "c.m"}]}, "1+1\\n"}'),  # Check
    ('CopyDirectory["sub", "emptydir"]', Raises(ValueError)),                                 # doc: dir2 must not exist
    ('CopyDirectory["nothere", "x"]', Raises(ValueError)),
    ('CopyDirectory["sub", "sub/inner/deeper"]', Raises(ValueError)),
    ('CopyDirectory["sub", 1]', Raises(TypeError)),
    ('RenameDirectory["sub", "renamed"] == FileNameJoin[{w27dir, "renamed"}]', 'True'),       # doc
    ('Module[{}, RenameDirectory["sub", "rn"]; {DirectoryQ["sub"], FileExistsQ["rn/inner/c.m"]}]', '{False, True}'),  # Check
    ('RenameDirectory["sub", "emptydir"]', Raises(ValueError)),                               # doc: dir2 must not exist
    ('RenameDirectory["a.txt", "zz"]', Raises(ValueError)),
    ('RenameDirectory["sub", "sub/x"]', Raises(ValueError)),
    ('RenameDirectory[1, "x"]', Raises(TypeError)),
]

# ─────────────────────────────── streams ───────────────────────────────

STREAMS_CASES = [
    # OpenRead, Close, Streams
    ('Module[{s = OpenRead["a.txt"]}, {Head[s], First[s] == FileNameJoin[{w27dir, "a.txt"}], Close[s] == First[s]}]',
     '{InputStream, True, True}'),                                                            # doc: InputStream["name", n]
    ('Module[{s = OpenRead[File["a.txt"]], t = OpenRead["a.txt"]}, {Last[t] > Last[s], Close[s]; Close[t]; True}]',
     '{True, True}'),                                                                         # doc: unique serial numbers
    ('Module[{s = OpenRead["a.txt", BinaryFormat -> True]}, {Read[s, Byte], Close[s]; 1}]', '{49, 1}'),  # "1" = 49
    ('Module[{s = OpenRead["c.m", Path -> {"sub/inner"}]}, {Read[s], Close[s]; 1}]', '{2, 1}'),  # Path option
    ('OpenRead["nothere.txt"]', Raises(ValueError)),                                          # doc: missing file
    ('OpenRead["sub"]', Raises(ValueError)),
    ('OpenRead["!ls"]', Raises(NotImplementedError)),
    ('OpenRead["a.txt", Method -> "NoSuchMethod"]', Raises(ValueError)),
    ('OpenRead["a.txt", CharacterEncoding -> "EBCDIC"]', Raises(NotImplementedError)),
    ('OpenRead["a.txt", BinaryFormat -> 2]', Raises(ValueError)),
    ('OpenRead[3]', Raises(TypeError)),
    ('Module[{s = OpenRead["a.txt"]}, {MemberQ[Streams[], s], Close[s]; MemberQ[Streams[], s]}]', '{True, False}'),  # Check
    ('Module[{s = OpenRead["a.txt"]}, Close[s]; Close[s]]', Raises(ValueError)),             # closed stream
    ('Module[{}, Read["a.txt", Number]; Close["a.txt"] == FileNameJoin[{w27dir, "a.txt"}]]', 'True'),  # doc: Close["name"]
    ('Close["a.txt"]', Raises(ValueError)),                                                   # no stream of that name
    ('Close[StringToStream["x"]]', 'String'),
    ('Close[OutputStream["stdout", 1]]', Raises(ValueError)),
    ('Close[1]', Raises(TypeError)),
    ('Take[Streams[], 2]', '{OutputStream["stdout", 1], OutputStream["stderr", 2]}'),         # doc: standard streams
    ('Module[{s = OpenWrite["w.txt"]}, {Streams[FileNameJoin[{w27dir, "w.txt"}]] == {s}, Close[s]; '
     'Streams[FileNameJoin[{w27dir, "w.txt"}]]}]', '{True, {}}'),                             # Check
    ('Streams["stdout"]', '{OutputStream["stdout", 1]}'),
    ('Streams[1]', Raises(TypeError)),
    # OpenWrite, OpenAppend, Write, WriteString, WriteLine
    ('Module[{s = OpenWrite["w.txt"]}, Write[s, x^2 + 1]; Close[s]; ReadList["w.txt", String]]', '{"1 + x^2"}'),
    ('Module[{s = OpenWrite["w.txt"]}, Write[s, a, b, "c"]; Close[s]; ReadString["w.txt"]]', '"ab\\"c\\"\\n"'),  # doc: in sequence
    ('Module[{s = OpenWrite[]}, Close[s]; DirectoryName[s[[1]]] == DirectoryName[FileNameJoin[{$TemporaryDirectory, "x"}]]]',
     'True'),                                                                                 # doc: OpenWrite[]
    ('Module[{s = OpenWrite["a.txt"]}, Close[s]; FileByteCount["a.txt"]]', '0'),              # doc: deletes contents
    ('OpenWrite["a.txt", PageWidth -> 70]', Raises(NotImplementedError)),
    ('OpenWrite["nothere/x.txt"]', Raises(ValueError)),
    ('OpenWrite["sub"]', Raises(ValueError)),
    ('OpenWrite["a.txt", FormatType -> TraditionalForm]', Raises(NotImplementedError)),
    ('Module[{s = OpenWrite["o.txt", FormatType -> OutputForm]}, Write[s, "q"]; Close[s]; ReadString["o.txt"]]', '"q\\n"'),
    ('Module[{s = OpenWrite["e.txt", CharacterEncoding -> "ASCII"]}, WriteString[s, "\\:03b1"]; Close[s]; '
     'ReadString["e.txt"]]', '"\\\\:03b1"'),                                                  # ASCII escapes
    ('Module[{s = OpenAppend["a.txt"]}, WriteString[s, "end\\n"]; Close[s]; Last[ReadList["a.txt", String]]]', '"end"'),
    ('Module[{s = OpenAppend["fresh.txt"]}, WriteLine[s, "one"]; Close[s]; ReadList["fresh.txt", String]]', '{"one"}'),
    ('Module[{s = OpenAppend[File["a.txt"]]}, {Head[s], Close[s]; FileByteCount["a.txt"]}]', '{OutputStream, 22}'),  # Check
    ('OpenAppend["nothere/x"]', Raises(ValueError)),
    ('OpenAppend[2]', Raises(TypeError)),
    ('Module[{}, Write["wf.txt", 1/2]; Write["wf.txt", {1, 2}]; Close["wf.txt"]; ReadList["wf.txt"]]', '{1/2, {1, 2}}'),  # Check: Write/ReadList
    ('Module[{}, Write[File["wf.txt"], a + b]; Close["wf.txt"]; Read["wf.txt", {Word, Word, Word}]]',
     '{"a", "+", "b"}'),                                                                      # doc: Read scope example
    ('Module[{}, Write[{"w1.txt", "w2.txt"}, 7]; Close["w1.txt"]; Close["w2.txt"]; {ReadString["w1.txt"], ReadString["w2.txt"]}]',
     '{"7\\n", "7\\n"}'),                                                                     # doc: list of channels
    ('Write[$StandardOutputStream, 1 + 1]', Pred(lambda g: g is None, "Null")),
    ('Write[3, x]', Raises(TypeError)),
    ('Write[StringToStream["x"], 1]', Raises(ValueError)),
    ('Module[{}, WriteString["ws.txt", "ab", 1 + 2, "c"]; Close["ws.txt"]; ReadString["ws.txt"]]', '"ab3c"'),  # doc: OutputForm
    ('Module[{}, WriteString["ws.txt", "x"]; WriteString["ws.txt", "y"]; Close["ws.txt"]; FileByteCount["ws.txt"]]', '2'),  # no newline
    ('Module[{}, WriteString["a.txt", "z"]; Close["a.txt"]; ReadString["a.txt"]]', '"z"'),   # doc: erases if not open
    ('WriteString[1, "x"]', Raises(TypeError)),
    ('Module[{}, WriteLine["wl.txt", "first"]; WriteLine["wl.txt", "second"]; Close["wl.txt"]; ReadList["wl.txt", String]]',
     '{"first", "second"}'),                                                                  # Check: ReadLine/ReadList
    ('Module[{}, WriteLine[File["wl.txt"], ""]; Close["wl.txt"]; FileByteCount["wl.txt"]]', '1'),
    ('WriteLine["wl.txt", 5]', Raises(TypeError)),
    ('WriteLine["wl.txt", w27x]', 'WriteLine["wl.txt", w27x]'),
    # Read
    ('Read[StringToStream["abcdefg 123456"]]', '123456 abcdefg'),                             # doc: Read basic example
    ('Module[{s = StringToStream["123 abc 456"]}, {Read[s, Number], Read[s, {Word, Number}]}]',
     '{123, {"abc", 456}}'),                                                                  # doc: Read basic example
    ('Module[{s = StringToStream["12.3 456 789"]}, Table[Read[s, Number], {4}]]',
     '{12.3, 456, 789, EndOfFile}'),                                                          # doc: Number example
    ('Module[{s = StringToStream["a\\:03b1\\:2135"]}, Table[Read[s, Byte], {3}]]', '{97, 945, 8501}'),  # doc: possible issues
    ('Module[{s = StringToStream["xy"]}, {Read[s, Character], Read[s, Character], Read[s, Character]}]',
     '{"x", "y", EndOfFile}'),
    ('Read[StringToStream["2.5e3"], Real]', '2500.'),
    ('Read[StringToStream["7"], Real]', '7.'),                                                # doc: Real always approximate
    ('Read[StringToStream["2*^3"], Number]', '2000'),                                         # doc: no decimal point: integer
    ('Read[StringToStream["1.5D2"], Number]', '150.'),                                        # doc: Fortran form
    ('Read[StringToStream["1+1"], Hold[Expression]]', 'Hold[1 + 1]'),                         # doc: Hold[Expression]
    ('Read[StringToStream["{1,\\n 2}\\n3"]]', '{1, 2}'),                                      # expression over lines
    ('Read[StringToStream["one two\\nthree"], Record]', '"one two"'),
    ('Read[StringToStream["one two\\nthree"], String]', '"one two"'),
    ('Read[StringToStream["a,b;c"], {Word, Word}, WordSeparators -> {",", ";"}]', '{"a", "b"}'),
    ('Read[StringToStream["1 2"], {{Number, Number}, {Number, Number}}]', '{{1, 2}, {EndOfFile, EndOfFile}}'),  # doc: depth first
    ('Read[StringToStream[""], Word]', 'EndOfFile'),                                          # doc: EndOfFile
    ('Module[{}, Read["a.txt", Number]; Read["a.txt", Number]]', '2'),                        # doc: file kept open
    ('Read[OpenRead["nums.txt"], {Number, Number, Number, Number, Number}]', '{1, 2.5, -300, 400, 0.5}'),  # C/Fortran forms
    ('Read[StringToStream["abc"], Number]', Raises(ValueError)),                              # doc: $Failed -> raises
    ('Read[StringToStream["f[1,"]]', Raises(ValueError)),                                      # syntax error
    ('Read[StringToStream["x"], Foo]', Raises(ValueError)),
    ('Read[StringToStream["x"], 1]', Raises(TypeError)),
    ('Read[StringToStream["x"], Word, RecordSeparators -> {{"<", ">"}}]', Raises(NotImplementedError)),
    ('Read[w27s]', 'Read[w27s]'),
    ('Module[{s = StringToStream["10 20"]}, Read[s, Number] + Read[s, Number]]', '30'),       # Check: successive reads
    # ReadList
    ('ReadList[StringToStream["a b\\nc d\\n\\ne"], Word]', '{"a", "b", "c", "d", "e"}'),      # doc: whitespace incl. newlines
    ('ReadList[StringToStream["a b\\nc d\\n\\ne"], Word, RecordLists -> True]', '{{"a", "b"}, {"c", "d"}, {"e"}}'),  # doc: RecordLists
    ('ReadList[StringToStream["1 2\\n3"], Number, RecordLists -> True]', '{{1, 2}, {3}}'),
    ('ReadList[StringToStream["1 2\\n3"], Number, RecordLists -> False]', '{1, 2, 3}'),
    ('ReadList[StringToStream["a\\n\\nb"], String]', '{"a", "", "b"}'),
    ('ReadList[StringToStream["a\\n\\nb"], Record]', '{"a", "b"}'),
    ('ReadList[StringToStream["a\\n\\nb"], Record, NullRecords -> True]', '{"a", "", "b"}'),   # doc: NullRecords
    ('ReadList[StringToStream["a,,b"], Word, WordSeparators -> ","]', '{"a", "b"}'),
    ('ReadList[StringToStream["a,,b"], Word, WordSeparators -> ",", NullWords -> True]', '{"a", "", "b"}'),  # doc: NullWords
    ('ReadList[StringToStream["a+b c"], Word, TokenWords -> {"+"}]', '{"a", "+", "b", "c"}'),  # doc: TokenWords
    ('ReadList[StringToStream["1 2 3 4 5"], {Number, Number}]', '{{1, 2}, {3, 4}, {5, EndOfFile}}'),  # doc: partial
    ('ReadList[StringToStream["1 2 3 4 5"], Number, 3]', '{1, 2, 3}'),
    ('ReadList["a.txt", Record]', '{"1 2 3", "hello world", "x+y"}'),                         # sample file
    ('ReadList["a.txt"]', '{6, hello world, x + y}'),                                         # expressions, evaluated
    ('ReadList[File["sub/b.txt"], Character]', '{"B", "\\n"}'),
    ('Module[{}, ReadList["a.txt"]; Select[Streams[], First[#] == FileNameJoin[{w27dir, "a.txt"}] &]]', '{}'),  # doc: closed
    ('Module[{s = OpenRead["a.txt"]}, Read[s, Record]; {ReadList[s, Record], MemberQ[Streams[], s]}]',
     '{{"hello world", "x+y"}, True}'),                                                       # doc: open stream kept
    ('ReadList["crlf.txt", String]', '{"a", "b"}'),
    ('ReadList["empty.txt"]', '{}'),
    ('ReadList["a.txt", Number]', Raises(ValueError)),                                        # doc: objects not of the type
    ('ReadList["nothere.txt"]', Raises(ValueError)),
    ('ReadList["a.txt", Word, -1]', Raises(ValueError)),
    ('ReadList["a.txt", Word, RecordLists -> 1]', Raises(ValueError)),
    ('ReadList[StringToStream["x"], {}]', Raises(ValueError)),
    ('ReadList["a.txt", "Word"]', Raises(TypeError)),
    # Skip
    ('Module[{s = StringToStream["a b c"]}, Skip[s, Word]; Read[s, Word]]', '"b"'),          # doc: like Read
    ('Module[{s = StringToStream["1 2 3 4"]}, Skip[s, Number, 2]; ReadList[s, Number]]', '{3, 4}'),
    ('Skip[StringToStream["a"], Word]', Pred(lambda g: g is None, "Null")),                   # doc: Null
    ('Skip[StringToStream["a"], Word, 2]', 'EndOfFile'),
    ('Module[{s = StringToStream["x y z"]}, Skip[s, Word, 1]; ReadList[s, Word] == Rest[ReadList[StringToStream["x y z"], Word]]]',
     'True'),                                                                                 # Check
    ('Skip[StringToStream["a"], Word, -2]', Raises(ValueError)),
    ('Skip[StringToStream["a"], 7]', Raises(TypeError)),
    # ReadLine, ReadString, ReadByteArray
    ('Module[{}, {ReadLine["a.txt"], ReadLine["a.txt"], ReadLine["a.txt"], ReadLine["a.txt"]}]',
     '{"1 2 3", "hello world", "x+y", EndOfFile}'),                                           # doc: EndOfFile when exhausted
    ('ReadLine["crlf.txt"]', '"a"'),                                                          # doc: "\r\n" newlines
    ('ReadLine[StringToStream["no newline"]]', '"no newline"'),
    ('Module[{s = OpenRead["a.txt"], t}, t = Table[ReadLine[s], {3}]; Close[s]; t == ReadList["a.txt", String]]', 'True'),  # Check
    ('ReadLine[StringToStream[""]]', 'EndOfFile'),
    ('ReadLine[1]', Raises(TypeError)),
    ('ReadString["a.txt"]', '"1 2 3\\nhello world\\nx+y\\n"'),
    ('ReadString[StringToStream["ab;cd;ef"], ";"]', '"ab"'),                                  # doc: up to the terminator
    ('Module[{s = StringToStream["ab;cd;ef"]}, {ReadString[s, ";"], ReadString[s, ";"], ReadString[s, ";"], ReadString[s]}]',
     '{"ab", "cd", "ef", EndOfFile}'),                                                        # doc: exhausted -> EndOfFile
    ('ReadString[StringToStream["ab12cd"], DigitCharacter ..]', '"ab"'),                      # doc: string pattern
    ('ReadString[StringToStream["rest"], EndOfBuffer]', '"rest"'),                             # doc: EndOfBuffer
    ('Module[{s = StringToStream["x\\ny"]}, ReadString[s, "\\n" | "\\r\\n"] == ReadLine[StringToStream["x\\ny"]]]', 'True'),  # doc: ReadLine equivalence
    ('Module[{}, WriteString["rs.txt", "q\\:03b1z"]; Close["rs.txt"]; ReadString["rs.txt"]]', '"q\\:03b1z"'),  # Check: round trip
    ('ReadString[StringToStream[""]]', 'EndOfFile'),
    ('ReadString[StringToStream["x"], ""]', Raises(ValueError)),
    ('ReadString[StringToStream["x"], {1}]', Raises(TypeError)),
    ('ReadString[StringToStream["x"], TimeConstraint -> 1]', Raises(NotImplementedError)),
    ('ReadByteArray["bin.dat"]', 'ByteArray[{1, 2, 3, 255}]'),                                # sample bytes
    ('Module[{s = OpenRead["bin.dat", BinaryFormat -> True]}, {ReadByteArray[s, 2], ReadByteArray[s, EndOfBuffer], ReadByteArray[s]}]',
     '{ByteArray[{1, 2}], ByteArray[{3, 255}], EndOfFile}'),                                  # doc: n bytes, EndOfFile
    ('ReadByteArray[StringToStream["ab;cd"], ByteArray[{59}]]', 'ByteArray[{97, 98}]'),       # doc: terminator dropped
    ('ReadByteArray["crlf.txt", EndOfLine]', 'ByteArray[{97}]'),                               # doc: EndOfLine
    ('ReadByteArray[StringToStream["a|b"], {ByteArray[{35}], ByteArray[{124}]}]', 'ByteArray[{97}]'),
    ('Module[{}, BinaryWrite["rb.bin", {0, 128, 255}]; Close["rb.bin"]; Normal[ReadByteArray["rb.bin"]]]',
     '{0, 128, 255}'),                                                                        # Check: round trip
    ('ReadByteArray[StringToStream["\\:03b1"]]', Raises(ValueError)),
    ('ReadByteArray["bin.dat", -1]', Raises(ValueError)),
    ('ReadByteArray["bin.dat", "x"]', Raises(TypeError)),
    # StreamPosition, SetStreamPosition
    ('Module[{s = OpenRead["a.txt"]}, {StreamPosition[s], Read[s, Record]; StreamPosition[s], Close[s]; 0}]',
     '{0, 6, 0}'),                                                                            # bytes: "1 2 3\n"
    ('Module[{s = StringToStream["abc"]}, Read[s, Character]; StreamPosition[s]]', '1'),
    ('Module[{s = OpenWrite["sp.txt"]}, WriteString[s, "abcd"]; {StreamPosition[s], Close[s]; 0}]', '{4, 0}'),
    ('StreamPosition[$StandardOutputStream]', Raises(NotImplementedError)),
    ('StreamPosition[3]', Raises(TypeError)),
    ('Module[{s = OpenRead["a.txt"]}, {Read[s, Word], SetStreamPosition[s, 0], Read[s, Word]}]', '{"1", 0, "1"}'),  # doc: 0 = start
    ('Module[{s = OpenRead["a.txt"]}, {SetStreamPosition[s, Infinity], Read[s, Word]}]', '{22, EndOfFile}'),  # doc: Infinity = end
    ('Module[{s = StringToStream["abcdef"]}, SetStreamPosition[s, 3]; Read[s, Word]]', '"def"'),
    ('Module[{s = OpenRead["a.txt"], p}, Read[s, Record]; p = StreamPosition[s]; Read[s, Record]; SetStreamPosition[s, p]; Read[s, Record]]',
     '"hello world"'),                                                                        # Check: reposition
    ('SetStreamPosition[OpenRead["a.txt"], 100]', Raises(ValueError)),
    ('SetStreamPosition[OpenRead["a.txt"], -1]', Raises(ValueError)),
    ('SetStreamPosition[OpenRead["a.txt"], "x"]', Raises(TypeError)),
    ('SetStreamPosition[$StandardOutputStream, 0]', Raises(NotImplementedError)),
]

# ─────────────────────────────── binary I/O ───────────────────────────────

BINARY = [
    ('Module[{}, BinaryWrite["f.bin", 1.0, "Real64", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     '{63, 240, 0, 0, 0, 0, 0, 0}'),                                                          # IEEE binary64 1.0
    ('Module[{}, BinaryWrite["f.bin", 0.1, "Real32", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     '{61, 204, 204, 205}'),                                                                  # IEEE binary32 0.1 = 3DCCCCCD
    ('Module[{}, BinaryWrite["f.bin", 1/10, "Real32", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     '{61, 204, 204, 205}'),                                                                  # exact 1/10 rounds the same
    ('Module[{}, BinaryWrite["f.bin", 1, "Real128", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     '{63, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}'),                                  # IEEE binary128 1.0
    ('Module[{}, BinaryWrite["f.bin", -2, "Real128", ByteOrdering -> 1]; Close["f.bin"]; Take[BinaryReadList["f.bin"], 2]]',
     '{192, 0}'),                                                                             # binary128 -2.0 = C000...
    ('Module[{}, BinaryWrite["f.bin", Infinity, "Real64", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     '{127, 240, 0, 0, 0, 0, 0, 0}'),                                                         # IEEE +inf
    ('Module[{}, BinaryWrite["f.bin", -Infinity, "Real32", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     '{255, 128, 0, 0}'),                                                                     # IEEE binary32 -inf
    ('Module[{}, BinaryWrite["f.bin", 258, "Integer16", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     'IntegerDigits[258, 256]'),                                                              # doc: +1 = IntegerDigits order
    ('Module[{}, BinaryWrite["f.bin", 258, "Integer16", ByteOrdering -> -1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     'Reverse[IntegerDigits[258, 256]]'),                                                     # doc: -1 = least significant first
    ('Module[{}, BinaryWrite["f.bin", -1, "Integer32"]; Close["f.bin"]; BinaryReadList["f.bin"]]', '{255, 255, 255, 255}'),
    ('Module[{}, BinaryWrite["f.bin", 2^100, "UnsignedInteger128", ByteOrdering -> 1]; Close["f.bin"]; '
     'BinaryReadList["f.bin"]]', 'PadLeft[IntegerDigits[2^100, 256], 16]'),
    ('Module[{}, BinaryWrite["f.bin", "AB"]; Close["f.bin"]; BinaryReadList["f.bin"]]', '{65, 66}'),  # doc: Character8
    ('Module[{}, BinaryWrite["f.bin", ByteArray[{9, 8}]]; Close["f.bin"]; BinaryReadList["f.bin"]]', '{9, 8}'),
    ('Module[{}, BinaryWrite["f.bin", {1, 2, 3, 4}, {"Byte", "Integer8"}]; Close["f.bin"]; '
     'BinaryReadList["f.bin", {"Byte", "Integer8"}]]', '{{1, 2}, {3, 4}}'),                   # doc: types repeated
    ('Module[{}, BinaryWrite["f.bin", "hi", "TerminatedString"]; Close["f.bin"]; BinaryReadList["f.bin"]]', '{104, 105, 0}'),
    ('Module[{}, BinaryWrite["f.bin", "\\:03b1", "Character16", ByteOrdering -> 1]; Close["f.bin"]; BinaryReadList["f.bin"]]',
     '{3, 177}'),                                                                             # U+03B1
    ('Module[{s = OpenWrite["f.bin", BinaryFormat -> True]}, {BinaryWrite[s, {7}] === s, Close[s]; BinaryReadList["f.bin"]}]',
     '{True, {7}}'),                                                                          # returns the channel
    ('BinaryWrite["f.bin", 256]', Raises(ValueError)),                                        # doc: $Failed -> raises
    ('BinaryWrite["f.bin", 1.5, "Integer32"]', Raises(TypeError)),
    ('BinaryWrite["f.bin", 1.0*^39, "Real32"]', Raises(ValueError)),
    ('BinaryWrite["f.bin", "x", "Real64"]', Raises(TypeError)),
    ('BinaryWrite["f.bin", 1, "Int99"]', Raises(ValueError)),
    ('BinaryWrite["f.bin", 1, ByteOrdering -> 2]', Raises(ValueError)),
    ('BinaryWrite["f.bin", w27x, "Integer32"]', 'BinaryWrite["f.bin", w27x, "Integer32"]'),
    ('BinaryRead["bin.dat"]', '1'),                                                           # sample first byte
    ('Module[{}, {BinaryRead["bin.dat"], BinaryRead["bin.dat"], BinaryRead["bin.dat", "Integer8"]}]', '{1, 2, 3}'),  # doc: kept open
    ('BinaryRead["bin.dat", "UnsignedInteger32", ByteOrdering -> 1]', '16909311'),            # 0x010203FF
    ('BinaryRead["bin.dat", "Integer32", ByteOrdering -> -1]', '-16580095'),                  # 0xFF030201 signed
    ('BinaryRead["bin.dat", {"Integer16", "Byte"}, ByteOrdering -> 1]', '{258, 3}'),
    ('BinaryRead["bin.dat", "Integer24", ByteOrdering -> 1]', '66051'),                       # 0x010203
    ('BinaryRead[File["empty.txt"]]', 'EndOfFile'),                                           # doc: EndOfFile at end
    ('Module[{s = OpenRead["bin.dat", BinaryFormat -> True]}, {BinaryRead[s, "Real64"], BinaryRead[s, "Integer32", ByteOrdering -> 1]}]',
     '{EndOfFile, 16909311}'),                                                                # incomplete object not consumed
    ('Module[{}, BinaryWrite["f.bin", {Infinity, Indeterminate}, "Real64"]; Close["f.bin"]; BinaryReadList["f.bin", "Real64"]]',
     '{Infinity, Indeterminate}'),                                                            # doc: IEEE inf/NaN
    ('Module[{}, BinaryWrite["f.bin", N[Pi, 40], "Real128"]; Close["f.bin"]; Abs[BinaryRead["f.bin", "Real128"] - N[Pi, 40]] < 10^-33]',
     'True'),                                                                                 # Check: binary128 ~34 digits
    ('Module[{}, BinaryWrite["f.bin", 1.5 - 2.25 I, "Complex128"]; Close["f.bin"]; BinaryRead["f.bin", "Complex128"]]',
     '1.5 - 2.25 I'),
    ('Module[{}, BinaryWrite["f.bin", 3 + 4 I, "Complex64"]; Close["f.bin"]; BinaryRead["f.bin", "Complex64"]]', '3. + 4. I'),
    ('Module[{}, BinaryWrite["f.bin", "ok", "TerminatedString"]; Close["f.bin"]; BinaryRead["f.bin", "TerminatedString"]]', '"ok"'),
    ('Module[{}, BinaryWrite["f.bin", "\\:03b1", "Character32"]; Close["f.bin"]; BinaryRead["f.bin", "Character32"]]', '"\\:03b1"'),
    ('BinaryRead["bin.dat", "Real256"]', Raises(ValueError)),
    ('BinaryRead["bin.dat", 3]', Raises(TypeError)),
    ('BinaryRead["bin.dat", Byte]', 'BinaryRead["bin.dat", Byte]'),                           # symbolic type
    ('BinaryReadList["bin.dat"]', '{1, 2, 3, 255}'),                                          # sample bytes
    ('BinaryReadList[File["bin.dat"], "Integer16", ByteOrdering -> 1]', '{258, 1023}'),       # 0x0102, 0x03FF
    ('BinaryReadList["bin.dat", "Byte", 2]', '{1, 2}'),
    ('BinaryReadList["bin.dat", {"Integer16", "Integer16", "Integer16"}, ByteOrdering -> 1]', '{{258, 1023, EndOfFile}}'),  # doc
    ('BinaryReadList["bin.dat", "Real64"]', '{}'),
    ('BinaryReadList["bin.dat", "Byte", Path -> {"nothere"}]', '{1, 2, 3, 255}'),             # existing name wins
    ('BinaryReadList["c.m", "Byte", Path -> {"sub/inner"}]', 'ToCharacterCode["1+1\\n"]'),     # doc: Path option
    ('Module[{}, BinaryWrite["rt.bin", {-5, 70000}, "Integer32"]; Close["rt.bin"]; BinaryReadList["rt.bin", "Integer32"]]',
     '{-5, 70000}'),                                                                          # Check: round trip
    ('BinaryReadList["bin.dat", "Byte", -1]', Raises(ValueError)),
    ('BinaryReadList["nothere.bin"]', Raises(ValueError)),
    ('BinaryReadList["bin.dat", 2]', Raises(TypeError)),
]

# ─────────────────────────────── user-defined stream methods ───────────────────────────────

_STRINGNAME = ('DefineInputStreamMethod["w27StringName", {"ConstructorFunction" -> ({True, <|"Pos" -> 1, '
               '"Text" -> ToCharacterCode[#]|>} &), "ReadFunction" -> (With[{pos = #["Pos"]}, '
               'Replace[Take[#["Text"], {pos, Min[pos + #2 - 1, Length[#["Text"]]]}], '
               'bytes_List :> {bytes, Append[#, "Pos" -> pos + Length[bytes]]}]] &)}]')
_TEXTNAME = ('DefineInputStreamMethod["w27TextName", {"NameTestFunction" -> StringStartsQ["text:"], '
             '"ConstructorFunction" -> ({True, <|"Pos" -> 1, "Text" -> ToCharacterCode[StringDrop[#, 5]]|>} &), '
             '"ReadFunction" -> (With[{pos = #["Pos"]}, Replace[Take[#["Text"], {pos, Min[pos + #2 - 1, '
             'Length[#["Text"]]]}], bytes_List :> {bytes, Append[#, "Pos" -> pos + Length[bytes]]}]] &)}]')
_COLLECT = ('DefineOutputStreamMethod["w27Collect", {"ConstructorFunction" -> ({True, 0} &), '
            '"WriteFunction" -> Function[{st, b}, Sow[b]; {Length[b], st + Length[b]}]}]')

METHODS = [
    # doc: DefineInputStreamMethod basic example ("StringName"; UpTo replaced by Min, same bytes)
    (f'Module[{{}}, {_STRINGNAME}; Module[{{s = OpenRead["Hello, world.\\nHere is text.", Method -> "w27StringName"]}}, '
     '{ReadList[s, String], Close[s]}]]', '{{"Hello, world.", "Here is text."}, "Hello, world.\\nHere is text."}'),
    # doc: "TextName" scope example (NameTestFunction with Method -> Automatic)
    (f'Module[{{}}, {_TEXTNAME}; Module[{{s = OpenRead["text:Hello, world.\\nHere is text."]}}, '
     '{ReadList[s, String], Close[s]; 1}]]', '{{"Hello, world.", "Here is text."}, 1}'),
    (f'Module[{{}}, {_STRINGNAME}; Module[{{s = OpenRead["ab cd", Method -> {{"w27StringName", "Extra" -> 1}}]}}, '
     '{Read[s, Word], StreamPosition[s], SetStreamPosition[s, 0], Read[s, Word]}]]', '{"ab", 2, 0, "ab"}'),  # Check
    (f'Module[{{}}, {_STRINGNAME}; MemberQ[$InputStreamMethods, "w27StringName"]]', 'True'),    # doc: $InputStreamMethods
    (f'Module[{{}}, {_STRINGNAME}; {_STRINGNAME}]', Raises(ValueError)),                         # doc: same name fails
    ('DefineInputStreamMethod["w27Bad", {"ReadFunction" -> f}]', Raises(ValueError)),           # doc: constructor required
    ('DefineInputStreamMethod["w27Bad", {"Frobnicate" -> f, "ConstructorFunction" -> g}]', Raises(ValueError)),
    ('DefineInputStreamMethod["w27Bad", "ConstructorFunction"]', Raises(TypeError)),
    ('DefineInputStreamMethod[w27x, {}]', 'DefineInputStreamMethod[w27x, {}]'),
    ('Module[{}, DefineInputStreamMethod["w27No", {"ConstructorFunction" -> ({False, Null} &)}]; OpenRead["x", Method -> "w27No"]]',
     Raises(ValueError)),                                                                     # constructor failure
    ('Module[{}, DefineInputStreamMethod["w27Empty", {"ConstructorFunction" -> ({True, 0} &)}]; '
     'ReadList[OpenRead["x", Method -> "w27Empty"], Byte]]', '{}'),                           # doc: default ReadFunction
    ('Module[{}, DefineInputStreamMethod["w27Err", {"ConstructorFunction" -> ({True, 0} &), "ReadFunction" -> ({{}, #} &), '
     '"EndOfFileQFunction" -> ({False, #} &), "ErrorTextFunction" -> ({"broken", #} &)}]; ReadList[OpenRead["x", Method -> "w27Err"], Byte]]',
     Raises(ValueError)),                                                                     # doc: ErrorTextFunction
    (f'Module[{{}}, {_STRINGNAME}; RemoveInputStreamMethod["w27StringName"]; MemberQ[$InputStreamMethods, "w27StringName"]]',
     'False'),                                                                                # Check
    (f'Module[{{}}, {_STRINGNAME}; RemoveInputStreamMethod["w27StringName"]]', Pred(lambda g: g is None, "Null")),
    (f'Module[{{}}, {_STRINGNAME}; RemoveInputStreamMethod["w27StringName"]; OpenRead["x", Method -> "w27StringName"]]',
     Raises(ValueError)),
    ('RemoveInputStreamMethod["w27NeverDefined"]', Raises(ValueError)),
    ('RemoveInputStreamMethod[1]', Raises(TypeError)),
    ('First[$InputStreamMethods]', '"File"'),
    ('First[$OutputStreamMethods]', '"File"'),
    (f'Module[{{}}, {_COLLECT}; MemberQ[$OutputStreamMethods, "w27Collect"]]', 'True'),
    (f'Module[{{}}, {_COLLECT}; Reap[Module[{{s = OpenWrite["anything", Method -> "w27Collect"]}}, '
     'WriteString[s, "hi"]; WriteLine[s, "yo"]; {StreamPosition[s], Close[s]}]]]',
     '{{5, "anything"}, {{{104, 105}, {121, 111, 10}}}}'),                                    # Check: bytes reach the method
    (f'Module[{{}}, {_COLLECT}; Reap[Module[{{s = OpenAppend["x", Method -> "w27Collect"]}}, BinaryWrite[s, {{1, 2}}]; Close[s]]]]',
     '{"x", {{{1, 2}}}}'),
    ('Module[{}, DefineOutputStreamMethod["w27Ap", {"ConstructorFunction" -> Function[{n, ap, c, o}, {True, ap}], '
     '"WriteFunction" -> Function[{st, b}, Sow[st]; {Length[b], st}]}]; Reap[Module[{s = OpenAppend["x", Method -> "w27Ap"]}, '
     'WriteString[s, "z"]; Close[s]]][[2, 1]]]', '{True}'),                                   # doc: isAppend argument
    ('Module[{}, DefineOutputStreamMethod["w27Short", {"ConstructorFunction" -> ({True, 0} &), '
     '"WriteFunction" -> ({0, #1} &)}]; WriteString[OpenWrite["x", Method -> "w27Short"], "abc"]]', Raises(ValueError)),
    ('Module[{}, DefineOutputStreamMethod["w27Null", {"ConstructorFunction" -> ({True, 0} &)}]; '
     'Module[{s = OpenWrite["x", Method -> "w27Null"]}, WriteString[s, "abc"]; Close[s]]]', '"x"'),  # doc: no WriteFunction
    (f'Module[{{}}, {_COLLECT}; {_COLLECT}]', Raises(ValueError)),
    ('DefineOutputStreamMethod["w27B", {"ReadFunction" -> f, "ConstructorFunction" -> g}]', Raises(ValueError)),
    ('DefineOutputStreamMethod[1, {}]', Raises(TypeError)),
    ('DefineOutputStreamMethod[w27x, {}]', 'DefineOutputStreamMethod[w27x, {}]'),
    (f'Module[{{}}, {_COLLECT}; RemoveOutputStreamMethod["w27Collect"]; MemberQ[$OutputStreamMethods, "w27Collect"]]',
     'False'),                                                                                # Check
    (f'Module[{{}}, {_COLLECT}; RemoveOutputStreamMethod["w27Collect"]]', Pred(lambda g: g is None, "Null")),
    ('RemoveOutputStreamMethod["w27NeverDefined"]', Raises(ValueError)),
    ('RemoveOutputStreamMethod[{}]', Raises(TypeError)),
]

# ─────────────────────────── inert symbols and options (≥3 uses each; most are counted above) ───────────────────────────

INERT = [
    ('Head[OpenRead["a.txt"]] === InputStream', 'True'),                                      # doc: OpenRead gives InputStream
    ('Head[StringToStream["x"]] === InputStream', 'True'),                                    # doc: InputStream[String, n]
    ('Read[StringToStream["{1, 2}"], Expression]', '{1, 2}'),                                 # doc: Expression type
    ('Module[{}, Get["infile.m"]; StringQ[$InputFileName]]', 'True'),                         # doc: $InputFileName is a string
    ('Head[OpenRead["a.txt"]]', 'InputStream'),                                               # doc: OpenRead gives InputStream
    ('Head[StringToStream["x"]]', 'InputStream'),                                             # doc: InputStream[String, n]
    ('MatchQ[OpenRead["a.txt"], InputStream[_String, _Integer]]', 'True'),
    ('Head[OpenWrite["o.txt"]]', 'OutputStream'),                                             # doc: OpenWrite gives OutputStream
    ('MatchQ[OpenAppend["a.txt"], OutputStream[_String, _Integer]]', 'True'),
    ('Read[OutputStream["nothere", 99]]', Raises(ValueError)),
    ('File["x/y"]', 'File["x/y"]'),                                                           # doc: inert, need not exist
    ('FileExistsQ[File["nothere/at/all"]]', 'False'),
    ('ReadList[StringToStream["e"], Word] == {"e"} && Read[StringToStream[""], Character] === EndOfFile', 'True'),
    ('ReadLine[StringToStream[""]] === EndOfFile', 'True'),
    ('ReadByteArray[StringToStream["ab"], EndOfBuffer]', 'ByteArray[{97, 98}]'),             # doc: EndOfBuffer
    ('ReadString[StringToStream[""], EndOfBuffer]', 'EndOfFile'),
    ('ReadString[StringToStream["ab"], EndOfFile]', '"ab"'),
    ('ReadList[StringToStream["ab"], Byte]', '{97, 98}'),
    ('ReadList[StringToStream["ab"], Character]', '{"a", "b"}'),
    ('ReadList[StringToStream["1+2\\n3"], Expression]', '{3, 3}'),
    ('ReadList[StringToStream["1 x 2"], Number]', Raises(ValueError)),
    ('ReadList[StringToStream["r1\\nr2"], Record] == ReadList[StringToStream["r1\\nr2"], String]', 'True'),
    ('ReadList[StringToStream["w1 w2"], Word]', '{"w1", "w2"}'),
    ('Module[{s = OpenRead["bin.dat", BinaryFormat -> False]}, Read[s, Byte]]', '1'),        # BinaryFormat
    ('ReadList[OpenRead["a.txt", BinaryFormat -> True], Record]', '{"1 2 3", "hello world", "x+y"}'),
    ('BinaryReadList["bin.dat", "Integer16", ByteOrdering -> -1]', '{513, 65283 - 65536}'),   # ByteOrdering -1
    ('Module[{}, CreateFile["ci/a/b.txt", CreateIntermediateDirectories -> True]; DirectoryQ["ci/a"]]', 'True'),
    ('Module[{}, CopyFile["a.txt", "ow.txt"]; CopyFile["bin.dat", "ow.txt", OverwriteTarget -> True]; FileByteCount["ow.txt"]]', '4'),
    ('Module[{}, RenameFile["bin.dat", "a.txt", OverwriteTarget -> False]]', Raises(ValueError)),
    ('ReadList["a.txt", Word, RecordLists -> True]', '{{"1", "2", "3"}, {"hello", "world"}, {"x+y"}}'),
    ('FileSystemMap[FileExtension, ".", FileNameForms -> {"*.dat", "*.m"}]', '<|"bin.dat" -> "dat", "infile.m" -> "m"|>'),
    ('FindFile["init.m", Path -> {"Pkg"}] == FileNameJoin[{w27dir, "Pkg", "init.m"}]', 'True'),
    ('ReadList["c.m", Path -> {"sub/inner"}]', '{2}'),
    ('Module[{}, CreateDirectory["dc/x"]; DeleteDirectory["dc", DeleteContents -> True]; DirectoryQ["dc"]]', 'False'),
]

CASES = NAMES + DIRS + FILES + STREAMS_CASES + BINARY + METHODS + INERT


@pytest.fixture
def w27env(monkeypatch):
    from mikoshilang.library import wl_files as W
    from mikoshilang.library import wl_text as T
    orig = os.getcwd()
    d = os.path.realpath(tempfile.mkdtemp(prefix="w27-"))
    for rel, data in SAMPLE.items():
        p = os.path.join(d, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
    for sub in EMPTY_DIRS:
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    monkeypatch.chdir(d)
    monkeypatch.setenv("HOME", d)
    monkeypatch.setenv("USERPROFILE", d)
    monkeypatch.delenv("MIKOSHILANG_USERBASE", raising=False)
    monkeypatch.delenv("MIKOSHILANG_BASE", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", os.path.join(d, "tmp"))
    stack = list(W.DIRECTORY_STACK)
    W.DIRECTORY_STACK[:] = []
    strings = set(T.STRING_STREAMS)
    in_m, out_m = dict(W._IN_METHODS), dict(W._OUT_METHODS)
    open_before = set(W.STREAMS)
    ev('w27dir = "%s"' % d.replace("\\", "\\\\"))
    try:
        yield d
    finally:
        ev("Clear[w27dir]")
        for n in list(W.STREAMS):
            if n not in open_before:
                try:
                    W.STREAMS[n].close()
                except Exception:  # noqa: BLE001 - cleanup
                    W.STREAMS.pop(n, None)
        for n in list(T.STRING_STREAMS):
            if n not in strings:
                T.STRING_STREAMS.pop(n, None)
        W._IN_METHODS.clear()
        W._IN_METHODS.update(in_m)
        W._OUT_METHODS.clear()
        W._OUT_METHODS.update(out_m)
        W.DIRECTORY_STACK[:] = stack
        os.chdir(orig)
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w27(case, w27env):
    check(case)


def test_fileprint_output(w27env, capsys):
    """FilePrint prints the selected lines (doc: FilePrint[file, n], -n and m;;n;;s)."""
    ev('FilePrint["a.txt"]')
    assert capsys.readouterr().out == "1 2 3\nhello world\nx+y\n"
    ev('FilePrint["a.txt", 2]')
    assert capsys.readouterr().out == "1 2 3\nhello world\n"
    ev('FilePrint["a.txt", -1]')
    assert capsys.readouterr().out == "x+y\n"
    ev('FilePrint["a.txt", 3 ;; 1 ;; -2]')
    assert capsys.readouterr().out == "x+y\n1 2 3\n"


def test_write_to_stdout(w27env, capsys):
    """Write and WriteString to $StandardOutputStream go to standard output (OutputForm)."""
    ev("Write[$StandardOutputStream, 1 + 1]")
    ev('WriteString[OutputStream["stdout", 1], "abc"]')
    assert capsys.readouterr().out == "2\nabc"


def test_fixture_isolation(w27env):
    """The fixture's directory is the working directory, home and temporary directory parent; nothing escapes it."""
    d = w27env
    assert os.getcwd() == d
    assert ev("$HomeDirectory") == d
    assert os.path.dirname(ev("$TemporaryDirectory")) == d
