"""Wolfram batch W27: files, directories and streams (low-level I/O).

* file names: FileNameSplit, FileNameJoin, FileNameTake, FileNameDrop, FileNameDepth, FileBaseName, FileExtension,
  DirectoryName, ParentDirectory, ExpandFileName, AbsoluteFileName (with OperatingSystem -> "Unix" | "MacOSX" |
  "Windows" where Wolfram documents it);
* file system: FileExistsQ, DirectoryQ, FileType, FileByteCount, FileSize, FileDate, SetFileDate, FilePrint, FileNames,
  FindFile, FileSystemMap, FileSystemScan, CreateFile, CreateDirectory, CopyFile, RenameFile, DeleteFile,
  CopyDirectory, RenameDirectory, DeleteDirectory, Directory, SetDirectory, ResetDirectory, DirectoryStack;
* streams: OpenRead, OpenWrite, OpenAppend, Close, Streams, Read, ReadList, Skip, ReadLine, ReadString,
  ReadByteArray, BinaryRead, BinaryReadList, BinaryWrite, Write, WriteString, WriteLine, StreamPosition,
  SetStreamPosition and user-defined stream methods (DefineInputStreamMethod, DefineOutputStreamMethod, ...).
  String streams are the InputStream[String, n] objects of StringToStream (batch W11, ``wl_text``); their state and
  serial numbers are shared with that module, so Find/FindList and Read work on the same stream;
* system variables: $HomeDirectory, $TemporaryDirectory, $Path, $PathnameSeparator, $ByteOrdering, ...

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).  Where Wolfram prints a message and returns $Failed (missing file, non-empty directory, existing
target, unreadable object) these functions raise instead; the documented non-error results ($Failed from FindFile
when nothing is found, EndOfFile at the end of a stream) are kept.
"""
from __future__ import annotations

import datetime as _dt
import io
import os
import re
import shutil
import stat as _stat
import struct
import sys
import tempfile
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from ..evaluate import SYSTEM_VARIABLES as _SYSTEM_VARIABLES
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated
from .wl_common import L, ev, apply, is_sym, is_list, is_rule
from . import wl_text as _wtext
from .wl_importexport import input_text as _input_text, byte_array as _byte_array, ba_bytes as _ba_bytes, \
    is_byte_array as _is_byte_array

AREA = "files & streams (W27)"
NULL = None
EOF = Symbol("EndOfFile")
_FAILED = Symbol("$Failed")


# ═══════════════════════════════════════════ argument helpers ═══════════════════════════════════════════

def _form(x):
    from ..formatting import input_form
    try:
        return input_form(x)
    except Exception:  # pragma: no cover - formatting never fails for evaluated expressions
        return repr(x)


def _opt_name(r):
    if is_rule(r) or (isinstance(r, Expr) and r.head == "RuleDelayed" and len(r.args) == 2):
        k = r.args[0]
        if isinstance(k, Symbol):
            return k.name
        if isinstance(k, str):
            return k
    return None


def _opts(args, allowed, fname, unsupported=()):
    """Split trailing Name -> value options; a known-but-unsupported Wolfram option raises NotImplementedError,
    an unknown option name raises ValueError."""
    args = list(args)
    opts = {}
    while args and _opt_name(args[-1]) is not None:
        r = args.pop()
        n = _opt_name(r)
        if n in unsupported:
            raise NotImplementedError(f"{fname}: the option {n} is not supported")
        if n not in allowed:
            raise ValueError(f"{fname}: unknown option {n}")
        opts[n] = r.args[1]
    for a in args:
        if _opt_name(a) in allowed:
            raise ValueError(f"{fname}: options must come after the positional arguments")
    return args, opts


def _bool_opt(opts, name, fname, default):
    v = opts.get(name, default)
    if is_sym(v, "Automatic"):
        return default
    if v is True or v is False:
        return v
    raise ValueError(f"{fname}: {name} must be True or False, not {_form(v)}")


def _is_symbolic(x):
    return isinstance(x, Symbol) or (isinstance(x, Expr) and x.head not in ("List", "File", "Rule", "String")
                                     and not KL.atomq(x) and x.head not in ("InputStream", "OutputStream"))


def _name(x, fname, what="file name"):
    """A file name given as "name" or File["name"] -> str (not expanded). Symbolic -> Unevaluated."""
    if isinstance(x, str):
        return x
    if isinstance(x, Expr) and x.head == "File" and len(x.args) == 1:
        if isinstance(x.args[0], str):
            return x.args[0]
        if isinstance(x.args[0], Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: File[...] must contain a string, not {_form(x.args[0])}")
    if isinstance(x, Symbol) or (isinstance(x, Expr) and not is_list(x) and x.head not in
                                 ("InputStream", "OutputStream", "Rule", "ByteArray")):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(x)} is not a valid {what} (a string or File[\"path\"] is expected)")


def _path(x, fname, what="file name"):
    """A file name expanded for the file system: "~" is the home directory."""
    s = _name(x, fname, what)
    if s == "":
        raise ValueError(f"{fname}: the {what} must not be empty")
    if "\x00" in s:
        raise ValueError(f"{fname}: the {what} must not contain a null character")
    return _expanduser(s)


def _expanduser(s):
    if s == "~" or s.startswith("~/") or (os.name == "nt" and s.startswith("~\\")):
        return home_directory() + s[1:]
    return s


def _int(x, fname, what="argument"):
    if isinstance(x, bool):
        raise TypeError(f"{fname}: {what} must be an integer, not {x}")
    if K.is_int(x):
        return int(x)
    if isinstance(x, Symbol) or (isinstance(x, Expr) and not is_list(x)):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be an integer, not {_form(x)}")


def _oserr(fname, e, path):
    return ValueError(f"{fname}: {e.strerror or e} ({path})")


# ═══════════════════════════════════════════ system directories ═══════════════════════════════════════════

def home_directory():
    if os.name == "nt":
        h = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    else:
        h = os.environ.get("HOME") or os.path.expanduser("~")
    return os.path.abspath(h)


def temporary_directory():
    v = ev(Symbol("$TemporaryDirectory"))
    if not isinstance(v, str) or not v:
        raise ValueError(f"$TemporaryDirectory: {_form(v)} is not a directory name")
    return v


def _platform():
    if os.name == "nt":
        return "Windows"
    return "MacOSX" if sys.platform == "darwin" else "Unix"


def _user_base():
    env = os.environ.get("MIKOSHILANG_USERBASE")
    if env:
        return os.path.abspath(env)
    p = _platform()
    if p == "Windows":
        return os.path.join(os.environ.get("APPDATA") or os.path.join(home_directory(), "AppData", "Roaming"),
                            "Mikoshilang")
    if p == "MacOSX":
        return os.path.join(home_directory(), "Library", "Mikoshilang")
    return os.path.join(home_directory(), ".Mikoshilang")


def _base():
    env = os.environ.get("MIKOSHILANG_BASE")
    if env:
        return os.path.abspath(env)
    p = _platform()
    if p == "Windows":
        return os.path.join(os.environ.get("PROGRAMDATA") or "C:\\ProgramData", "Mikoshilang")
    if p == "MacOSX":
        return "/Library/Mikoshilang"
    return "/usr/share/Mikoshilang"


def _documents():
    return os.path.join(home_directory(), "Documents")


def _installation():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _root():
    if os.name == "nt":
        d = os.path.splitdrive(os.getcwd())[0]
        return (d or "C:") + "\\"
    return "/"


_INITIAL_DIRECTORY = os.getcwd()
_INPUT_FILE = [""]


def _default_path():
    from . import wl_evaluation as WE
    ub, b = _user_base(), _base()
    out = [".", os.path.join(ub, "Kernel"), os.path.join(ub, "Autoload"), os.path.join(ub, "Applications"),
           os.path.join(b, "Kernel"), os.path.join(b, "Autoload"), os.path.join(b, "Applications")]
    out += [d for d in WE.PACKAGE_PATH if isinstance(d, str)]
    env = os.environ.get("MIKOSHILANG_PATH")
    if env:
        out += [p for p in env.split(os.pathsep) if p]
    return L(*out)


_SYSVARS = {
    "$HomeDirectory": (home_directory,
                       "the user's home directory as an absolute name ($HOME on Unix, %USERPROFILE% on Windows)"),
    "$TemporaryDirectory": (lambda: os.path.abspath(tempfile.gettempdir()),
                            "the directory for temporary files (Python tempfile.gettempdir(), honouring TMPDIR); "
                            "CreateFile[], CreateDirectory[] and OpenWrite[] use its current value, so it can be "
                            "reassigned or localised with Block"),
    "$RootDirectory": (_root, "the root directory of the file system (\"/\" on Unix; the current drive on Windows)"),
    "$InitialDirectory": (lambda: _INITIAL_DIRECTORY,
                          "the working directory when MikoshiLang was started (imported)"),
    "$UserBaseDirectory": (_user_base, "the per-user base directory: ~/.Mikoshilang on Linux, ~/Library/Mikoshilang on "
                                       "macOS, %APPDATA%\\Mikoshilang on Windows, or $MIKOSHILANG_USERBASE when set "
                                       "(the Wolfram product name is replaced by Mikoshilang)"),
    "$BaseDirectory": (_base, "the system-wide base directory: /usr/share/Mikoshilang on Linux, /Library/Mikoshilang "
                              "on macOS, %PROGRAMDATA%\\Mikoshilang on Windows, or $MIKOSHILANG_BASE when set"),
    "$InstallationDirectory": (_installation, "the directory MikoshiLang is installed in (the parent of the "
                                              "mikoshilang package)"),
    "$UserDocumentsDirectory": (_documents, "the user's documents directory, ~/Documents on every system as documented"),
    "$WolframDocumentsDirectory": (lambda: os.path.join(_documents(), "Wolfram"),
                                   "$UserDocumentsDirectory/Wolfram, as documented"),
    "$PathnameSeparator": (lambda: os.sep, "the pathname separator, \"/\" (\"\\\\\" on Windows)"),
    "$ByteOrdering": (lambda: 1 if sys.byteorder == "big" else -1,
                      "the native byte ordering: -1 (little-endian, x86 and ARM) or +1 (big-endian); the default "
                      "ByteOrdering of BinaryRead, BinaryWrite and BinaryReadList"),
    "$InputFileName": (lambda: _INPUT_FILE[-1], "the absolute name of the file being read by Get (<<) or Needs, and "
                                                "\"\" otherwise"),
    "$Path": (_default_path, "directories searched by FindFile, OpenRead and the Read functions for relative file "
                             "names: \".\", the Kernel, Autoload and Applications subdirectories of "
                             "$UserBaseDirectory and $BaseDirectory, then Get's PACKAGE_PATH and $MIKOSHILANG_PATH "
                             "entries; assigning $Path (or Block) changes the search"),
    "$StandardOutputStream": (lambda: Expr("OutputStream", "stdout", 1), "OutputStream[\"stdout\", 1]"),
    "$StandardErrorStream": (lambda: Expr("OutputStream", "stderr", 2), "OutputStream[\"stderr\", 2]"),
    "$InputStreamMethods": (lambda: L("File", *sorted(_IN_METHODS)),
                            "the Method settings for OpenRead: \"File\" and the names defined with "
                            "DefineInputStreamMethod"),
    "$OutputStreamMethods": (lambda: L("File", *sorted(_OUT_METHODS)),
                             "the Method settings for OpenWrite and OpenAppend: \"File\" and the names defined with "
                             "DefineOutputStreamMethod"),
}
for _n, (_g, _d) in _SYSVARS.items():
    _SYSTEM_VARIABLES[_n] = _g
    record(_n, AREA, "added", f"{_n}: {_d}. A user assignment takes precedence. Check: reference (Wolfram "
                              f"documentation of {_n} and the Python os/sys/tempfile values) in tests.")


def _wrap_get():
    """Keep $InputFileName set while Get (<<) or Needs reads a file (wraps wl_evaluation._get_file)."""
    from . import wl_evaluation as WE
    orig = WE._get_file
    if getattr(orig, "_w27", False):
        return

    def _get_file(path):
        _INPUT_FILE.append(os.path.abspath(path))
        try:
            return orig(path)
        finally:
            _INPUT_FILE.pop()
    _get_file._w27 = True
    WE._get_file = _get_file


_wrap_get()


# ═══════════════════════════════════════════ file-name operations ═══════════════════════════════════════════

_OS_NAMES = ("Unix", "MacOSX", "Windows")


def _os_opt(o, fname):
    v = o.get("OperatingSystem")
    if v is None or is_sym(v, "Automatic") or (isinstance(v, Symbol) and v.name == "$OperatingSystem"):
        return _platform()
    if v in _OS_NAMES:
        return v
    if v == "Linux":
        return "Unix"
    raise ValueError(f"{fname}: OperatingSystem must be \"Unix\", \"MacOSX\" or \"Windows\", not {_form(v)}")


def _sep_of(osn):
    return "\\" if osn == "Windows" else "/"


def split_name(s, osn):
    """FileNameSplit: absolute names start with ""; a Windows drive "C:" or share "\\\\server\\share" is one part."""
    if osn == "Windows":
        t = s.replace("/", "\\")
        if t.startswith("\\\\"):
            rest = [p for p in t[2:].split("\\")]
            head = "\\\\" + "\\".join(rest[:2])
            return [head] + [p for p in rest[2:] if p]
        m = re.match(r"^([A-Za-z]:)(.*)$", t)
        if m:
            return [m.group(1)] + [p for p in m.group(2).split("\\") if p]
        parts = [p for p in t.split("\\") if p]
        return ([""] + parts) if t.startswith("\\") else parts
    parts = [p for p in s.split("/") if p]
    return ([""] + parts) if s.startswith("/") else parts


def join_parts(parts, osn):
    sep = _sep_of(osn)
    if not parts:
        return ""
    if parts[0] == "":
        return sep + sep.join(parts[1:])
    if osn == "Windows" and re.match(r"^[A-Za-z]:$", parts[0]):
        return parts[0] + sep + sep.join(parts[1:]) if len(parts) > 1 else parts[0] + sep
    return sep.join(parts)


def _join(names, osn):
    parts = []
    for i, p in enumerate(names):
        if p == "":
            if i == 0:
                parts.append("")            # {"", "a", ...}: an absolute name
            continue
        sp_ = split_name(p, osn)
        if (i > 0 or parts == [""]) and sp_ and sp_[0] == "":
            sp_ = sp_[1:]
        if i > 0 and osn == "Windows" and sp_ and (re.match(r"^[A-Za-z]:$", sp_[0]) or sp_[0].startswith("\\\\")):
            raise ValueError(f"FileNameJoin: {p!r} is an absolute name and cannot follow other path elements")
        parts.extend(sp_)
    return join_parts(parts, osn)


def _name_opts(rest, fname):
    args, o = _opts(rest, ("OperatingSystem",), fname)
    return args, _os_opt(o, fname)


@fn("FileNameSplit", area=AREA, status="added",
    notes="FileNameSplit[\"name\"] splits at the pathname separators; an absolute name gives a first element \"\"; "
          "repeated separators are ignored; OperatingSystem -> \"Unix\" | \"MacOSX\" | \"Windows\" (Windows accepts "
          "\\ and /, a drive \"C:\" or share \"\\\\server\\share\" is one part); File[\"name\"] is accepted; symbolic "
          "input stays unevaluated; non-strings raise TypeError, a bad OperatingSystem ValueError. URL[...] is not "
          "supported. Check: exact (string operations); round trips with FileNameJoin in tests.")
def _filenamesplit(name: Expr, *rest: Option) -> ListOf[String]:
    args, osn = _name_opts(rest, "FileNameSplit")
    if args:
        raise ValueError("FileNameSplit: called with too many arguments")
    return L(*split_name(_name(name, "FileNameSplit"), osn))


@fn("FileNameJoin", area=AREA, status="added",
    notes="FileNameJoin[{\"n1\", \"n2\", ...}] joins names (which may contain separators) with the pathname "
          "separator; {\"\", ...} gives an absolute name; FileNameJoin[\"name\"] canonicalises the separators; "
          "OperatingSystem option as FileNameSplit. Elements must be strings or File[...] (TypeError otherwise); a "
          "symbolic element keeps the call unevaluated. Check: exact; FileNameJoin[FileNameSplit[s]] == s round "
          "trips in tests.")
def _filenamejoin(names: ListOf[String] | String, *rest: Option) -> String:
    args, osn = _name_opts(rest, "FileNameJoin")
    if args:
        raise ValueError("FileNameJoin: called with too many arguments")
    if isinstance(names, str) or (isinstance(names, Expr) and names.head == "File"):
        return _join([_name(names, "FileNameJoin")], osn)
    if not is_list(names):
        if isinstance(names, Symbol) or isinstance(names, Expr):
            raise Unevaluated()
        raise TypeError(f"FileNameJoin: {_form(names)} is not a list of file name parts")
    return _join([_name(n, "FileNameJoin") for n in names.args], osn)


def _take_spec(spec, n, fname, drop=False):
    """Indices (0-based, increasing) selected by a Take/Drop-style spec on n elements."""
    if K.is_int(spec) and not isinstance(spec, bool):
        k = int(spec)
        if k >= 0:
            sel = range(0, min(k, n))
        else:
            sel = range(max(0, n + k), n)
        return list(sel)
    if is_list(spec) and spec.args and len(spec.args) <= 2 and all(K.is_int(a) and not isinstance(a, bool)
                                                                   for a in spec.args):
        m = int(spec.args[0])
        k = int(spec.args[-1])
        m = m + n + 1 if m < 0 else m
        k = k + n + 1 if k < 0 else k
        if m < 1 or k > n or (m > k + 1):
            raise ValueError(f"{fname}: cannot {'drop' if drop else 'take'} elements {_form(spec)} of a name with "
                             f"{n} element(s)")
        return list(range(m - 1, k))
    if isinstance(spec, Symbol) or (isinstance(spec, Expr) and not is_list(spec)):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(spec)} is not a valid element specification (n, -n or {{m, n}})")


@fn("FileNameTake", area=AREA, status="added",
    notes="FileNameTake[\"name\"] gives the last path element; FileNameTake[name, n] the first n, -n the last n, "
          "{m, n} elements m through n (more than present: those present), joined with the separator; "
          "OperatingSystem option; File[...] accepted. Check: exact; FileNameJoin of Take/Drop parts in tests.")
def _filenametake(name: Expr, *rest: Integer | ListOf[Integer] | Option) -> String:
    args, osn = _name_opts(rest, "FileNameTake")
    if len(args) > 1:
        raise ValueError("FileNameTake: called with too many arguments")
    parts = split_name(_name(name, "FileNameTake"), osn)
    idx = _take_spec(args[0] if args else -1, len(parts), "FileNameTake")
    return join_parts([parts[i] for i in idx], osn)


@fn("FileNameDrop", area=AREA, status="added",
    notes="FileNameDrop[\"name\"] drops the last path element; FileNameDrop[name, n] drops the first n, -n the last "
          "n, {m, n} elements m through n (more than present: drops those present); OperatingSystem option; "
          "File[...] accepted. Check: exact; FileNameTake/FileNameDrop complement each other in tests.")
def _filenamedrop(name: Expr, *rest: Integer | ListOf[Integer] | Option) -> String:
    args, osn = _name_opts(rest, "FileNameDrop")
    if len(args) > 1:
        raise ValueError("FileNameDrop: called with too many arguments")
    parts = split_name(_name(name, "FileNameDrop"), osn)
    idx = set(_take_spec(args[0] if args else -1, len(parts), "FileNameDrop", drop=True))
    return join_parts([p for i, p in enumerate(parts) if i not in idx], osn)


@fn("FileNameDepth", area=AREA, status="added",
    notes="FileNameDepth[\"name\"] is the number of path elements, Length[FileNameSplit[name]] (an absolute name "
          "counts its leading root element \"\"); OperatingSystem option; File[...] accepted. Check: exact; "
          "equality with Length[FileNameSplit[...]] in tests.")
def _filenamedepth(name: Expr, *rest: Option) -> NonNegInteger:
    args, osn = _name_opts(rest, "FileNameDepth")
    if args:
        raise ValueError("FileNameDepth: called with too many arguments")
    return len(split_name(_name(name, "FileNameDepth"), osn))


def _last_component(s, osn):
    seps = ("\\", "/") if osn == "Windows" else ("/",)
    if s.endswith(seps):
        return ""                   # the name has the form of a directory name
    parts = split_name(s, osn)
    if not parts:
        return ""
    last = parts[-1]
    if last == "" or (osn == "Windows" and (re.match(r"^[A-Za-z]:$", last) or last.startswith("\\\\"))):
        return ""
    return last


@fn("FileExtension", area=AREA, status="added",
    notes="FileExtension[\"name.ext\"] gives \"ext\", the text after the last \".\" of the last path element; \"\" "
          "when there is no \".\", when the name ends with \".\" or has the form of a directory name (ends with a "
          "separator); directory parts are ignored; OperatingSystem option; File[...] accepted. Check: exact; "
          "FileBaseName <> \".\" <> FileExtension reproduces the last element in tests.")
def _fileextension(name: Expr, *rest: Option) -> String:
    args, osn = _name_opts(rest, "FileExtension")
    if args:
        raise ValueError("FileExtension: called with too many arguments")
    last = _last_component(_name(name, "FileExtension"), osn)
    k = last.rfind(".")
    return "" if k < 0 else last[k + 1:]


@fn("FileBaseName", area=AREA, status="added",
    notes="FileBaseName[\"dir/name.ext\"] gives \"name\": the last path element without its last \".ext\" "
          "(\"file.tar.gz\" -> \"file.tar\", \"file.\" -> \"file\"); OperatingSystem option; File[...] accepted. "
          "Check: exact; reconstruction with FileExtension in tests.")
def _filebasename(name: Expr, *rest: Option) -> String:
    args, osn = _name_opts(rest, "FileBaseName")
    if args:
        raise ValueError("FileBaseName: called with too many arguments")
    last = _last_component(_name(name, "FileBaseName"), osn)
    k = last.rfind(".")
    return last if k < 0 else last[:k]


def _dirname_once(s, osn):
    seps = ("\\", "/") if osn == "Windows" else ("/",)
    sep = _sep_of(osn)
    t = s.replace("/", "\\") if osn == "Windows" else s
    # strip trailing separators, but keep a root
    while len(t) > 1 and t.endswith(seps) and not (osn == "Windows" and re.match(r"^[A-Za-z]:\\$", t)):
        t = t[:-1]
    if osn == "Windows" and re.match(r"^[A-Za-z]:\\?$", t):
        return t if t.endswith("\\") else t + "\\"
    k = max(t.rfind(c) for c in seps)
    if k < 0:
        if osn == "Windows" and re.match(r"^[A-Za-z]:", t):
            return t[:2]
        return ""
    return t[:k + 1] if t[:k + 1] else sep


@fn("DirectoryName", area=AREA, status="added",
    notes="DirectoryName[\"dir/name\"] gives the directory part including its trailing separator (\"a/b/c\" -> "
          "\"a/b/\", \"/Users/steve/report.nb\" -> \"/Users/steve/\"); \"\" when there is no directory part; "
          "DirectoryName[name, n] applies it n times; OperatingSystem option; File[...] accepted (a string is "
          "returned). Check: exact; reference values from the Wolfram documentation in tests.")
def _directoryname(name: Expr, *rest: PosInteger | Option) -> String:
    args, osn = _name_opts(rest, "DirectoryName")
    if len(args) > 1:
        raise ValueError("DirectoryName: called with too many arguments")
    n = 1
    if args:
        n = _int(args[0], "DirectoryName", "argument 2")
        if n < 1:
            raise ValueError("DirectoryName: argument 2 must be a positive integer")
    s = _name(name, "DirectoryName")
    for _ in range(n):
        s = _dirname_once(s, osn)
    return s


@fn("ExpandFileName", area=AREA, status="added",
    notes="ExpandFileName[\"name\"] gives the absolute name: interpreted relative to Directory[], \"~\" expanded "
          "to $HomeDirectory, . and .. resolved textually (the file need not exist); File[...] accepted. URL[...] "
          "is not supported. Check: exact (os.path.normpath/join); tests compare with Directory[] and "
          "FileNameJoin.")
def _expandfilename(name: Expr) -> String:
    s = _expanduser(_name(name, "ExpandFileName"))
    if s == "":
        return os.getcwd()
    return os.path.normpath(os.path.join(os.getcwd(), s))


@fn("AbsoluteFileName", area=AREA, status="added",
    notes="AbsoluteFileName[\"name\"] gives the absolute name of an existing file or directory (relative to "
          "Directory[], . and .. resolved); a name that does not exist raises ValueError (Wolfram: message and "
          "$Failed); File[...] accepted. Check: exact; tests check FileExistsQ of the result and agreement with "
          "ExpandFileName.")
def _absolutefilename(name: Expr) -> String:
    p = _path(name, "AbsoluteFileName")
    full = os.path.normpath(os.path.join(os.getcwd(), p))
    if not os.path.exists(full):
        raise ValueError(f"AbsoluteFileName: the file {_name(name, 'AbsoluteFileName')!r} does not exist")
    return full


@fn("ParentDirectory", area=AREA, status="added",
    notes="ParentDirectory[] is the parent of Directory[]; ParentDirectory[n] goes n levels up; "
          "ParentDirectory[\"dir\"] and ParentDirectory[\"dir\", n] give the full (absolute) name of the parent "
          "(relative names are taken relative to Directory[]); the root is its own parent; File[...] accepted. "
          "Check: exact; tests compare with DirectoryName/ExpandFileName.")
def _parentdirectory(*args: Expr | Integer) -> String:
    if len(args) > 2:
        raise ValueError("ParentDirectory: called with too many arguments")
    n = 1
    if not args:
        d = os.getcwd()
    elif len(args) == 1 and K.is_int(args[0]) and not isinstance(args[0], bool):
        d, n = os.getcwd(), int(args[0])
    else:
        d = os.path.normpath(os.path.join(os.getcwd(), _path(args[0], "ParentDirectory", "directory name")))
        if len(args) == 2:
            n = _int(args[1], "ParentDirectory", "argument 2")
    if n < 0:
        raise ValueError("ParentDirectory: the number of levels must be non-negative")
    for _ in range(n):
        d = os.path.dirname(d)
    return d


# ═══════════════════════════════════════════ file-system queries ═══════════════════════════════════════════

def _query_path(x, fname):
    """Name for a query that answers False/None for a missing file (FileExistsQ, DirectoryQ, FileType)."""
    s = _name(x, fname)
    if s == "" or "\x00" in s:
        return None
    return _expanduser(s)


@fn("FileExistsQ", area=AREA, status="added",
    notes="FileExistsQ[\"name\"] is True if a file or directory of that name exists (relative to Directory[]; "
          "\"~\" expanded), else False; File[...] accepted; a symbol stays unevaluated; other non-strings raise "
          "TypeError (Wolfram: message and False). Check: exact (os.path.exists); tests create and delete files in a "
          "temporary directory.")
def _fileexistsq(name: Expr) -> Boolean:
    p = _query_path(name, "FileExistsQ")
    return bool(p) and os.path.exists(p)


@fn("DirectoryQ", area=AREA, status="added",
    notes="DirectoryQ[\"name\"] is True if name is an existing directory, else False; File[...] accepted; symbolic "
          "input stays unevaluated; other non-strings raise TypeError. Check: exact (os.path.isdir); tests in a "
          "temporary directory.")
def _directoryq(name: Expr) -> Boolean:
    p = _query_path(name, "DirectoryQ")
    return bool(p) and os.path.isdir(p)


@fn("FileType", area=AREA, status="added",
    notes="FileType[\"name\"] gives File, Directory, Special (devices, pipes, sockets) or None when nothing of that "
          "name exists (symbolic links are followed); File[...] accepted. Check: exact (os.stat); tests in a "
          "temporary directory.")
def _filetype(name: Expr) -> Expr:
    p = _query_path(name, "FileType")
    if not p or not os.path.exists(p):
        return Symbol("None")
    if os.path.isdir(p):
        return Symbol("Directory")
    if os.path.isfile(p):
        return Symbol("File")
    return Symbol("Special")


def _existing(x, fname):
    p = _path(x, fname)
    if not os.path.exists(p):
        raise ValueError(f"{fname}: the file {_name(x, fname)!r} does not exist")
    return p


@fn("FileByteCount", area=AREA, status="added",
    notes="FileByteCount[\"file\"] is the number of bytes in the file (os.stat size; for a directory the size of "
          "its directory entry); a missing file raises ValueError (Wolfram: message and $Failed); File[...] "
          "accepted. Check: exact; tests write known byte counts (BinaryWrite, WriteString) and compare.")
def _filebytecount(name: Expr) -> NonNegInteger:
    return os.stat(_existing(name, "FileByteCount")).st_size


@fn("FileSize", area=AREA, status="added",
    notes="FileSize[\"file\"] gives Quantity[n, \"Bytes\"] with the exact byte count. Deliberate deviation: Wolfram "
          "rescales to kilobytes, megabytes, ... for display; the quantity is the same. A missing file raises "
          "ValueError; File[...] accepted. Check: exact; equality with FileByteCount in tests.")
def _filesize(name: Expr) -> Expr:
    return Expr("Quantity", os.stat(_existing(name, "FileSize")).st_size, "Bytes")


def _tz():
    v = ev(Symbol("$TimeZone"))
    if isinstance(v, bool) or not isinstance(v, (int, float, sp.Integer, sp.Float, sp.Rational)):
        raise ValueError(f"FileDate: $TimeZone {_form(v)} is not a number of hours")
    return float(v)


def _date_object(t):
    tz = _tz()
    d = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=t + tz * 3600)
    sec = d.second + d.microsecond / 1e6
    return ev(Expr("DateObject", L(d.year, d.month, d.day, d.hour, d.minute, float(sec)), "Instant", "Gregorian",
                   ev(Symbol("$TimeZone"))))


_DATE_TYPES = ("Access", "Change", "Creation", "Modification")


def _stat_time(st, kind, fname):
    if kind == "Access":
        return st.st_atime_ns / 1e9
    if kind == "Modification":
        return st.st_mtime_ns / 1e9
    if kind == "Creation":
        b = getattr(st, "st_birthtime", None)
        if b is None and os.name == "nt":
            b = st.st_ctime
        if b is None:
            raise NotImplementedError(f"{fname}: the creation date is not available on this system")
        return float(b)
    if kind == "Change":
        if os.name == "nt":
            raise NotImplementedError(f"{fname}: the attribute change date is not available on Windows")
        return st.st_ctime_ns / 1e9
    raise ValueError(f"{fname}: the date type must be one of \"Access\", \"Change\", \"Creation\", "
                     f"\"Modification\" or \"Rules\", not {_form(kind)}")


@fn("FileDate", area=AREA, status="added",
    notes="FileDate[\"file\"] gives the modification date as a DateObject in $TimeZone (seconds as a machine real "
          "with the file system's sub-second resolution); FileDate[file, \"Access\" | \"Modification\" | "
          "\"Change\" | \"Creation\"] gives that date and \"Rules\" a list of rules for the types this system "
          "records (\"Creation\" raises NotImplementedError where the file system has no birth time, e.g. Linux; "
          "\"Change\" on Windows); a missing file raises ValueError. Check: reference; SetFileDate/FileDate round "
          "trips in tests.")
def _filedate(name: Expr, *rest: String) -> Expr:
    if len(rest) > 1:
        raise ValueError("FileDate: called with too many arguments")
    st = os.stat(_existing(name, "FileDate"))
    kind = rest[0] if rest else "Modification"
    if kind == "Rules":
        out = []
        for k in _DATE_TYPES:
            try:
                out.append(Expr("Rule", k, _date_object(_stat_time(st, k, "FileDate"))))
            except NotImplementedError:
                pass
        return L(*out)
    if isinstance(kind, Symbol):
        raise Unevaluated()
    if not isinstance(kind, str):
        raise TypeError(f"FileDate: the date type must be a string, not {_form(kind)}")
    return _date_object(_stat_time(st, kind, "FileDate"))


def _unix_time(date, fname):
    if isinstance(date, Expr) and date.head == "DateObject" or is_list(date):
        a = ev(Expr("AbsoluteTime", date))
    elif isinstance(date, (int, float, sp.Integer, sp.Float, sp.Rational)) and not isinstance(date, bool):
        a = date                    # an absolute time (seconds since 1900 in $TimeZone)
    elif isinstance(date, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"{fname}: {_form(date)} is not a date (a DateObject, date list or absolute time)")
    try:
        a = float(a)
    except (TypeError, ValueError):
        raise ValueError(f"{fname}: cannot convert {_form(date)} to a time")
    return a - 2208988800 - _tz() * 3600


@fn("SetFileDate", area=AREA, status="added",
    notes="SetFileDate[\"file\"] sets the modification and access dates to now; SetFileDate[file, date] to the "
          "date (DateObject, date list or absolute time, in $TimeZone); SetFileDate[file, date, \"Access\" | "
          "\"Modification\" | All] sets only that type (\"Creation\" cannot be set: NotImplementedError); returns "
          "Null; a missing file raises ValueError. Check: self-check (the file's dates are read back and compared "
          "after setting; ArithmeticError on mismatch); FileDate round trips in tests.")
def _setfiledate(name: Expr, *rest: Expr) -> Expr:
    if len(rest) > 2:
        raise ValueError("SetFileDate: called with too many arguments")
    p = _existing(name, "SetFileDate")
    import time as _time
    t = _time.time() if not rest else _unix_time(rest[0], "SetFileDate")
    kind = rest[1] if len(rest) == 2 else Symbol("All")
    st = os.stat(p)
    at, mt = st.st_atime_ns, st.st_mtime_ns
    ns = int(round(t * 1e9))
    if is_sym(kind, "All"):
        at = mt = ns
    elif kind == "Access":
        at = ns
    elif kind == "Modification":
        mt = ns
    elif kind == "Creation":
        raise NotImplementedError("SetFileDate: the creation date cannot be set")
    elif isinstance(kind, Symbol):
        raise Unevaluated()
    else:
        raise ValueError(f"SetFileDate: the date type must be \"Access\", \"Modification\" or All, not {_form(kind)}")
    os.utime(p, ns=(at, mt))
    st2 = os.stat(p)
    # self-check: the file system may round to its resolution (FAT: 2 s), never by more
    if abs(st2.st_mtime_ns - mt) > 2_000_000_000 or abs(st2.st_atime_ns - at) > 2_000_000_000:
        raise ArithmeticError("SetFileDate: result failed verification (the date was not recorded)")
    return NULL


def _span_lines(lines, spec, fname):
    n = len(lines)
    if K.is_int(spec) and not isinstance(spec, bool):
        k = int(spec)
        return lines[:k] if k >= 0 else lines[max(0, n + k):]
    if isinstance(spec, Expr) and spec.head == "Span" and len(spec.args) in (2, 3):
        def idx(v, default):
            if is_sym(v, "All"):
                return default
            if K.is_int(v) and not isinstance(v, bool):
                v = int(v)
                return v + n + 1 if v < 0 else v
            raise TypeError(f"{fname}: {_form(spec)} is not a valid span of lines")
        a = idx(spec.args[0], 1)
        b = idx(spec.args[1], n)
        st = 1
        if len(spec.args) == 3:
            if not (K.is_int(spec.args[2]) and int(spec.args[2]) != 0):
                raise ValueError(f"{fname}: the step of {_form(spec)} must be a non-zero integer")
            st = int(spec.args[2])
        if st > 0:
            return [lines[i - 1] for i in range(a, b + 1, st) if 1 <= i <= n]
        return [lines[i - 1] for i in range(a, b - 1, st) if 1 <= i <= n]
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(spec)} is not a valid line specification (n, -n or m;;n;;s)")


@fn("FilePrint", area=AREA, status="added",
    notes="FilePrint[\"file\"] prints the lines of a UTF-8 text file (through $Output, like Print) and returns "
          "Null; FilePrint[file, n] the first n lines, -n the last n, m;;n;;s a span (s may be negative); a missing "
          "file raises ValueError, a file that is not valid UTF-8 raises ValueError. Check: exact; tests capture "
          "the printed text.")
def _fileprint(name: Expr, *rest: Integer | Expr) -> Expr:
    if len(rest) > 1:
        raise ValueError("FilePrint: called with too many arguments")
    p = _existing(name, "FilePrint")
    if os.path.isdir(p):
        raise ValueError(f"FilePrint: {p!r} is a directory")
    with open(p, "rb") as fh:
        data = fh.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"FilePrint: {p!r} is not a UTF-8 text file")
    lines = re.split(r"\r\n|\n|\r", text)
    if lines and lines[-1] == "":
        lines.pop()
    if rest:
        lines = _span_lines(lines, rest[0], "FilePrint")
    from .wl_errors import print_line
    print_line("\n".join(lines))
    return NULL


# ═══════════════════════════════════════════ FileNames, FindFile ═══════════════════════════════════════════

def _abbrev_regex(s):
    out = []
    for ch in s:
        if ch == "*":
            out.append(".*")
        elif ch == "@":
            out.append("[^A-Z]+")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _form_matcher(form, ic, fname):
    """A predicate on a base file name for one FileNames form."""
    if is_sym(form, "All"):
        return lambda nm: True
    if isinstance(form, str):
        rx = re.compile(_abbrev_regex(form), re.I | re.S if ic else re.S)
        return lambda nm: rx.fullmatch(nm) is not None
    if isinstance(form, Expr) and form.head == "Verbatim" and len(form.args) == 1 and isinstance(form.args[0], str):
        lit = form.args[0]
        return (lambda nm: nm.lower() == lit.lower()) if ic else (lambda nm: nm == lit)
    if isinstance(form, Expr) and form.head in ("StringExpression", "RegularExpression", "Alternatives", "Blank",
                                                 "BlankSequence", "BlankNullSequence", "Pattern", "Condition",
                                                 "PatternTest", "Repeated", "Except", "Shortest", "Longest"):
        def m(nm):
            r = ev(Expr("StringMatchQ", nm, form, Expr("Rule", Symbol("IgnoreCase"), ic)))
            if r is not True and r is not False:
                raise ValueError(f"{fname}: {_form(form)} is not a valid string pattern")
            return r
        return m
    if isinstance(form, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(form)} is not a file name form (a string pattern or All)")


def _forms(forms, ic, fname):
    """[(dirpart, predicate)] for a form or list of forms; a string form may carry a literal directory part."""
    items = list(forms.args) if is_list(forms) else [forms]
    out = []
    for f in items:
        if isinstance(f, Expr) and f.head == "File" and len(f.args) == 1 and isinstance(f.args[0], str):
            f = Expr("Verbatim", f.args[0])
        lit = f.args[0] if isinstance(f, Expr) and f.head == "Verbatim" and len(f.args) == 1 \
            and isinstance(f.args[0], str) else None
        s = f if isinstance(f, str) else lit
        if s is not None and (os.sep in s or "/" in s):
            k = max(s.rfind("/"), s.rfind(os.sep))
            dpart, base = s[:k + 1] or "/", s[k + 1:]
            if isinstance(f, str) and any(c in dpart for c in "*@"):
                raise NotImplementedError(f"{fname}: wildcards in the directory part of {s!r} are not supported")
            if base == "":
                dpart, base = s.rstrip("/" + os.sep) or s, None
                k = max(dpart.rfind("/"), dpart.rfind(os.sep))
                dpart, base = dpart[:k + 1], dpart[k + 1:]
            pred = _form_matcher(base if isinstance(f, str) else Expr("Verbatim", base), ic, fname)
            out.append((_expanduser(dpart), pred))
        else:
            out.append(("", _form_matcher(f, ic, fname)))
    return out


def _level_spec(n, fname):
    if is_sym(n, "Infinity") or (isinstance(n, Expr) and n.head == "DirectedInfinity"):
        return 1, None
    if K.is_int(n) and not isinstance(n, bool):
        if int(n) < 1:
            raise ValueError(f"{fname}: the level must be a positive integer or Infinity")
        return 1, int(n)
    if is_list(n) and len(n.args) in (1, 2):
        vals = []
        for v in n.args:
            if is_sym(v, "Infinity") or (isinstance(v, Expr) and v.head == "DirectedInfinity"):
                vals.append(None)
            elif K.is_int(v) and not isinstance(v, bool) and int(v) >= 1:
                vals.append(int(v))
            else:
                raise ValueError(f"{fname}: {_form(n)} is not a valid level specification")
        lo, hi = vals[0], vals[-1]
        if lo is None:
            raise ValueError(f"{fname}: {_form(n)} is not a valid level specification")
        if hi is not None and hi < lo:
            raise ValueError(f"{fname}: {_form(n)} is not a valid level specification")
        return lo, hi
    if isinstance(n, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(n)} is not a valid level specification (n, {{n}}, {{m, n}} or Infinity)")


def _listdir(d):
    try:
        return sorted(os.listdir(d))
    except OSError:
        return []


def _sort_strings(names):
    return ev(Expr("Sort", L(*names)))


def _ignorecase(o, fname):
    v = o.get("IgnoreCase", Symbol("Automatic"))
    if is_sym(v, "Automatic"):
        return os.name == "nt"
    if v is True or v is False:
        return v
    raise ValueError(f"{fname}: IgnoreCase must be True, False or Automatic")


@fn("FileNames", area=AREA, status="added",
    notes="FileNames[] lists the files and directories in Directory[]; FileNames[form] those whose names match "
          "(abbreviated patterns with * and @, Verbatim[\"s\"], StringExpression, RegularExpression, Alternatives "
          "of these, All); a list of forms matches any; a form may start with a literal directory part "
          "(\"dir/*.txt\"); FileNames[forms, dir | {dir1, ...}] searches those directories and gives "
          "FileNameJoin[{dir, name}]; FileNames[forms, dirs, n] also subdirectories down to level n, {n} exactly "
          "level n, Infinity all (symbolic links to directories are not followed); IgnoreCase -> True | False | "
          "Automatic (case-insensitive only on Windows); results sorted as by Sort; missing directories contribute "
          "nothing. Wildcards in a directory part raise NotImplementedError. Check: exact (directory listing); "
          "tests build a known tree in a temporary directory.")
def _filenames(*args: Expr) -> ListOf[String]:
    args, o = _opts(args, ("IgnoreCase",), "FileNames")
    if len(args) > 3:
        raise ValueError("FileNames: called with too many arguments")
    ic = _ignorecase(o, "FileNames")
    forms = args[0] if args else Symbol("All")
    lo, hi = _level_spec(args[2], "FileNames") if len(args) == 3 else (1, 1)
    if len(args) >= 2:
        dirs = args[1]
        dl = list(dirs.args) if is_list(dirs) else [dirs]
        roots = [(_name(d, "FileNames", "directory name"), True) for d in dl]
    else:
        roots = [("", False)]
    specs = _forms(forms, ic, "FileNames")
    found = []
    seen = set()
    for root, given in roots:
        for dpart, pred in specs:
            if dpart and os.path.isabs(dpart):
                base_disp, base_fs = dpart, dpart
            else:
                base_disp = os.path.join(root, dpart) if root else dpart
                base_fs = os.path.join(_expanduser(root) if root else os.getcwd(), dpart)

            def walk(fs, disp, level):
                for e in _listdir(fs):
                    full = os.path.join(fs, e)
                    shown = os.path.join(disp, e) if disp else e
                    if level >= lo and pred(e) and shown not in seen:
                        seen.add(shown)
                        found.append(shown)
                    if (hi is None or level < hi) and os.path.isdir(full) and not os.path.islink(full):
                        walk(full, shown, level + 1)
            walk(base_fs, base_disp, 1)
    return _sort_strings(found)


def _path_dirs(o, fname):
    v = o.get("Path")
    if v is None:
        v = ev(Symbol("$Path"))
    out = []

    def flat(x):
        if is_list(x):
            for y in x.args:
                flat(y)
        elif isinstance(x, str):
            out.append(x)
        elif isinstance(x, Expr) and x.head == "File" and x.args and isinstance(x.args[0], str):
            out.append(x.args[0])
        else:
            raise TypeError(f"{fname}: the Path setting {_form(x)} must be a list of directory names")
    flat(v)
    return [os.getcwd() if d == "." else _expanduser(d) for d in out]


_INIT_FILES = (os.path.join("Kernel", "init.wl"), os.path.join("Kernel", "init.m"), "init.wl", "init.m")


def _resolve_in_dir(p):
    if os.path.isfile(p):
        return os.path.abspath(p)
    if os.path.isdir(p):
        for i in _INIT_FILES:
            q = os.path.join(p, i)
            if os.path.isfile(q):
                return os.path.abspath(q)
    return None


def find_file(name, o, fname):
    """FindFile: an absolute name, or $Failed (None)."""
    s = _expanduser(name)
    if re.fullmatch(r"(?:[A-Za-z$][A-Za-z0-9$]*`)+", s):
        rel = os.path.join(*s.rstrip("`").split("`"))
        cands = [rel + ".wl", rel + ".m", rel]
    else:
        cands = [s]
    if os.path.isabs(s):
        for c in cands:
            r = _resolve_in_dir(c)
            if r:
                return r
        return None
    for d in _path_dirs(o, fname):
        for c in cands:
            r = _resolve_in_dir(os.path.join(d, c))
            if r:
                return r
    return None


@fn("FindFile", area=AREA, status="added",
    notes="FindFile[\"name\"] searches the directories of $Path (or the Path option) in order and gives the "
          "absolute name of the first file found; a directory resolves to its Kernel/init.wl, Kernel/init.m, "
          "init.wl or init.m; \"ctx`\" looks for ctx.wl, ctx.m or a ctx directory (MX files are not supported); "
          "absolute names are checked directly; the documented result $Failed when nothing is found. File[...] "
          "accepted. Check: exact; tests create files in a temporary directory and search with Path.")
def _findfile(name: Expr, *rest: Option) -> String | Expr:
    args, o = _opts(rest, ("Path",), "FindFile")
    if args:
        raise ValueError("FindFile: called with too many arguments")
    s = _name(name, "FindFile")
    if s == "":
        raise ValueError("FindFile: the file name must not be empty")
    r = find_file(s, o, "FindFile")
    return _FAILED if r is None else r


# ═══════════════════════════════════════════ FileSystemMap, FileSystemScan ═══════════════════════════════════════════

_FSMAP_OPTS = ("IncludeDirectories", "FileNameForms", "MaxItems")


def _fs_opts(o, fname):
    incl = _bool_opt(o, "IncludeDirectories", fname, False)
    forms = o.get("FileNameForms", Symbol("All"))
    preds = [p for _d, p in _forms(forms, _ignorecase({}, fname), fname)]
    mi = o.get("MaxItems", Symbol("Infinity"))
    if is_sym(mi, "Infinity") or is_sym(mi, "All"):
        mi = None
    elif K.is_int(mi) and not isinstance(mi, bool) and int(mi) >= 0:
        mi = int(mi)
    else:
        raise ValueError(f"{fname}: MaxItems must be a non-negative integer or Infinity")
    return incl, preds, mi


def _root_dir(root, fname):
    p = _abs(_path(root, fname, "directory name"))
    if not os.path.isdir(p):
        raise ValueError(f"{fname}: {_name(root, fname)!r} is not an existing directory")
    return p


@fn("FileSystemMap", area=AREA, status="added",
    notes="FileSystemMap[f, root] gives <|name -> f[full name], ...|> for the files directly in root; "
          "FileSystemMap[f, root, n] nests subdirectories down to level n as associations (Infinity: all); "
          "{m, n} includes only files from level m to n; options FileNameForms (forms as in FileNames), "
          "IncludeDirectories -> True (a directory's association gets All -> f[dir]; directories at the last level "
          "become name -> f[dir]) and MaxItems (at most n entries per directory, in sorted order); symbolic links "
          "to directories are not followed. The 4-argument form FileSystemMap[f, root, lev, r] raises "
          "NotImplementedError. Check: exact; tests compare with FileNames and FileByteCount on a known tree.")
def _filesystemmap(f: Fn, root: Expr, *rest: Integer | ListOf[Integer] | Option) -> Assoc:
    args, o = _opts(rest, _FSMAP_OPTS, "FileSystemMap")
    if len(args) > 1:
        if len(args) == 2:
            raise NotImplementedError("FileSystemMap: combining levels (the 4-argument form) is not supported")
        raise ValueError("FileSystemMap: called with too many arguments")
    lo, hi = _level_spec(args[0], "FileSystemMap") if args else (1, 1)
    incl, preds, mi = _fs_opts(o, "FileSystemMap")
    base = _root_dir(root, "FileSystemMap")

    def walk(d, level):
        rules = []
        for e in _listdir(d):
            if mi is not None and len(rules) >= mi:
                break
            full = os.path.join(d, e)
            if os.path.isdir(full) and not os.path.islink(full):
                if hi is None or level < hi:
                    sub = walk(full, level + 1)
                    if incl:
                        sub = [Expr("Rule", Symbol("All"), apply(f, full))] + sub
                    rules.append(Expr("Rule", e, Expr("Association", *sub)))
                elif incl and level >= lo and any(p(e) for p in preds):
                    rules.append(Expr("Rule", e, apply(f, full)))
            elif level >= lo and any(p(e) for p in preds):
                rules.append(Expr("Rule", e, apply(f, full)))
        return rules
    return Expr("Association", *walk(base, 1))


@fn("FileSystemScan", area=AREA, status="added",
    notes="FileSystemScan[f, root] evaluates f[full name] for each file directly in root (sorted order) and "
          "returns Null; FileSystemScan[f, root, n] goes down to level n ({m, n}: files from level m to n, "
          "Infinity: all); options FileNameForms, IncludeDirectories (f is also applied to directories) and "
          "MaxItems, as in FileSystemMap. Check: exact; tests collect the visited names with Sow/Reap.")
def _filesystemscan(f: Fn, root: Expr, *rest: Integer | ListOf[Integer] | Option) -> Expr:
    args, o = _opts(rest, _FSMAP_OPTS, "FileSystemScan")
    if len(args) > 1:
        raise ValueError("FileSystemScan: called with too many arguments")
    lo, hi = _level_spec(args[0], "FileSystemScan") if args else (1, 1)
    incl, preds, mi = _fs_opts(o, "FileSystemScan")
    base = _root_dir(root, "FileSystemScan")

    def walk(d, level):
        count = 0
        for e in _listdir(d):
            if mi is not None and count >= mi:
                break
            full = os.path.join(d, e)
            if os.path.isdir(full) and not os.path.islink(full):
                count += 1
                if incl and level >= lo and any(p(e) for p in preds):
                    apply(f, full)
                if hi is None or level < hi:
                    walk(full, level + 1)
            elif level >= lo and any(p(e) for p in preds):
                count += 1
                apply(f, full)
    walk(base, 1)
    return NULL


# ═══════════════════════════════════════════ creating, copying, deleting ═══════════════════════════════════════════

_CREATE_OPTS = ("CreateIntermediateDirectories", "OverwriteTarget")


def _abs(p):
    return os.path.normpath(os.path.join(os.getcwd(), p))


@fn("CreateFile", area=AREA, status="added", listable=True,
    notes="CreateFile[\"file\"] creates an empty file (relative to Directory[]) and returns its absolute name; "
          "CreateFile[] creates a new file in $TemporaryDirectory; CreateIntermediateDirectories -> True (default) "
          "creates missing parent directories; an existing file raises ValueError unless OverwriteTarget -> True, "
          "which truncates it to zero size; an existing directory of that name raises ValueError; Listable; "
          "File[...] accepted. Check: exact; tests check FileExistsQ and FileByteCount == 0.")
def _createfile(*args: Expr) -> String:
    args, o = _opts(args, _CREATE_OPTS, "CreateFile")
    inter = _bool_opt(o, "CreateIntermediateDirectories", "CreateFile", True)
    over = _bool_opt(o, "OverwriteTarget", "CreateFile", False)
    if len(args) > 1:
        raise ValueError("CreateFile: called with too many arguments")
    if not args:
        fd, p = tempfile.mkstemp(prefix="m", dir=temporary_directory())
        os.close(fd)
        return os.path.abspath(p)
    p = _abs(_path(args[0], "CreateFile"))
    if os.path.isdir(p):
        raise ValueError(f"CreateFile: {p!r} is an existing directory")
    if os.path.exists(p) and not over:
        raise ValueError(f"CreateFile: the file {p!r} already exists (use OverwriteTarget -> True to truncate it)")
    parent = os.path.dirname(p)
    if not os.path.isdir(parent):
        if not inter:
            raise ValueError(f"CreateFile: the directory {parent!r} does not exist")
        os.makedirs(parent, exist_ok=True)
    try:
        with open(p, "wb"):
            pass
    except OSError as e:
        raise _oserr("CreateFile", e, p)
    return p


@fn("CreateDirectory", area=AREA, status="added", listable=True,
    notes="CreateDirectory[\"dir\"] creates the directory (relative to Directory[]) and returns its absolute name; "
          "CreateDirectory[] creates a new directory in $TemporaryDirectory; CreateIntermediateDirectories -> True "
          "(default) creates missing parents, False raises ValueError when the parent is missing; an existing file "
          "or directory of that name raises ValueError (Wolfram: message and $Failed); Listable; File[...] "
          "accepted. Check: exact; tests check DirectoryQ and an empty FileNames listing.")
def _createdirectory(*args: Expr) -> String:
    args, o = _opts(args, ("CreateIntermediateDirectories",), "CreateDirectory")
    inter = _bool_opt(o, "CreateIntermediateDirectories", "CreateDirectory", True)
    if len(args) > 1:
        raise ValueError("CreateDirectory: called with too many arguments")
    if not args:
        return os.path.abspath(tempfile.mkdtemp(prefix="m", dir=temporary_directory()))
    p = _abs(_path(args[0], "CreateDirectory", "directory name"))
    if os.path.exists(p):
        raise ValueError(f"CreateDirectory: {p!r} already exists")
    parent = os.path.dirname(p)
    try:
        if inter:
            os.makedirs(p)
        else:
            if not os.path.isdir(parent):
                raise ValueError(f"CreateDirectory: the directory {parent!r} does not exist "
                                 f"(CreateIntermediateDirectories -> False)")
            os.mkdir(p)
    except OSError as e:
        raise _oserr("CreateDirectory", e, p)
    return p


def _target_free(dest, over, fname, src_is_dir=False):
    if os.path.lexists(dest):
        if not over:
            raise ValueError(f"{fname}: the target {dest!r} already exists")
        if os.path.isdir(dest) and not os.path.islink(dest):
            raise ValueError(f"{fname}: the target {dest!r} is a directory and cannot be overwritten")


@fn("CopyFile", area=AREA, status="added",
    notes="CopyFile[\"src\", \"dest\"] copies a file (contents, permissions and modification date) and returns the "
          "absolute name of dest; src must be an existing file (ValueError otherwise, also for a directory); dest "
          "must not exist unless OverwriteTarget -> True (an existing directory is never overwritten); URLs, cloud "
          "and local objects are not supported (TypeError for non-file specifications). Check: exact; tests "
          "compare contents (ReadString) and FileDate of source and copy.")
def _copyfile(src: Expr, dest: Expr, *rest: Option) -> String:
    args, o = _opts(rest, ("OverwriteTarget",), "CopyFile")
    if args:
        raise ValueError("CopyFile: called with too many arguments")
    over = _bool_opt(o, "OverwriteTarget", "CopyFile", False)
    s = _abs(_path(src, "CopyFile"))
    d = _abs(_path(dest, "CopyFile"))
    if not os.path.exists(s):
        raise ValueError(f"CopyFile: the file {s!r} does not exist")
    if os.path.isdir(s):
        raise ValueError(f"CopyFile: {s!r} is a directory (use CopyDirectory)")
    if os.path.exists(d) and os.path.samefile(s, d):
        raise ValueError(f"CopyFile: source and target are the same file ({s!r})")
    _target_free(d, over, "CopyFile")
    if not os.path.isdir(os.path.dirname(d)):
        raise ValueError(f"CopyFile: the directory {os.path.dirname(d)!r} does not exist")
    try:
        shutil.copy2(s, d)
    except OSError as e:
        raise _oserr("CopyFile", e, d)
    return d


@fn("RenameFile", area=AREA, status="added",
    notes="RenameFile[\"old\", \"new\"] renames (moves) a file, keeping its modification date, and returns the "
          "absolute new name; old must be an existing file (a directory raises ValueError: use RenameDirectory); "
          "new must not exist unless OverwriteTarget -> True (an existing directory is never replaced). "
          "Check: exact; tests check FileExistsQ of both names and the moved contents.")
def _renamefile(old: Expr, new: Expr, *rest: Option) -> String:
    args, o = _opts(rest, ("OverwriteTarget",), "RenameFile")
    if args:
        raise ValueError("RenameFile: called with too many arguments")
    over = _bool_opt(o, "OverwriteTarget", "RenameFile", False)
    s = _abs(_path(old, "RenameFile"))
    d = _abs(_path(new, "RenameFile"))
    if not os.path.lexists(s):
        raise ValueError(f"RenameFile: the file {s!r} does not exist")
    if os.path.isdir(s) and not os.path.islink(s):
        raise ValueError(f"RenameFile: {s!r} is a directory (use RenameDirectory)")
    if s == d:
        return d
    _target_free(d, over, "RenameFile")
    if not os.path.isdir(os.path.dirname(d)):
        raise ValueError(f"RenameFile: the directory {os.path.dirname(d)!r} does not exist")
    try:
        os.replace(s, d)
    except OSError:
        try:
            shutil.copy2(s, d)
            os.remove(s)
        except OSError as e:
            raise _oserr("RenameFile", e, d)
    return d


@fn("DeleteFile", area=AREA, status="added",
    notes="DeleteFile[\"file\"] or DeleteFile[{file1, ...}] deletes files and returns Null; every name is checked "
          "first: a missing file or a directory (use DeleteDirectory) raises ValueError and nothing is deleted; "
          "File[...] accepted. Check: exact; tests check FileExistsQ before and after.")
def _deletefile(files: Expr | ListOf[Expr]) -> Expr:
    items = list(files.args) if is_list(files) else [files]
    paths = []
    for x in items:
        p = _abs(_path(x, "DeleteFile"))
        if not os.path.lexists(p):
            raise ValueError(f"DeleteFile: the file {p!r} does not exist")
        if os.path.isdir(p) and not os.path.islink(p):
            raise ValueError(f"DeleteFile: {p!r} is a directory (use DeleteDirectory)")
        paths.append(p)
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass                            # listed twice
        except OSError as e:
            raise _oserr("DeleteFile", e, p)
    return NULL


@fn("DeleteDirectory", area=AREA, status="added",
    notes="DeleteDirectory[\"dir\"] deletes an empty directory and returns Null; a non-empty directory raises "
          "ValueError unless DeleteContents -> True, which deletes it with everything in it (symbolic links inside "
          "are removed, not followed); a missing directory or a file raises ValueError (Wolfram: message and "
          "$Failed); the current directory's ancestors can be deleted like any other; File[...] accepted. "
          "Check: exact; tests check DirectoryQ before and after and that a non-empty directory survives without "
          "DeleteContents.")
def _deletedirectory(d: Expr, *rest: Option) -> Expr:
    args, o = _opts(rest, ("DeleteContents",), "DeleteDirectory")
    if args:
        raise ValueError("DeleteDirectory: called with too many arguments")
    contents = _bool_opt(o, "DeleteContents", "DeleteDirectory", False)
    p = _abs(_path(d, "DeleteDirectory", "directory name"))
    if not os.path.lexists(p):
        raise ValueError(f"DeleteDirectory: the directory {p!r} does not exist")
    if os.path.islink(p) or not os.path.isdir(p):
        raise ValueError(f"DeleteDirectory: {p!r} is not a directory")
    try:
        if os.listdir(p):
            if not contents:
                raise ValueError(f"DeleteDirectory: the directory {p!r} is not empty (use DeleteContents -> True)")
            shutil.rmtree(p)
        else:
            os.rmdir(p)
    except OSError as e:
        raise _oserr("DeleteDirectory", e, p)
    return NULL


record("DeleteContents", AREA, "added",
       "DeleteContents -> True is the option of DeleteDirectory that deletes a non-empty directory with its contents "
       "(default False: a non-empty directory raises). Check: reference (Wolfram documentation) in tests.")


@fn("CopyDirectory", area=AREA, status="added",
    notes="CopyDirectory[\"dir1\", \"dir2\"] copies dir1 with all its files and subdirectories to dir2 (modification "
          "dates kept, symbolic links copied as links) and returns the absolute name of dir2; dir1 must be an "
          "existing directory and dir2 must not exist (ValueError otherwise); dir2 may not lie inside dir1. "
          "Check: exact; tests compare FileNames listings and contents of the two trees.")
def _copydirectory(d1: Expr, d2: Expr) -> String:
    s = _abs(_path(d1, "CopyDirectory", "directory name"))
    d = _abs(_path(d2, "CopyDirectory", "directory name"))
    if not os.path.isdir(s):
        raise ValueError(f"CopyDirectory: {s!r} is not an existing directory")
    if os.path.lexists(d):
        raise ValueError(f"CopyDirectory: the target {d!r} already exists")
    if (d + os.sep).startswith(s.rstrip(os.sep) + os.sep):
        raise ValueError(f"CopyDirectory: the target {d!r} lies inside the source directory")
    try:
        shutil.copytree(s, d, symlinks=True)
    except (OSError, shutil.Error) as e:
        raise ValueError(f"CopyDirectory: {e}")
    return d


@fn("RenameDirectory", area=AREA, status="added",
    notes="RenameDirectory[\"dir1\", \"dir2\"] renames (moves) a directory and returns the absolute new name; dir1 "
          "must be an existing directory, dir2 must not exist (ValueError otherwise) and may not lie inside dir1. "
          "Check: exact; tests check DirectoryQ of both names and the moved contents.")
def _renamedirectory(d1: Expr, d2: Expr) -> String:
    s = _abs(_path(d1, "RenameDirectory", "directory name"))
    d = _abs(_path(d2, "RenameDirectory", "directory name"))
    if not os.path.isdir(s) or os.path.islink(s):
        raise ValueError(f"RenameDirectory: {s!r} is not an existing directory")
    if os.path.lexists(d):
        raise ValueError(f"RenameDirectory: the target {d!r} already exists")
    if (d + os.sep).startswith(s.rstrip(os.sep) + os.sep):
        raise ValueError(f"RenameDirectory: the target {d!r} lies inside the source directory")
    if not os.path.isdir(os.path.dirname(d)):
        raise ValueError(f"RenameDirectory: the directory {os.path.dirname(d)!r} does not exist")
    try:
        shutil.move(s, d)
    except (OSError, shutil.Error) as e:
        raise ValueError(f"RenameDirectory: {e}")
    return d


# ═══════════════════════════════════════════ working directory ═══════════════════════════════════════════

DIRECTORY_STACK = []          # most recent first, as DirectoryStack[] shows it


@fn("Directory", area=AREA, status="added",
    notes="Directory[] gives the current working directory (absolute name). Check: exact (os.getcwd); "
          "SetDirectory/ResetDirectory round trips in tests.")
def _directory() -> String:
    return os.getcwd()


@fn("SetDirectory", area=AREA, status="added",
    notes="SetDirectory[\"dir\"] makes dir (relative to the current directory) the working directory, pushes the "
          "previous one onto DirectoryStack[] and returns the new absolute name; SetDirectory[] goes to "
          "$HomeDirectory; a missing directory or a file raises ValueError; File[...] accepted. The working "
          "directory is process-wide, as in Wolfram. Check: self-check (Directory[] is compared with the target "
          "after changing; ArithmeticError on mismatch); round trips in tests.")
def _setdirectory(*args: Expr) -> String:
    if len(args) > 1:
        raise ValueError("SetDirectory: called with too many arguments")
    target = home_directory() if not args else _abs(_path(args[0], "SetDirectory", "directory name"))
    if not os.path.isdir(target):
        raise ValueError(f"SetDirectory: {target!r} is not an existing directory")
    old = os.getcwd()
    os.chdir(target)
    DIRECTORY_STACK.insert(0, old)
    new = os.getcwd()
    if os.path.realpath(new) != os.path.realpath(target):
        raise ArithmeticError("SetDirectory: result failed verification")
    return new


@fn("ResetDirectory", area=AREA, status="added",
    notes="ResetDirectory[] returns to the directory that was current before the last SetDirectory (popping "
          "DirectoryStack[]) and gives its name; an empty stack raises ValueError (Wolfram: message); a stacked "
          "directory that no longer exists raises ValueError (it stays popped). Check: exact; round trips in "
          "tests.")
def _resetdirectory() -> String:
    if not DIRECTORY_STACK:
        raise ValueError("ResetDirectory: the directory stack is empty")
    d = DIRECTORY_STACK.pop(0)
    try:
        os.chdir(d)
    except OSError as e:
        raise _oserr("ResetDirectory", e, d)
    return os.getcwd()


@fn("DirectoryStack", area=AREA, status="added",
    notes="DirectoryStack[] gives the directories saved by SetDirectory, most recent first ({} initially). "
          "Check: exact; tests after SetDirectory/ResetDirectory.")
def _directorystack() -> ListOf[String]:
    return L(*DIRECTORY_STACK)


# ═══════════════════════════════════════════ streams: objects and registry ═══════════════════════════════════════════
#
# Serial numbers are shared with StringToStream (wl_text._STREAM_COUNTER), so they are unique across all streams.
# File and method streams live in STREAMS; string streams in wl_text.STRING_STREAMS ({"text", "pos"}: positions
# count characters, and Byte reads character codes, as Wolfram documents for StringToStream).

STREAMS = {}
_IN_METHODS = {}
_OUT_METHODS = {}

_ENCODINGS = {"UTF8": "utf-8", "UTF-8": "utf-8", "ASCII": "ascii", "PrintableASCII": "ascii",
              "ISOLatin1": "latin-1", "ISO8859-1": "latin-1", "Latin1": "latin-1"}


def _next_serial():
    _wtext._STREAM_COUNTER[0] += 1
    return _wtext._STREAM_COUNTER[0]


def _encoding(o, fname):
    v = o.get("CharacterEncoding", Symbol("Automatic"))
    if is_sym(v, "Automatic") or v is None:
        return "utf-8"
    if isinstance(v, str):
        if v in _ENCODINGS:
            return _ENCODINGS[v]
        raise NotImplementedError(f"{fname}: the character encoding {v!r} is not supported (UTF8, ASCII and "
                                  f"ISOLatin1 are)")
    raise ValueError(f"{fname}: CharacterEncoding must be a string or Automatic, not {_form(v)}")


class _Stream:
    """An open file or method stream."""

    def __init__(self, kind, name, raw, binary=False, enc="utf-8", method="File", ropts=None, fmt="InputForm"):
        self.kind, self.name, self.raw = kind, name, raw
        self.binary, self.enc, self.method = binary, enc, method
        self.ropts = ropts or {}
        self.fmt = fmt
        self.n = _next_serial()
        STREAMS[self.n] = self

    def expr(self):
        return Expr("InputStream" if self.kind == "in" else "OutputStream", self.name, self.n)

    def close(self):
        STREAMS.pop(self.n, None)
        self.raw.close()


class _StdStream:
    kind = "out"
    binary = False
    enc = "utf-8"
    fmt = "OutputForm"

    def __init__(self, name, n):
        self.name, self.n = name, n

    def expr(self):
        return Expr("OutputStream", self.name, self.n)

    @property
    def raw(self):
        return self

    def write(self, data):
        f = sys.stdout if self.name == "stdout" else sys.stderr
        f.write(data.decode("utf-8", errors="replace"))
        f.flush()

    def tell(self):
        raise NotImplementedError(f"StreamPosition: the position of {self.name} is not available")

    def seek(self, *a):
        raise NotImplementedError(f"SetStreamPosition: {self.name} cannot be repositioned")


_STD = {1: _StdStream("stdout", 1), 2: _StdStream("stderr", 2)}


class _StringStream:
    """Adapter for an InputStream[String, n] of StringToStream."""
    kind = "in"
    binary = False
    enc = None
    method = "String"
    ropts = {}

    def __init__(self, n, state):
        self.n, self.state = n, state
        self.name = Symbol("String")

    def expr(self):
        return Expr("InputStream", Symbol("String"), self.n)

    def close(self):
        _wtext.STRING_STREAMS.pop(self.n, None)


def close_all_streams():
    """Close every file and method stream (used by the test-suite teardown)."""
    for s in list(STREAMS.values()):
        try:
            s.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            STREAMS.pop(s.n, None)


# ── method streams (DefineInputStreamMethod / DefineOutputStreamMethod) ──

def _pair(res, fname, what):
    if not (is_list(res) and len(res.args) == 2):
        raise ValueError(f"{fname}: the stream method's {what} must return a list {{value, state}}, not {_form(res)}")
    return res.args[0], res.args[1]


def _byte_list(x, fname):
    if _is_byte_array(x):
        return _ba_bytes(x, fname)
    if is_list(x) and all(K.is_int(b) and not isinstance(b, bool) and 0 <= int(b) <= 255 for b in x.args):
        return bytes(int(b) for b in x.args)
    raise ValueError(f"{fname}: the stream method's ReadFunction must return a list of bytes, not {_form(x)}")


class _MethodIn:
    """Raw binary reader over a user-defined input stream method; everything read is buffered (so the stream can
    always be repositioned)."""

    def __init__(self, funcs, state, caller):
        self.f, self.state, self.caller = funcs, state, caller
        self.buf = bytearray()
        self.pos = 0
        self.eof = False

    def _call(self, key, *args):
        return apply(self.f[key], self.state, *args)

    def _fill(self, need):
        empty = 0
        while (need is None or len(self.buf) < need) and not self.eof:
            want = 65536 if need is None else max(need - len(self.buf), 4096)
            if "ReadFunction" not in self.f:
                self.eof = True
                break
            if "WaitForInputFunction" in self.f:
                self.state = self._call("WaitForInputFunction")
            data, self.state = _pair(self._call("ReadFunction", want), self.caller, "ReadFunction")
            data = _byte_list(data, self.caller)
            self.buf.extend(data)
            if len(data) < want:
                flag = True
                if "EndOfFileQFunction" in self.f:
                    flag, self.state = _pair(self._call("EndOfFileQFunction"), self.caller, "EndOfFileQFunction")
                if flag is True:
                    self.eof = True
                    break
                if "ErrorTextFunction" in self.f:
                    err, self.state = _pair(self._call("ErrorTextFunction"), self.caller, "ErrorTextFunction")
                    if isinstance(err, str):
                        if "ClearErrorFunction" in self.f:
                            _x, self.state = _pair(self._call("ClearErrorFunction"), self.caller,
                                                   "ClearErrorFunction")
                        raise ValueError(f"{self.caller}: the stream method reported an error: {err}")
                empty = empty + 1 if not data else 0
                if empty > 1000:
                    raise ValueError(f"{self.caller}: the stream method returns no input but is not at its end")

    def read(self, n=-1):
        if n is None or n < 0:
            self._fill(None)
            out = bytes(self.buf[self.pos:])
        else:
            self._fill(self.pos + n)
            out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += len(out)
        return out

    def tell(self):
        return self.pos

    def seek(self, p, whence=0):
        if whence == 2:
            self._fill(None)
            self.pos = len(self.buf) + p
        else:
            self._fill(p)
            self.pos = min(p, len(self.buf))
        return self.pos

    def close(self):
        if "CloseFunction" in self.f:
            apply(self.f["CloseFunction"], self.state)


class _MethodOut:
    def __init__(self, funcs, state, caller):
        self.f, self.state, self.caller = funcs, state, caller
        self.count = 0

    def write(self, data):
        if not data:
            return
        if "WriteFunction" not in self.f:
            return                      # documented default: output is ignored
        n, self.state = _pair(apply(self.f["WriteFunction"], self.state, L(*data)), self.caller, "WriteFunction")
        if not (K.is_int(n) and int(n) == len(data)):
            raise ValueError(f"{self.caller}: the stream method wrote {_form(n)} of {len(data)} bytes")
        self.count += len(data)

    def tell(self):
        if "StreamPositionFunction" in self.f:
            n, self.state = _pair(apply(self.f["StreamPositionFunction"], self.state), "StreamPosition",
                                  "StreamPositionFunction")
            if not K.is_int(n):
                raise ValueError(f"StreamPosition: the stream method gave the position {_form(n)}")
            return int(n)
        return self.count

    def seek(self, *a):
        raise NotImplementedError("SetStreamPosition: streams of a user-defined output method cannot be repositioned")

    def close(self):
        if "FlushFunction" in self.f:
            _x, self.state = _pair(apply(self.f["FlushFunction"], self.state), "Close", "FlushFunction")
        if "CloseFunction" in self.f:
            apply(self.f["CloseFunction"], self.state)


_IN_FUNCS = ("NameTestFunction", "ConstructorFunction", "CloseFunction", "ReadFunction", "EndOfFileQFunction",
             "WaitForInputFunction", "SeekFunction", "StreamPositionFunction", "StreamSizeFunction",
             "SeekableQFunction", "ErrorTextFunction", "ClearErrorFunction", "OptionChangesFunction")
_OUT_FUNCS = ("NameTestFunction", "ConstructorFunction", "CloseFunction", "WriteFunction", "FlushFunction",
              "StreamPositionFunction", "OptionChangesFunction")


def _define_method(name, funcs, table, allowed, fname):
    if not isinstance(name, str):
        if isinstance(name, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the method name must be a string, not {_form(name)}")
    if name in ("File", "String") or name in table:
        raise ValueError(f"{fname}: a stream method named {name!r} already exists")
    if not is_list(funcs):
        raise TypeError(f"{fname}: the stream functions must be a list of \"name\" -> function rules")
    d = {}
    for r in funcs.args:
        k = _opt_name(r)
        if k is None:
            raise TypeError(f"{fname}: {_form(r)} is not a \"name\" -> function rule")
        if k not in allowed:
            raise ValueError(f"{fname}: unknown stream function {k!r}")
        d[k] = r.args[1]
    if "ConstructorFunction" not in d:
        raise ValueError(f"{fname}: the \"ConstructorFunction\" must be given")
    table[name] = d
    return NULL


@fn("DefineInputStreamMethod", area=AREA, status="added",
    notes="DefineInputStreamMethod[\"name\", {\"ConstructorFunction\" -> f, \"ReadFunction\" -> g, ...}] defines an "
          "input stream method used by OpenRead[..., Method -> \"name\"] (or Method -> {\"name\", opts}), or by "
          "OpenRead with Method -> Automatic when its \"NameTestFunction\" gives True; supported functions: "
          "ConstructorFunction (required; f[name, caller, opts] -> {True, state}), ReadFunction (g[state, n] -> "
          "{bytes, state}), EndOfFileQFunction, ErrorTextFunction/ClearErrorFunction (an error raises ValueError "
          "after clearing it), WaitForInputFunction, CloseFunction, NameTestFunction; Seek-, StreamPosition-, "
          "StreamSize-, SeekableQ- and OptionChangesFunction are accepted but not needed: the stream buffers what it "
          "has read, so StreamPosition/SetStreamPosition always work. An existing name raises ValueError. Returns "
          "Null. Check: reference (the Wolfram documentation's \"StringName\" example) in tests.")
def _defineinputstreammethod(name: String, funcs: ListOf[Rule]) -> Expr:
    return _define_method(name, funcs, _IN_METHODS, _IN_FUNCS, "DefineInputStreamMethod")


@fn("DefineOutputStreamMethod", area=AREA, status="added",
    notes="DefineOutputStreamMethod[\"name\", {\"ConstructorFunction\" -> f, \"WriteFunction\" -> g, ...}] defines an "
          "output stream method used by OpenWrite/OpenAppend[..., Method -> \"name\"] or Method -> Automatic with a "
          "\"NameTestFunction\"; f[name, isAppend, caller, opts] -> {True, state}; g[state, {bytes}] -> {count, "
          "state} (a short count raises ValueError); FlushFunction and CloseFunction are called on Close; "
          "StreamPositionFunction is used by StreamPosition; without a WriteFunction output is ignored (as "
          "documented). An existing name raises ValueError. Returns Null. Check: reference (round trip through a "
          "collecting method) in tests.")
def _defineoutputstreammethod(name: String, funcs: ListOf[Rule]) -> Expr:
    return _define_method(name, funcs, _OUT_METHODS, _OUT_FUNCS, "DefineOutputStreamMethod")


def _remove_method(name, table, fname):
    if not isinstance(name, str):
        if isinstance(name, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the method name must be a string, not {_form(name)}")
    if name not in table:
        raise ValueError(f"{fname}: there is no stream method named {name!r}")
    del table[name]
    return NULL


@fn("RemoveInputStreamMethod", area=AREA, status="added",
    notes="RemoveInputStreamMethod[\"name\"] removes a method defined with DefineInputStreamMethod (streams already "
          "open keep working) and returns Null; an unknown name raises ValueError. Check: exact; "
          "$InputStreamMethods before and after in tests.")
def _removeinputstreammethod(name: String) -> Expr:
    return _remove_method(name, _IN_METHODS, "RemoveInputStreamMethod")


@fn("RemoveOutputStreamMethod", area=AREA, status="added",
    notes="RemoveOutputStreamMethod[\"name\"] removes a method defined with DefineOutputStreamMethod and returns "
          "Null; an unknown name raises ValueError. Check: exact; $OutputStreamMethods before and after in tests.")
def _removeoutputstreammethod(name: String) -> Expr:
    return _remove_method(name, _OUT_METHODS, "RemoveOutputStreamMethod")


def _method_opt(o, fname):
    """(method name or None for Automatic, extra method options as a list of rules)."""
    m = o.get("Method", Symbol("Automatic"))
    if is_sym(m, "Automatic"):
        return None, []
    if isinstance(m, str):
        return m, []
    if is_list(m) and m.args and isinstance(m.args[0], str) and all(_opt_name(r) for r in m.args[1:]):
        return m.args[0], list(m.args[1:])
    raise ValueError(f"{fname}: Method must be a method name, {{name, opts...}} or Automatic, not {_form(m)}")


def _name_test(funcs, name):
    t = funcs.get("NameTestFunction")
    return t is not None and apply(t, name) is True


# ═══════════════════════════════════════════ opening streams ═══════════════════════════════════════════

_READ_OPTS = ("RecordSeparators", "WordSeparators", "NullRecords", "NullWords", "TokenWords")
_OPENREAD_OPTS = ("BinaryFormat", "CharacterEncoding", "Method", "Path") + _READ_OPTS
_OPENWRITE_OPTS = ("BinaryFormat", "CharacterEncoding", "FormatType", "Method", "PageWidth", "NumberMarks",
                   "TotalWidth")


def _resolve_read(name, o, fname):
    """The absolute name of an existing file to read: the name itself or a $Path (Path option) directory."""
    s = _expanduser(name)
    if os.path.isabs(s) or os.path.exists(s):
        p = _abs(s)
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            raise ValueError(f"{fname}: {p!r} is a directory")
        raise ValueError(f"{fname}: cannot open {name!r}: the file does not exist")
    for d in _path_dirs(o, fname):
        p = os.path.join(d, s)
        if os.path.isfile(p):
            return _abs(p)
    raise ValueError(f"{fname}: cannot open {name!r}: the file does not exist")


def _open_read(x, o, fname):
    name = _name(x, fname)
    if name == "":
        raise ValueError(f"{fname}: the file name must not be empty")
    binary = _bool_opt(o, "BinaryFormat", fname, False)
    enc = _encoding(o, fname)
    ropts = {k: v for k, v in o.items() if k in _READ_OPTS}
    _read_opts(ropts, fname)            # validate now
    meth, mopts = _method_opt(o, fname)
    if meth is None:
        meth = next((m for m, f in _IN_METHODS.items() if _name_test(f, name)), "File")
    if meth == "File":
        if name.startswith("!"):
            raise NotImplementedError(f"{fname}: pipes (\"!command\") are not supported")
        p = _resolve_read(name, o, fname)
        try:
            raw = open(p, "rb")
        except OSError as e:
            raise _oserr(fname, e, p)
        return _Stream("in", p, raw, binary, enc, "File", ropts)
    funcs = _IN_METHODS.get(meth)
    if funcs is None:
        raise ValueError(f"{fname}: unknown input stream method {meth!r} (see $InputStreamMethods)")
    res = apply(funcs["ConstructorFunction"], name, Symbol(fname), L(*mopts))
    ok, state = _pair(res, fname, "ConstructorFunction")
    if ok is not True:
        raise ValueError(f"{fname}: the stream method {meth!r} could not open {name!r}")
    return _Stream("in", name, _MethodIn(funcs, state, fname), binary, enc, meth, ropts)


def _format_type(o, fname):
    ft = o.get("FormatType", Symbol("InputForm"))
    if is_sym(ft, "InputForm") or is_sym(ft, "Automatic"):
        return "InputForm"
    if is_sym(ft, "OutputForm"):
        return "OutputForm"
    raise NotImplementedError(f"{fname}: FormatType {_form(ft)} is not supported (InputForm and OutputForm are)")


def _open_write(x, o, fname, append):
    for k, ok in (("PageWidth", ("Infinity",)), ("TotalWidth", ("Infinity",)), ("NumberMarks", ("$NumberMarks",))):
        if k in o and not any(is_sym(o[k], v) for v in ok + ("Automatic",)):
            raise NotImplementedError(f"{fname}: {k} -> {_form(o[k])} is not supported (lines are never wrapped)")
    binary = _bool_opt(o, "BinaryFormat", fname, False)
    enc = _encoding(o, fname)
    fmt = _format_type(o, fname)
    meth, mopts = _method_opt(o, fname)
    if x is None:
        if meth not in (None, "File"):
            raise ValueError(f"{fname}: a stream method needs a stream name")
        fd, p = tempfile.mkstemp(prefix="m", dir=temporary_directory())
        os.close(fd)
        return _Stream("out", os.path.abspath(p), open(p, "wb"), binary, enc, "File", fmt=fmt)
    name = _name(x, fname)
    if name == "":
        raise ValueError(f"{fname}: the file name must not be empty")
    if meth is None:
        meth = next((m for m, f in _OUT_METHODS.items() if _name_test(f, name)), "File")
    if meth == "File":
        if name.startswith("!"):
            raise NotImplementedError(f"{fname}: pipes (\"!command\") are not supported")
        p = _abs(_expanduser(name))
        if os.path.isdir(p):
            raise ValueError(f"{fname}: {p!r} is a directory")
        if not os.path.isdir(os.path.dirname(p)):
            raise ValueError(f"{fname}: cannot open {name!r}: the directory {os.path.dirname(p)!r} does not exist")
        try:
            raw = open(p, "ab" if append else "wb")
        except OSError as e:
            raise _oserr(fname, e, p)
        return _Stream("out", p, raw, binary, enc, "File", fmt=fmt)
    funcs = _OUT_METHODS.get(meth)
    if funcs is None:
        raise ValueError(f"{fname}: unknown output stream method {meth!r} (see $OutputStreamMethods)")
    res = apply(funcs["ConstructorFunction"], name, bool(append), Symbol(fname), L(*mopts))
    ok, state = _pair(res, fname, "ConstructorFunction")
    if ok is not True:
        raise ValueError(f"{fname}: the stream method {meth!r} could not open {name!r}")
    return _Stream("out", name, _MethodOut(funcs, state, fname), binary, enc, meth, fmt=fmt)


@fn("OpenRead", area=AREA, status="added",
    notes="OpenRead[\"file\"] opens a file for reading and gives InputStream[\"full name\", n] (n unique across all "
          "streams); relative names are looked up in Directory[] and then in $Path (or the Path option); options "
          "BinaryFormat, CharacterEncoding (UTF8 default, ASCII, ISOLatin1), Method (\"File\", a method of "
          "DefineInputStreamMethod, {method, opts}), and the stream defaults RecordSeparators, WordSeparators, "
          "NullRecords, NullWords, TokenWords used by Read; a missing file or a directory raises ValueError "
          "(Wolfram: message and $Failed); pipes \"!cmd\" raise NotImplementedError. Check: exact; "
          "Streams[] membership and Read round trips in tests.")
def _openread(x: Expr, *rest: Option) -> Expr:
    args, o = _opts(rest, _OPENREAD_OPTS, "OpenRead")
    if args:
        raise ValueError("OpenRead: called with too many arguments")
    return _open_read(x, o, "OpenRead").expr()


@fn("OpenWrite", area=AREA, status="added",
    notes="OpenWrite[\"file\"] opens a file for writing, deleting its contents, and gives OutputStream[\"full "
          "name\", n]; OpenWrite[] opens a new file in $TemporaryDirectory; options BinaryFormat, CharacterEncoding "
          "(UTF8 default; ASCII writes non-ASCII characters as \\:hhhh escapes), FormatType (InputForm default, "
          "OutputForm), Method (\"File\" or a DefineOutputStreamMethod method). Lines are never wrapped: PageWidth, "
          "TotalWidth and NumberMarks other than their no-op values raise NotImplementedError; a missing directory "
          "raises ValueError; pipes raise NotImplementedError. Check: exact; Write/Read round trips in tests.")
def _openwrite(*args: Expr) -> Expr:
    args, o = _opts(args, _OPENWRITE_OPTS, "OpenWrite")
    if len(args) > 1:
        raise ValueError("OpenWrite: called with too many arguments")
    return _open_write(args[0] if args else None, o, "OpenWrite", False).expr()


@fn("OpenAppend", area=AREA, status="added",
    notes="OpenAppend[\"file\"] opens a file for writing at its end (creating it if needed) and gives "
          "OutputStream[\"full name\", n]; options as OpenWrite (a stream method's constructor gets isAppend = "
          "True). Check: exact; tests append to a file and read it back.")
def _openappend(x: Expr, *rest: Option) -> Expr:
    args, o = _opts(rest, _OPENWRITE_OPTS, "OpenAppend")
    if args:
        raise ValueError("OpenAppend: called with too many arguments")
    return _open_write(x, o, "OpenAppend", True).expr()


def _stream_obj(x, fname):
    """The open stream for InputStream[...]/OutputStream[...]; raises ValueError for a closed stream."""
    if not (isinstance(x, Expr) and x.head in ("InputStream", "OutputStream") and len(x.args) == 2):
        return None
    name, n = x.args
    if not K.is_int(n):
        raise TypeError(f"{fname}: {_form(x)} is not a valid stream")
    n = int(n)
    if x.head == "OutputStream" and n in _STD and name == _STD[n].name:
        return _STD[n]
    if x.head == "InputStream" and is_sym(name, "String"):
        st = _wtext.STRING_STREAMS.get(n)
        if st is None:
            raise ValueError(f"{fname}: {_form(x)} is not open")
        return _StringStream(n, st)
    s = STREAMS.get(n)
    if s is None or s.name != name or (s.kind == "in") != (x.head == "InputStream"):
        raise ValueError(f"{fname}: {_form(x)} is not open")
    return s


def _open_by_name(name, kind):
    out = [s for s in STREAMS.values() if s.kind == kind and s.name == name]
    if kind == "out":
        out += [s for s in _STD.values() if s.name == name]
    return out


def _input(x, fname, o=None, auto_close=False, binary=False):
    """(stream, opened_here) for an input argument: a stream object, or a file name ("name" or File) whose open
    stream is reused, or which is opened (with BinaryFormat when binary)."""
    s = _stream_obj(x, fname)
    if s is not None:
        if s.kind != "in":
            raise ValueError(f"{fname}: {_form(x)} is an output stream")
        return s, False
    name = _name(x, fname)
    if name == "":
        raise ValueError(f"{fname}: the file name must not be empty")
    try:
        full = _resolve_read(name, o or {}, fname)
    except ValueError:
        full = None
    cands = [s for s in STREAMS.values() if s.kind == "in" and (s.name == name or s.name == full)]
    if len(cands) > 1:
        raise ValueError(f"{fname}: more than one input stream is open for {name!r}; give the stream")
    if cands:
        return cands[0], False
    oo = {"BinaryFormat": True} if binary else {}
    if o and "Path" in o:
        oo["Path"] = o["Path"]
    return _open_read(x, oo, fname), True


def _output(x, fname):
    s = _stream_obj(x, fname)
    if s is not None:
        if s.kind != "out":
            raise ValueError(f"{fname}: {_form(x)} is an input stream")
        return s
    name = _name(x, fname)
    full = _abs(_expanduser(name)) if name else name
    cands = _open_by_name(full, "out") or _open_by_name(name, "out")
    if len(cands) > 1:
        raise ValueError(f"{fname}: more than one output stream is open for {name!r}; give the stream")
    if cands:
        return cands[0]
    return _open_write(x, {}, fname, False)


@fn("Close", area=AREA, status="added",
    notes="Close[stream] closes an InputStream or OutputStream (file, string or method stream) and gives its name "
          "(the symbol String for StringToStream streams); Close[\"name\"] closes the only open stream of that "
          "name; closing a closed or unknown stream, or a name with several open streams, raises ValueError; the "
          "standard streams stdout/stderr cannot be closed (ValueError). Check: exact; Streams[] before and after "
          "in tests.")
def _close(x: Expr) -> String | Expr:
    s = _stream_obj(x, "Close")
    if s is None:
        name = _name(x, "Close", "stream or file name")
        full = _abs(_expanduser(name)) if name else name
        cands = [t for t in STREAMS.values() if t.name in (name, full)]
        if not cands:
            cands = [_StringStream(n, st) for n, st in _wtext.STRING_STREAMS.items()] if name == "String" else []
        if len(cands) != 1:
            raise ValueError(f"Close: {'no' if not cands else 'more than one'} open stream named {name!r}")
        s = cands[0]
    if isinstance(s, _StdStream):
        raise ValueError(f"Close: the standard stream {s.name} cannot be closed")
    name = s.name
    s.close()
    return name


@fn("Streams", area=AREA, status="added",
    notes="Streams[] lists the open streams: OutputStream[\"stdout\", 1], OutputStream[\"stderr\", 2], then the open "
          "string, file and method streams in order of their serial numbers; Streams[\"name\"] only those with "
          "that name. Check: exact; tests after OpenRead/OpenWrite/Close.")
def _streams(*args: String) -> ListOf[Expr]:
    if len(args) > 1:
        raise ValueError("Streams: called with too many arguments")
    allst = [(s.n, s.expr()) for s in _STD.values()]
    allst += [(n, Expr("InputStream", Symbol("String"), n)) for n in _wtext.STRING_STREAMS]
    allst += [(s.n, s.expr()) for s in STREAMS.values()]
    allst.sort(key=lambda t: t[0])
    out = [e for _n, e in allst]
    if args:
        nm = args[0]
        if isinstance(nm, Expr) and nm.head == "File":
            nm = _name(nm, "Streams")
        if not isinstance(nm, str):
            if isinstance(nm, Symbol):
                raise Unevaluated()
            raise TypeError(f"Streams: {_form(nm)} is not a stream name")
        full = _abs(_expanduser(nm)) if nm not in ("stdout", "stderr", "String", "") else nm
        out = [e for e in out if e.args[0] in (nm, full) or (nm == "String" and is_sym(e.args[0], "String"))]
    return L(*out)


# ═══════════════════════════════════════════ readers ═══════════════════════════════════════════

class _Reader:
    """Character/byte reading primitives with positions in stream units (bytes; characters for string streams)."""

    def peekc(self):
        p = self.tell()
        c = self.getc()
        self.seek(p)
        return c

    def match(self, seps):
        """The first separator (longest first) the stream continues with, consumed; None if none matches."""
        for s in seps:
            if not s:
                continue
            p = self.tell()
            ok = True
            for ch in s:
                if self.getc() != ch:
                    ok = False
                    break
            if ok:
                return s
            self.seek(p)
        return None

    def at(self, seps):
        p = self.tell()
        m = self.match(seps)
        self.seek(p)
        return m

    def rest_text(self):
        out = []
        while True:
            c = self.getc()
            if c is None:
                return "".join(out)
            out.append(c)


class _ByteReader(_Reader):
    def __init__(self, raw, enc, fname):
        self.raw, self.enc, self.fname = raw, enc or "utf-8", fname

    def tell(self):
        return self.raw.tell()

    def seek(self, p):
        self.raw.seek(p)

    def end(self):
        p = self.raw.tell()
        e = self.raw.seek(0, 2)
        e = self.raw.tell()
        self.raw.seek(p)
        return e

    def getbyte(self):
        b = self.raw.read(1)
        return b[0] if b else None

    def readbytes(self, n):
        return self.raw.read(n) if n >= 0 else self.raw.read()

    def getc(self):
        b = self.raw.read(1)
        if not b:
            return None
        if self.enc == "latin-1":
            return b.decode("latin-1")
        if self.enc == "ascii":
            if b[0] > 127:
                raise ValueError(f"{self.fname}: byte {b[0]} is not ASCII (use CharacterEncoding -> \"UTF8\")")
            return b.decode("ascii")
        lead = b[0]
        n = 1 if lead < 0x80 else 2 if lead >> 5 == 6 else 3 if lead >> 4 == 14 else 4 if lead >> 3 == 30 else 0
        if n == 0:
            raise ValueError(f"{self.fname}: invalid UTF-8 data at byte {self.raw.tell() - 1}")
        if n > 1:
            b += self.raw.read(n - 1)
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{self.fname}: invalid UTF-8 data at byte {self.raw.tell() - len(b)}")


class _CharReader(_Reader):
    """A StringToStream stream: one unit per character."""

    def __init__(self, state, fname):
        self.st, self.fname = state, fname

    def tell(self):
        return self.st["pos"]

    def seek(self, p):
        self.st["pos"] = max(0, min(p, len(self.st["text"])))

    def end(self):
        return len(self.st["text"])

    def getc(self):
        t, p = self.st["text"], self.st["pos"]
        if p >= len(t):
            return None
        self.st["pos"] = p + 1
        return t[p]

    def getbyte(self):
        c = self.getc()
        return None if c is None else ord(c)

    def readbytes(self, n):
        t, p = self.st["text"], self.st["pos"]
        chunk = t[p:] if n < 0 else t[p:p + n]
        if any(ord(c) > 255 for c in chunk):
            raise ValueError(f"{self.fname}: the string stream contains characters beyond the byte range")
        self.st["pos"] = p + len(chunk)
        return chunk.encode("latin-1")


def _reader(s, fname):
    if isinstance(s, _StringStream):
        return _CharReader(s.state, fname)
    return _ByteReader(s.raw, s.enc, fname)


# ── text object types ──

def _strings_opt(v, fname, name):
    if isinstance(v, str):
        return [v]
    if is_list(v) and all(isinstance(x, str) for x in v.args):
        return list(v.args)
    if is_list(v) and any(is_list(x) for x in v.args):
        raise NotImplementedError(f"{fname}: {name} with begin/end delimiter pairs is not supported")
    raise ValueError(f"{fname}: {name} must be a string or a list of strings, not {_form(v)}")


def _read_opts(o, fname, base=None):
    d = dict(base or {})
    d.update(o)
    rs = d.get("RecordSeparators")
    rs = ["\r\n", "\n", "\r"] if rs is None or is_sym(rs, "Automatic") else _strings_opt(rs, fname, "RecordSeparators")
    ws = d.get("WordSeparators")
    ws = [" ", "\t"] if ws is None or is_sym(ws, "Automatic") else _strings_opt(ws, fname, "WordSeparators")
    tw = d.get("TokenWords")
    tw = [] if tw is None else _strings_opt(tw, fname, "TokenWords") if not (is_list(tw) and not tw.args) else []
    return {"rs": sorted(set(rs), key=len, reverse=True), "ws": sorted(set(ws), key=len, reverse=True),
            "tw": sorted(set(tw), key=len, reverse=True),
            "nr": _bool_opt(d, "NullRecords", fname, False), "nw": _bool_opt(d, "NullWords", fname, False)}


_EOFV = object()
_TYPES = ("Byte", "Character", "Expression", "Number", "Real", "Record", "String", "Word")
_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+|\*\^[+-]?\d+)?")
_NUM_CHARS = set("0123456789+-.eEdD*^")


def _skip_ws(r, ro):
    seps = sorted(set(ro["ws"] + ro["rs"] + [" ", "\t", "\n", "\r"]), key=len, reverse=True)
    while r.match(seps):
        pass


def _number_value(tok, real, fname):
    m = re.fullmatch(r"([+-]?)(\d*)\.?(\d*)(?:(?:[eEdD]|\*\^)([+-]?\d+))?", tok)
    sign, ip, fp, ex = m.group(1), m.group(2), m.group(3), m.group(4)
    has_point = "." in tok.split("*^")[0].replace("E", "e").split("e")[0].replace("D", "e").replace("d", "e")
    e = int(ex) if ex else 0
    digits = (ip + fp).lstrip("0")
    if not has_point and not real and (ex is None or e >= 0):
        v = int(ip) * 10 ** e
        return -v if sign == "-" else v
    mant = f"{sign}{ip or '0'}.{fp or '0'}e{e}"
    if len(digits) > 16:
        return sp.Float(mant, len(digits))
    return float(mant)


def _read_number(r, ro, real, fname):
    _skip_ws(r, ro)
    start = r.tell()
    buf = []
    while len(buf) < 4096:
        c = r.getc()
        if c is None or c not in _NUM_CHARS:
            break
        buf.append(c)
    r.seek(start)
    if not buf and r.peekc() is None:
        return _EOFV
    m = _NUM_RE.match("".join(buf))
    if not m or not m.group(0):
        nxt = "".join(buf[:20]) or (r.peekc() or "")
        raise ValueError(f"{fname}: the data {nxt!r} at position {start} is not a number")
    r.seek(start + len(m.group(0)))          # number characters are one unit each in every encoding
    return _number_value(m.group(0), real, fname)


def _read_line(r):
    """Characters up to "\\n" (consumed); a trailing "\\r" is dropped. None at end of input."""
    out = []
    c = r.getc()
    if c is None:
        return None
    while c is not None and c != "\n":
        out.append(c)
        c = r.getc()
    if out and out[-1] == "\r":
        out.pop()
    return "".join(out)


def _read_record(r, ro):
    if not ro["nr"]:
        while r.match(ro["rs"]):
            pass
    if r.peekc() is None:
        return _EOFV
    out = []
    while True:
        if r.match(ro["rs"]):
            break
        c = r.getc()
        if c is None:
            break
        out.append(c)
    return "".join(out)


def _read_word(r, ro):
    seps = sorted(set(ro["ws"] + ro["rs"]), key=len, reverse=True)
    if not ro["nw"]:
        while r.match(seps):
            pass
    if r.peekc() is None:
        return _EOFV
    t = r.match(ro["tw"]) if ro["tw"] else None
    if t is not None:
        return t
    out = []
    while True:
        if r.at(seps) or (ro["tw"] and r.at(ro["tw"])):
            break
        c = r.getc()
        if c is None:
            break
        out.append(c)
    if ro["nw"]:
        r.match(seps)
    return "".join(out)


def _expr_complete(text):
    from ..parser import tokenize, ParseError
    from .wl_evaluation import _CONTINUES
    try:
        toks = tokenize(text)
    except ParseError:
        if '"' in text or "(*" in text:
            return False
        raise
    if not toks:
        return None
    depth = 0
    for t in toks:
        if t.type in ("LPAREN", "LSQUARE", "LBRACE", "LASSOC"):
            depth += 1
        elif t.type in ("RPAREN", "RSQUARE", "RBRACE", "RASSOC"):
            depth -= 1
    return not (depth > 0 or toks[-1].type in _CONTINUES)


def _read_expression(r, fname):
    from ..parser import parse, ParseError
    while True:
        _skip_ws(r, {"ws": [], "rs": []})
        if r.peekc() is None:
            return _EOFV
        start = r.tell()
        buf = ""
        while True:
            line = _read_line(r)
            if line is None:
                break
            buf += line + "\n"
            try:
                done = _expr_complete(buf)
            except ParseError as e:
                r.seek(start)
                raise ValueError(f"{fname}: syntax error in {buf.strip()!r}: {e}")
            if done is None:
                break                       # only comments: skip them
            if done:
                break
        if not buf.strip():
            continue
        try:
            if _expr_complete(buf) is None:
                continue
            e = parse(buf)
        except ParseError as e:
            r.seek(start)
            raise ValueError(f"{fname}: syntax error in {buf.strip()!r}: {e}")
        return e


def _read_type(r, t, ro, fname):
    n = t.name
    if n == "Character":
        c = r.getc()
        return _EOFV if c is None else c
    if n == "Byte":
        b = r.getbyte()
        return _EOFV if b is None else b
    if n == "String":
        s = _read_line(r)
        return _EOFV if s is None else s
    if n == "Record":
        return _read_record(r, ro)
    if n == "Word":
        return _read_word(r, ro)
    if n in ("Number", "Real"):
        return _read_number(r, ro, n == "Real", fname)
    if n == "Expression":
        return _read_expression(r, fname)
    raise ValueError(f"{fname}: {n} is not a type to read")


def _check_types(t, fname):
    if isinstance(t, Symbol):
        if t.name not in _TYPES:
            raise ValueError(f"{fname}: {t.name} is not a type to read (Byte, Character, Expression, Number, Real, "
                             f"Record, String, Word)")
        return
    if isinstance(t, Expr) and not KL.atomq(t) and isinstance(t.head, str):
        if not t.args:
            raise ValueError(f"{fname}: {_form(t)} contains no types to read")
        for a in t.args:
            _check_types(a, fname)
        return
    raise TypeError(f"{fname}: {_form(t)} is not a type or structure of types to read")


class _Partial:
    """A partially read type structure (head plus values that may include _EOFV)."""
    __slots__ = ("head", "vals")

    def __init__(self, head, vals):
        self.head, self.vals = head, vals


def _has_data(v):
    if v is _EOFV:
        return False
    if isinstance(v, _Partial):
        return any(_has_data(a) for a in v.vals)
    return True


def _fill_eof(v):
    if v is _EOFV:
        return EOF
    if isinstance(v, _Partial):
        return Expr(v.head, *(_fill_eof(a) for a in v.vals))
    return v


def _read_struct(r, t, ro, fname, reader=None, top=True):
    """Read the (nested) type structure t depth-first; _EOFV when nothing at all could be read, otherwise
    EndOfFile in every position that was not reached."""
    reader = reader or _read_type
    if isinstance(t, (Symbol, str)):
        return reader(r, t, ro, fname)
    vals = [_read_struct(r, a, ro, fname, reader, False) for a in t.args]
    e = _Partial(t.head, vals)
    if not top:
        return e
    return _fill_eof(e) if _has_data(e) else _EOFV


def _finish(v):
    return EOF if v is _EOFV else ev(v)


def _types_arg(rest, fname, default):
    t = rest[0] if rest else Symbol(default)
    if isinstance(t, str) and fname.startswith("Binary"):
        return t
    _check_types(t, fname)
    return t


@fn("Read", area=AREA, status="added",
    notes="Read[stream] reads one expression (Expression, evaluated); Read[stream, type] with type Byte (character "
          "code for string streams), Character, Expression, Number (an integer unless there is a decimal point; "
          "C/Fortran exponents e, E, d, D and *^; a negative exponent without a point gives a real), Real, Record, "
          "String (up to a newline), Word, or any structure of types such as {Number, Number} or Hold[Expression] "
          "(filled depth-first and then evaluated); EndOfFile at the end (and in the positions not reached); a "
          "file name reuses its open stream or opens one, as documented; options RecordSeparators, WordSeparators, "
          "NullRecords, NullWords, TokenWords (or the stream's); data that is not of the requested type and syntax "
          "errors raise ValueError (Wolfram: $Failed); begin/end separator pairs raise NotImplementedError. "
          "Check: reference (Wolfram documentation examples) and Write/Read round trips in tests.")
def _read(x: Expr, *rest: Expr | Option) -> Expr:
    args, o = _opts(rest, _READ_OPTS + ("Path",), "Read")
    if len(args) > 1:
        raise ValueError("Read: called with too many arguments")
    t = _types_arg(args, "Read", "Expression")
    s, _opened = _input(x, "Read", o)
    ro = _read_opts({k: v for k, v in o.items() if k != "Path"}, "Read", s.ropts)
    return _finish(_read_struct(_reader(s, "Read"), t, ro, "Read"))


def _read_all(r, t, ro, n, fname, record_lists=False):
    out = []
    if record_lists:
        while n is None or len(out) < n:
            rec = _read_record(r, ro)
            if rec is _EOFV:
                break
            sub = _CharReader({"text": rec, "pos": 0}, fname)
            items = []
            while True:
                v = _read_struct(sub, t, ro, fname)
                if v is _EOFV:
                    break
                items.append(v)
            out.append(L(*items))
        return ev(L(*out))
    while n is None or len(out) < n:
        v = _read_struct(r, t, ro, fname)
        if v is _EOFV:
            break
        out.append(v)
    return ev(L(*out))


def _count_arg(x, fname):
    if is_sym(x, "Infinity") or is_sym(x, "All"):
        return None
    n = _int(x, fname, "the number of objects")
    if n < 0:
        raise ValueError(f"{fname}: the number of objects must be non-negative")
    return n


@fn("ReadList", area=AREA, status="added",
    notes="ReadList[src] reads all remaining expressions; ReadList[src, type] all objects of a type or structure "
          "of types (as in Read; partly read structures get EndOfFile in the missing positions); ReadList[src, "
          "type, n] at most n; src is a stream (read from its current position, left open), a file name or "
          "File[...] (opened, read from the start and closed unless it was already open) or a string stream; "
          "RecordLists -> True gives one sublist per record; options as Read; data that does not fit the type "
          "raises ValueError. Check: reference (Wolfram documentation examples) and Write/ReadList round trips in "
          "tests.")
def _readlist(x: Expr, *rest: Expr | Option) -> ListOf[Expr]:
    args, o = _opts(rest, _READ_OPTS + ("RecordLists", "Path"), "ReadList")
    if len(args) > 2:
        raise ValueError("ReadList: called with too many arguments")
    t = _types_arg(args[:1], "ReadList", "Expression")
    n = _count_arg(args[1], "ReadList") if len(args) == 2 else None
    rl = _bool_opt(o, "RecordLists", "ReadList", False)
    s, opened = _input(x, "ReadList", o)
    try:
        ro = _read_opts({k: v for k, v in o.items() if k not in ("RecordLists", "Path")}, "ReadList", s.ropts)
        return _read_all(_reader(s, "ReadList"), t, ro, n, "ReadList", rl)
    finally:
        if opened:
            s.close()


@fn("Skip", area=AREA, status="added",
    notes="Skip[stream, type] skips one object of the type (or structure of types) and Skip[stream, type, n] n of "
          "them, as Read would read them, and returns Null; EndOfFile if the end is reached first (the stream is "
          "then at its end); options as Read; a file name reuses or opens a stream. Check: exact; tests compare "
          "Skip followed by Read with ReadList.")
def _skip(x: Expr, t: Expr, *rest: Integer | Option) -> Expr:
    args, o = _opts(rest, _READ_OPTS, "Skip")
    if len(args) > 1:
        raise ValueError("Skip: called with too many arguments")
    _check_types(t, "Skip")
    n = _count_arg(args[0], "Skip") if args else 1
    s, _opened = _input(x, "Skip", o)
    ro = _read_opts(o, "Skip", s.ropts)
    r = _reader(s, "Skip")
    k = 0
    while n is None or k < n:
        if _read_struct(r, t, ro, "Skip") is _EOFV:
            return EOF if n is not None else NULL
        k += 1
    return NULL


@fn("ReadLine", area=AREA, status="added",
    notes="ReadLine[src] reads the next line (up to \"\\n\" or \"\\r\\n\", not included) from a stream, file name "
          "(opened with BinaryFormat -> True and left open) or File[...]; EndOfFile when the source is exhausted. "
          "Check: reference (Wolfram documentation) and ReadList[src, String] agreement in tests.")
def _readline(x: Expr) -> String | Expr:
    s, _opened = _input(x, "ReadLine", binary=True)
    v = _read_line(_reader(s, "ReadLine"))
    return EOF if v is None else v


def _find_term(r, term, fname):
    """ReadString up to a terminator: the text before it (terminator consumed), or the rest of the input."""
    start = r.tell()
    text = r.rest_text()
    if isinstance(term, str):
        if term == "":
            raise ValueError(f"{fname}: the terminator must not be empty")
        k = text.find(term)
        span = None if k < 0 else (k, k + len(term))
    else:
        pos = ev(Expr("StringPosition", text, term, 1))
        if not is_list(pos):
            raise ValueError(f"{fname}: {_form(term)} is not a valid string pattern")
        span = None
        if pos.args:
            a, b = pos.args[0].args
            span = (int(a) - 1, int(b))
            if span[1] <= span[0]:
                raise ValueError(f"{fname}: the terminator {_form(term)} matches the empty string")
    if span is None:
        return text
    r.seek(start)
    for _ in range(span[1]):
        r.getc()
    return text[:span[0]]


@fn("ReadString", area=AREA, status="added",
    notes="ReadString[src] reads everything remaining in a stream, file name or File[...] as a string; "
          "ReadString[src, term] reads up to the terminator (a string or string pattern; the terminator is "
          "consumed and dropped, the rest of the input when absent), EndOfFile (default) or EndOfBuffer (for files "
          "and strings all content is available, so the rest of the input); EndOfFile when the source is "
          "exhausted; a file name is opened with BinaryFormat -> True and closed after reading to EndOfFile, else "
          "left open. TimeConstraint raises NotImplementedError. Check: reference and WriteString/ReadString round "
          "trips in tests.")
def _readstring(x: Expr, *rest: Expr | Option) -> String | Expr:
    args, o = _opts(rest, (), "ReadString", unsupported=("TimeConstraint",))
    if len(args) > 1:
        raise ValueError("ReadString: called with too many arguments")
    term = args[0] if args else EOF
    whole = is_sym(term, "EndOfFile") or is_sym(term, "EndOfBuffer")
    if not whole and not isinstance(term, str) and not (isinstance(term, Expr) and not is_list(term)):
        raise TypeError(f"ReadString: {_form(term)} is not a terminator")
    s, opened = _input(x, "ReadString", binary=True)
    try:
        r = _reader(s, "ReadString")
        if is_sym(term, "EndOfBuffer") and hasattr(getattr(s, "raw", None), "available"):
            return s.raw.available(r.rest_text, EOF)     # pipes (W29 processes): only the data available now
        if r.peekc() is None:
            return EOF
        return r.rest_text() if whole else _find_term(r, term, "ReadString")
    finally:
        if opened and is_sym(term, "EndOfFile"):
            s.close()


def _byte_terms(term, fname):
    if is_sym(term, "EndOfLine"):
        return [b"\r\n", b"\n"]
    if _is_byte_array(term):
        b = _ba_bytes(term, fname)
        if not b:
            raise ValueError(f"{fname}: the terminator must not be empty")
        return [b]
    if is_list(term) and term.args:
        out = []
        for t in term.args:
            out.extend(_byte_terms(t, fname))
        return sorted(out, key=len, reverse=True)
    if isinstance(term, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(term)} is not a terminator (n, ByteArray[...], EndOfLine, EndOfBuffer, "
                    f"EndOfFile or a list of these)")


@fn("ReadByteArray", area=AREA, status="added",
    notes="ReadByteArray[src] reads the remaining bytes as a ByteArray; ReadByteArray[src, n] up to n bytes; "
          "ReadByteArray[src, term] up to a terminator ByteArray[...], EndOfLine (\"\\r\\n\" or \"\\n\"), "
          "EndOfBuffer (the rest, all content being available) or a list of these (the terminator is dropped); "
          "EndOfFile when the source is exhausted; a file name is opened and closed after reading to EndOfFile; "
          "string streams give their character codes (codes above 255 raise ValueError). Check: reference "
          "(explicit bytes) and BinaryWrite/ReadByteArray round trips in tests.")
def _readbytearray(x: Expr, *rest: Expr | Option) -> Expr:
    args, o = _opts(rest, (), "ReadByteArray", unsupported=("TimeConstraint",))
    if len(args) > 1:
        raise ValueError("ReadByteArray: called with too many arguments")
    term = args[0] if args else EOF
    whole = is_sym(term, "EndOfFile") or is_sym(term, "EndOfBuffer")
    count = None
    terms = None
    if not whole:
        if K.is_int(term) and not isinstance(term, bool):
            count = int(term)
            if count < 0:
                raise ValueError("ReadByteArray: the number of bytes must be non-negative")
        else:
            terms = _byte_terms(term, "ReadByteArray")
    s, opened = _input(x, "ReadByteArray", binary=True)
    try:
        r = _reader(s, "ReadByteArray")
        p = r.tell()
        if r.getbyte() is None:
            return EOF if count != 0 else _byte_array(b"")
        r.seek(p)
        if whole:
            return _byte_array(r.readbytes(-1))
        if count is not None:
            return _byte_array(r.readbytes(count))
        out = bytearray()
        while True:
            hit = None
            for t in terms:
                q = r.tell()
                if r.readbytes(len(t)) == t:
                    hit = t
                    break
                r.seek(q)
            if hit is not None:
                break
            b = r.getbyte()
            if b is None:
                break
            out.append(b)
        return _byte_array(bytes(out))
    finally:
        if opened and is_sym(term, "EndOfFile"):
            s.close()


@fn("StreamPosition", area=AREA, status="added",
    notes="StreamPosition[stream] gives the current position: bytes from the start for file and method streams "
          "(bytes written for output streams), characters for StringToStream streams; stdout/stderr raise "
          "NotImplementedError. Check: exact; SetStreamPosition round trips in tests.")
def _streamposition(x: Expr) -> NonNegInteger:
    s = _stream_obj(x, "StreamPosition")
    if s is None:
        s, _o = _input(x, "StreamPosition")
    if isinstance(s, _StringStream):
        return s.state["pos"]
    return int(s.raw.tell())


@fn("SetStreamPosition", area=AREA, status="added",
    notes="SetStreamPosition[stream, n] moves the current point to position n (as given by StreamPosition) and "
          "SetStreamPosition[stream, Infinity] to the end; returns the new position; a position beyond the end of "
          "an input stream raises ValueError; stdout/stderr and output method streams raise NotImplementedError. "
          "Check: exact; reading again after repositioning gives the same objects in tests.")
def _setstreamposition(x: Expr, n: NonNegInteger | Expr) -> NonNegInteger:
    s = _stream_obj(x, "SetStreamPosition")
    if s is None:
        s, _o = _input(x, "SetStreamPosition")
    if isinstance(s, _StdStream):
        raise NotImplementedError(f"SetStreamPosition: {s.name} cannot be repositioned")
    if s.kind == "out" and isinstance(s.raw, _MethodOut):
        raise NotImplementedError("SetStreamPosition: streams of a user-defined output method cannot be "
                                  "repositioned")
    r = _CharReader(s.state, "SetStreamPosition") if isinstance(s, _StringStream) else None
    if is_sym(n, "Infinity") or (isinstance(n, Expr) and n.head == "DirectedInfinity"):
        if r is not None:
            r.seek(r.end())
            return r.tell()
        s.raw.seek(0, 2)
        return int(s.raw.tell())
    k = _int(n, "SetStreamPosition", "the position")
    if k < 0:
        raise ValueError("SetStreamPosition: the position must be non-negative")
    if r is not None:
        if k > r.end():
            raise ValueError(f"SetStreamPosition: position {k} is beyond the end of the stream ({r.end()})")
        r.seek(k)
        return k
    if s.kind == "in":
        end = _ByteReader(s.raw, s.enc, "SetStreamPosition").end()
        if k > end:
            raise ValueError(f"SetStreamPosition: position {k} is beyond the end of the stream ({end})")
    s.raw.seek(k)
    return int(s.raw.tell())


# ═══════════════════════════════════════════ writing ═══════════════════════════════════════════

def _encode_text(text, s, fname):
    enc = s.enc or "utf-8"
    if enc == "ascii":
        return "".join(c if ord(c) < 128 else ("\\:%04x" % ord(c) if ord(c) <= 0xFFFF else "\\|%06x" % ord(c))
                       for c in text).encode("ascii")
    try:
        return text.encode(enc)
    except UnicodeEncodeError as e:
        raise ValueError(f"{fname}: the text cannot be encoded as {enc} ({e.reason})")


def _channels(ch, fname):
    items = list(ch.args) if is_list(ch) else [ch]
    if not items:
        raise ValueError(f"{fname}: no output channel given")
    return [_output(c, fname) for c in items]


def _put(streams, text, fname):
    for s in streams:
        s.raw.write(_encode_text(text, s, fname))
        if hasattr(s.raw, "flush"):
            s.raw.flush()


def _write_form(x, s):
    if s.fmt == "OutputForm":
        return x if isinstance(x, str) else ev(Expr("ToString", x))
    return _input_text(x)


@fn("Write", area=AREA, status="added",
    notes="Write[channel, e1, e2, ...] writes the expressions one after another in the stream's FormatType "
          "(InputForm for files: strings in quotes; OutputForm for stdout/stderr), then a newline, and returns "
          "Null; channel is an OutputStream, a file name or File[...] (opened with OpenWrite if not already open, "
          "and left open, as documented) or a list of channels; lines are never wrapped. Check: exact; "
          "Write/Read and Write/Get round trips in tests.")
def _write(ch: Expr, *exprs: Expr) -> Expr:
    for s in _channels(ch, "Write"):
        _put([s], "".join(_write_form(e, s) for e in exprs) + "\n", "Write")
    return NULL


def _string_arg(x):
    if x is None:
        return "Null"
    return x if isinstance(x, str) else ev(Expr("ToString", x))


@fn("WriteString", area=AREA, status="added",
    notes="WriteString[channel, e1, e2, ...] writes strings as they are (other expressions in OutputForm, as "
          "ToString) with no newline and returns Null; a file name not already open is opened with OpenWrite "
          "(erasing it) and left open; lists of channels accepted. Check: exact; ReadString round trips and "
          "FileByteCount in tests.")
def _writestring(ch: Expr, *exprs: Expr) -> Expr:
    text = "".join(_string_arg(e) for e in exprs)
    _put(_channels(ch, "WriteString"), text, "WriteString")
    return NULL


@fn("WriteLine", area=AREA, status="added",
    notes="WriteLine[channel, \"string\"] writes the string followed by a newline and returns Null; a file name not "
          "already open is opened with OpenWrite and left open; a non-string raises TypeError. Check: exact; "
          "ReadLine round trips in tests.")
def _writeline(ch: Expr, text: String) -> Expr:
    if not isinstance(text, str):
        if isinstance(text, Symbol):
            raise Unevaluated()
        raise TypeError(f"WriteLine: {_form(text)} is not a string")
    _put(_channels(ch, "WriteLine"), text + "\n", "WriteLine")
    return NULL


# ═══════════════════════════════════════════ binary I/O ═══════════════════════════════════════════

_BIN_INTS = {"Byte": (1, False), "UnsignedInteger8": (1, False), "Integer8": (1, True),
             "Integer16": (2, True), "Integer24": (3, True), "Integer32": (4, True), "Integer64": (8, True),
             "Integer128": (16, True), "UnsignedInteger16": (2, False), "UnsignedInteger24": (3, False),
             "UnsignedInteger32": (4, False), "UnsignedInteger64": (8, False), "UnsignedInteger128": (16, False)}
_BIN_REALS = {"Real32": 4, "Real64": 8, "Real128": 16}
_BIN_COMPLEX = {"Complex64": 4, "Complex128": 8, "Complex256": 16}
_BIN_CHARS = {"Character8": 1, "Character16": 2, "Character32": 4}
_BIN_TYPES = set(_BIN_INTS) | set(_BIN_REALS) | set(_BIN_COMPLEX) | set(_BIN_CHARS) | {"TerminatedString"}


def _byte_order(o, fname):
    v = o.get("ByteOrdering")
    if v is None or is_sym(v, "Automatic"):
        v = ev(Symbol("$ByteOrdering"))
    if K.is_int(v) and int(v) in (1, -1) and not isinstance(v, bool):
        return "big" if int(v) == 1 else "little"
    raise ValueError(f"{fname}: ByteOrdering must be 1 or -1, not {_form(v)}")


def _neg_inf():
    return ev(Expr("Times", -1, Symbol("Infinity")))


def _float_value(v):
    if v != v:
        return Symbol("Indeterminate")
    if v == float("inf"):
        return Symbol("Infinity")
    if v == float("-inf"):
        return _neg_inf()
    return float(v)


_Q_BIAS, _Q_MANT = 16383, 112


def _quad_decode(b):
    bits = int.from_bytes(b, "big")
    sign = bits >> 127
    ex = (bits >> _Q_MANT) & 0x7FFF
    frac = bits & ((1 << _Q_MANT) - 1)
    if ex == 0x7FFF:
        if frac:
            return Symbol("Indeterminate")
        return _neg_inf() if sign else Symbol("Infinity")
    if ex == 0:
        q = Fraction(frac, 1) * Fraction(2) ** (1 - _Q_BIAS - _Q_MANT)
    else:
        q = Fraction(frac | (1 << _Q_MANT), 1) * Fraction(2) ** (ex - _Q_BIAS - _Q_MANT)
    if sign:
        q = -q
    return sp.Float(sp.Rational(q.numerator, q.denominator), 34)


def _exact_value(x, fname, tname):
    """An exact Fraction (or "inf", "-inf", "nan") for a real number to be written in binary."""
    if isinstance(x, bool) or isinstance(x, str):
        raise TypeError(f"{fname}: {_form(x)} is not a real number ({tname})")
    if is_sym(x, "Infinity"):
        return "inf"
    if is_sym(x, "Indeterminate"):
        return "nan"
    if isinstance(x, Expr) and K.same(x, _neg_inf()):
        return "-inf"
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        if x != x:
            return "nan"
        if x in (float("inf"), float("-inf")):
            return "inf" if x > 0 else "-inf"
        return Fraction(x)
    from ..math_ops import to_sympy
    try:
        v = x if isinstance(x, sp.Basic) else to_sympy(x)
    except Exception:  # noqa: BLE001 - not convertible: decided below
        v = None
    if not isinstance(v, sp.Basic):
        if isinstance(x, Symbol) or isinstance(x, Expr):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_form(x)} is not a real number ({tname})")
    if v.free_symbols:
        raise Unevaluated()
    if isinstance(v, sp.Integer):
        return Fraction(int(v))
    if isinstance(v, sp.Rational):
        return Fraction(int(v.p), int(v.q))
    if isinstance(v, sp.Float):
        sign, man, e = v._mpf_[0], v._mpf_[1], v._mpf_[2]
        q = Fraction(int(man)) * (Fraction(2) ** int(e))
        return -q if sign else q
    if v.is_number and v.is_real:
        f = sp.Float(v, 60)                     # exact constants (Pi, Sqrt[2]): 60 digits, far beyond 113 bits
        sign, man, e = f._mpf_[0], f._mpf_[1], f._mpf_[2]
        q = Fraction(int(man)) * (Fraction(2) ** int(e))
        return -q if sign else q
    raise TypeError(f"{fname}: {_form(x)} is not a real number ({tname})")


_IEEE = {4: (8, 23), 8: (11, 52), 16: (15, 112)}


def _ieee_encode(q, nbytes, fname, tname):
    """Correctly rounded (round half to even) IEEE 754 binary encoding, big-endian; overflow raises."""
    eb, mb = _IEEE[nbytes]
    bias, emax = (1 << (eb - 1)) - 1, (1 << eb) - 1
    width = 8 * nbytes
    if q == "nan":
        return ((emax << mb) | (1 << (mb - 1))).to_bytes(nbytes, "big")
    if q in ("inf", "-inf"):
        return (((1 if q == "-inf" else 0) << (width - 1)) | (emax << mb)).to_bytes(nbytes, "big")
    sign = 1 if q < 0 else 0
    a = -q if sign else q
    if a == 0:
        return (sign << (width - 1)).to_bytes(nbytes, "big")
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    if e < 1 - bias:                            # subnormal range
        mi = _round_half_even(a / Fraction(2) ** (1 - bias - mb))
        if mi >> mb:                            # rounded up to the smallest normal number
            return ((sign << (width - 1)) | (1 << mb) | (mi - (1 << mb))).to_bytes(nbytes, "big")
        return ((sign << (width - 1)) | mi).to_bytes(nbytes, "big")
    mi = _round_half_even(a / Fraction(2) ** (e - mb))
    if mi >> (mb + 1):
        mi >>= 1
        e += 1
    if e + bias >= emax:
        raise ValueError(f"{fname}: {float(q):g} is out of range for {tname}")
    return ((sign << (width - 1)) | ((e + bias) << mb) | (mi - (1 << mb))).to_bytes(nbytes, "big")


def _round_half_even(m):
    f = m.numerator // m.denominator
    r = m - f
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and f % 2):
        f += 1
    return f


def _bin_read_one(r, t, bo, fname):
    if t in _BIN_INTS:
        n, signed = _BIN_INTS[t]
        b = r.readbytes(n)
        if len(b) < n:
            return _EOFV
        return int.from_bytes(b, bo, signed=signed)
    if t in _BIN_REALS:
        n = _BIN_REALS[t]
        b = r.readbytes(n)
        if len(b) < n:
            return _EOFV
        if n == 16:
            return _quad_decode(b if bo == "big" else b[::-1])
        return _float_value(struct.unpack((">" if bo == "big" else "<") + ("f" if n == 4 else "d"), b)[0])
    if t in _BIN_COMPLEX:
        n = _BIN_COMPLEX[t]
        b = r.readbytes(2 * n)
        if len(b) < 2 * n:
            return _EOFV
        parts = []
        for h in (b[:n], b[n:]):
            if n == 16:
                parts.append(_quad_decode(h if bo == "big" else h[::-1]))
            else:
                parts.append(_float_value(struct.unpack((">" if bo == "big" else "<") + ("f" if n == 4 else "d"),
                                                        h)[0]))
        if all(isinstance(v, float) for v in parts):
            return R_(complex(parts[0], parts[1])) if parts[1] != 0 else Expr("Complex", parts[0], 0.0)
        return ev(Expr("Plus", parts[0], Expr("Times", parts[1], Symbol("I"))))
    if t in _BIN_CHARS:
        n = _BIN_CHARS[t]
        b = r.readbytes(n)
        if len(b) < n:
            return _EOFV
        code = int.from_bytes(b, bo)
        try:
            return chr(code)
        except (ValueError, OverflowError):
            raise ValueError(f"{fname}: {code} is not a valid character code ({t})")
    if t == "TerminatedString":
        out = bytearray()
        first = r.getbyte()
        if first is None:
            return _EOFV
        b = first
        while b is not None and b != 0:
            out.append(b)
            b = r.getbyte()
        return out.decode("latin-1")
    raise ValueError(f"{fname}: {t!r} is not a binary type")


def _check_bin_types(t, fname):
    if isinstance(t, str):
        if t not in _BIN_TYPES:
            raise ValueError(f"{fname}: {t!r} is not a binary type (\"Byte\", \"Integer32\", \"Real64\", ...)")
        return
    if isinstance(t, Expr) and not KL.atomq(t) and isinstance(t.head, str) and t.args:
        for a in t.args:
            _check_bin_types(a, fname)
        return
    if isinstance(t, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(t)} is not a binary type or structure of types")


def _bin_one(r, t, bo, fname):
    p = r.tell()
    v = _bin_read_one(r, t, bo, fname)
    if v is _EOFV:
        r.seek(p)
    return v


def _bin_struct(r, t, bo, fname):
    return _read_struct(r, t, bo, fname, reader=lambda rr, tt, oo, ff: _bin_one(rr, tt, oo, ff))


@fn("BinaryRead", area=AREA, status="added",
    notes="BinaryRead[stream] reads one byte; BinaryRead[stream, \"type\"] one object of a binary type: \"Byte\", "
          "\"(Unsigned)Integer8|16|24|32|64|128\", \"Real32\", \"Real64\", \"Real128\" (IEEE quad, decoded exactly "
          "and given with 34 digits), \"Complex64|128|256\", \"Character8|16|32\", \"TerminatedString\" "
          "(null-terminated 8-bit characters); or a structure of types such as {\"Integer16\", \"Real64\"}; IEEE "
          "infinities give Infinity/-Infinity and NaN Indeterminate; EndOfFile at the end (an incomplete object is "
          "not consumed); ByteOrdering -> 1 | -1 (default $ByteOrdering); a file name reuses its open stream or "
          "opens one. Check: reference (IEEE 754 and IntegerDigits[n, 256] byte layouts) and BinaryWrite round "
          "trips in tests.")
def _binaryread(x: Expr, *rest: String | ListOf[String] | Option) -> Expr:
    args, o = _opts(rest, ("ByteOrdering", "Path"), "BinaryRead")
    if len(args) > 1:
        raise ValueError("BinaryRead: called with too many arguments")
    t = args[0] if args else "Byte"
    _check_bin_types(t, "BinaryRead")
    bo = _byte_order(o, "BinaryRead")
    s, _opened = _input(x, "BinaryRead", o, binary=True)
    return _finish(_bin_struct(_reader(s, "BinaryRead"), t, bo, "BinaryRead"))


@fn("BinaryReadList", area=AREA, status="added",
    notes="BinaryReadList[src] reads all remaining bytes; BinaryReadList[src, type] all objects of a binary type or "
          "structure of types (types as BinaryRead; a partly read structure gets EndOfFile in the missing "
          "positions); BinaryReadList[src, type, n] at most n; a file name is opened (BinaryFormat -> True), read "
          "from the start and closed unless it was already open; ByteOrdering option. Check: reference (explicit "
          "byte layouts) and BinaryWrite round trips in tests.")
def _binaryreadlist(x: Expr, *rest: Expr | Option) -> ListOf[Expr]:
    args, o = _opts(rest, ("ByteOrdering", "Path"), "BinaryReadList")
    if len(args) > 2:
        raise ValueError("BinaryReadList: called with too many arguments")
    t = args[0] if args else "Byte"
    _check_bin_types(t, "BinaryReadList")
    n = _count_arg(args[1], "BinaryReadList") if len(args) == 2 else None
    bo = _byte_order(o, "BinaryReadList")
    s, opened = _input(x, "BinaryReadList", o, binary=True)
    try:
        r = _reader(s, "BinaryReadList")
        out = []
        while n is None or len(out) < n:
            v = _bin_struct(r, t, bo, "BinaryReadList")
            if v is _EOFV:
                break
            out.append(v)
        return ev(L(*out))
    finally:
        if opened:
            s.close()


def _bin_encode(x, t, bo, fname):
    if t in _BIN_INTS:
        n, signed = _BIN_INTS[t]
        if isinstance(x, bool) or not K.is_int(x):
            if isinstance(x, Symbol) or (isinstance(x, Expr) and not K.is_number(x)):
                raise Unevaluated()
            raise TypeError(f"{fname}: {_form(x)} is not an integer ({t})")
        v = int(x)
        lo, hi = (-(1 << (8 * n - 1)), (1 << (8 * n - 1)) - 1) if signed else (0, (1 << (8 * n)) - 1)
        if not lo <= v <= hi:
            raise ValueError(f"{fname}: {v} is out of range for {t}")
        return v.to_bytes(n, bo, signed=signed)
    if t in _BIN_REALS:
        return _real_bytes(_exact_value(x, fname, t), _BIN_REALS[t], bo, fname, t)
    if t in _BIN_COMPLEX:
        n = _BIN_COMPLEX[t]
        if isinstance(x, Expr) and x.head == "Complex" and len(x.args) == 2:
            re_, im_ = x.args
        else:
            try:
                from ..math_ops import to_sympy
                v = to_sympy(x)
                if v.free_symbols:
                    raise Unevaluated()
                re_, im_ = R_(sp.re(v)), R_(sp.im(v))
            except Unevaluated:
                raise
            except Exception:  # noqa: BLE001
                raise TypeError(f"{fname}: {_form(x)} is not a number ({t})")
        return (_real_bytes(_exact_value(re_, fname, t), n, bo, fname, t)
                + _real_bytes(_exact_value(im_, fname, t), n, bo, fname, t))
    if t in _BIN_CHARS or t == "TerminatedString":
        if not isinstance(x, str):
            if isinstance(x, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: {_form(x)} is not a string ({t})")
        n = 1 if t == "TerminatedString" else _BIN_CHARS[t]
        out = bytearray()
        for c in x:
            if ord(c) >= 1 << (8 * n):
                raise ValueError(f"{fname}: the character {c!r} does not fit in {t}")
            if t == "TerminatedString" and ord(c) == 0:
                raise ValueError(f"{fname}: a TerminatedString cannot contain a null character")
            out += ord(c).to_bytes(n, bo)
        if t == "TerminatedString":
            out += b"\x00"
        return bytes(out)
    raise ValueError(f"{fname}: {t!r} is not a binary type")


def R_(v):
    from .core import R
    return R(v)


def _real_bytes(q, n, bo, fname, t):
    b = _ieee_encode(q, n, fname, t)
    return b if bo == "big" else b[::-1]


def _flatten(x):
    if is_list(x):
        out = []
        for a in x.args:
            out.extend(_flatten(a))
        return out
    return [x]


@fn("BinaryWrite", area=AREA, status="added",
    notes="BinaryWrite[channel, b] writes a byte or list of bytes (0-255), a ByteArray, or a string as "
          "\"Character8\"; BinaryWrite[channel, x, \"type\"] or a list of values with a type or a list of types "
          "(repeated cyclically); types as BinaryRead (exact numbers are rounded correctly to the IEEE format, "
          "Real128 included; Infinity, -Infinity and Indeterminate give IEEE infinities and NaN); ByteOrdering -> "
          "1 | -1; channel is an OutputStream, file name (opened with OpenWrite if not open, and left open) or "
          "File[...] or a list of them; returns the channel; a value that does not fit the type raises ValueError "
          "(the wrong kind: TypeError) and nothing is written. Check: reference (IEEE 754 encodings, "
          "IntegerDigits[n, 256]) and BinaryRead round trips in tests.")
def _binarywrite(ch: Expr, data: Expr, *rest: String | ListOf[String] | Option) -> Expr:
    args, o = _opts(rest, ("ByteOrdering",), "BinaryWrite")
    if len(args) > 1:
        raise ValueError("BinaryWrite: called with too many arguments")
    bo = _byte_order(o, "BinaryWrite")
    if args:
        t = args[0]
        types = list(t.args) if is_list(t) else [t]
        if not types or not all(isinstance(a, str) for a in types):
            raise TypeError(f"BinaryWrite: {_form(t)} is not a binary type or list of types")
        for a in types:
            _check_bin_types(a, "BinaryWrite")
        vals = _flatten(data)
        if isinstance(data, str) and all(a in _BIN_CHARS or a == "TerminatedString" for a in types):
            vals = [data] if types[0] == "TerminatedString" else list(data)
        out = b"".join(_bin_encode(v, types[i % len(types)], bo, "BinaryWrite") for i, v in enumerate(vals))
    elif _is_byte_array(data):
        out = _ba_bytes(data, "BinaryWrite")
    elif isinstance(data, str):
        out = b"".join(_bin_encode(c, "Character8", bo, "BinaryWrite") for c in data)
    else:
        out = b"".join(_bin_encode(v, "Byte", bo, "BinaryWrite") for v in _flatten(data))
    for s in _channels(ch, "BinaryWrite"):
        s.raw.write(out)
        if hasattr(s.raw, "flush"):
            s.raw.flush()
    return ch


# ═══════════════════════════════════════════ inert symbols and options ═══════════════════════════════════════════

_INERT = {
    "File": "File[\"path\"] is an inert wrapper for a local file name, accepted by every file function here (and by "
            "Import, Export, Get, ...); the path need not exist",
    "InputStream": "InputStream[name, n] is the inert object for an open input stream (OpenRead gives "
                   "InputStream[\"full name\", n], StringToStream InputStream[String, n]); Read, ReadList, ... act on it",
    "OutputStream": "OutputStream[name, n] is the inert object for an open output stream (OpenWrite, OpenAppend; "
                    "OutputStream[\"stdout\", 1], OutputStream[\"stderr\", 2])",
    "EndOfFile": "EndOfFile is returned by Read, ReadList, ReadLine, ReadString, ReadByteArray, BinaryRead, Find, ... "
                 "at the end of a stream, and is the default terminator of ReadString and ReadByteArray",
    "EndOfBuffer": "EndOfBuffer as the terminator of ReadString/ReadByteArray reads what is available without "
                   "waiting; for files and string streams that is the rest of the input",
    "Byte": "Byte is the Read/ReadList/Skip type for one byte, returned as an integer (a character code for string "
            "streams); BinaryRead uses \"Byte\"",
    "Character": "Character is the Read/ReadList/Skip type for one character, returned as a one-character string",
    "Expression": "Expression is the Read/ReadList type for one complete Wolfram Language expression (evaluated; "
                  "Hold[Expression] keeps it unevaluated) and the default type of Read and ReadList",
    "Number": "Number is the Read/ReadList type for an integer or approximate real number (C/Fortran exponent forms "
              "allowed; an integer when there is no decimal point)",
    "Record": "Record is the Read/ReadList type for the text between record separators (RecordSeparators)",
    "Word": "Word is the Read/ReadList type for the text between word or record separators (WordSeparators, "
            "RecordSeparators, TokenWords)",
    "BinaryFormat": "BinaryFormat -> True opens a stream for binary data (OpenRead, OpenWrite, OpenAppend); no "
                    "newline translation is done in either mode on any system",
    "ByteOrdering": "ByteOrdering -> 1 (big-endian) | -1 (little-endian) is the option of BinaryRead, BinaryReadList "
                    "and BinaryWrite (default $ByteOrdering) and of binary Import/Export formats",
    "CreateIntermediateDirectories": "CreateIntermediateDirectories -> True | False is the option of CreateFile and "
                                     "CreateDirectory (default True: missing parent directories are created)",
    "OverwriteTarget": "OverwriteTarget -> True | False is the option of CopyFile, RenameFile and CreateFile (default "
                       "False: an existing target raises)",
    "RecordLists": "RecordLists -> True makes ReadList return one sublist per record",
    "FileNameForms": "FileNameForms -> forms is the option of FileSystemMap and FileSystemScan selecting file names "
                     "by the same forms as FileNames (default All)",
    "Path": "Path -> {dir, ...} is the option of FindFile, OpenRead, Read, ReadList, BinaryRead and BinaryReadList "
            "giving the directories to search instead of $Path",
}
for _n, _d in _INERT.items():
    record(_n, AREA, "added", _d + ". Check: reference (Wolfram documentation) in tests.")
