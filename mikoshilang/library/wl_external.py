"""Wolfram batch W30: external interfaces (URLs and HTTP, cookies, network hosts, sockets and mail).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). Compilation and typed functions live in ``wl_external_compile``; foreign functions and raw
memory (ctypes) in ``wl_external_ffi``.

Network access goes through the Python standard library only (``http.client``, ``socket``, ``imaplib``). Nothing
here runs at import time, no credentials are read from disk, and ``$AllowInternet = False`` blocks every
non-loopback connection before it is attempted. Connection failures of the URL functions give the documented
``Failure["ConnectionFailure", <|...|>]`` result (Wolfram's URLRead/URLExecute/URLDownload do the same); the other
network heads raise.
"""
from __future__ import annotations

import base64
import datetime as _dt
import http.client
import http.cookies
import ipaddress
import os
import socket
import tempfile
import time
import urllib.parse
import urllib.request
import uuid

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
import importlib
from .core import Unevaluated, fn, record, L, is_list
from .wl_common import ev, is_sym, apply
from .wl_importexport import byte_array, is_byte_array, ba_bytes
from .wl_text import _codec

E = importlib.import_module("mikoshilang.evaluate")

AREA = "external interfaces (W30)"
NONE = Symbol("None")
_EPOCH1900 = 2208988800


# ═══════════════════════════════════════ small helpers ═══════════════════════════════════════

def _sym(n):
    return Symbol(n)


def _assoc(pairs):
    return KL.make_assoc([Expr("Rule", k, v) for k, v in pairs])


def _is_assoc(x):
    return isinstance(x, Expr) and x.head == "Association"


def _assoc_items(a):
    return [(r.args[0], KL.rule_value(r, ev)) for r in a.args]


def _key(k):
    """Association/option key as a Python string (symbol keys such as Method by name)."""
    if isinstance(k, str):
        return k
    if isinstance(k, Symbol):
        return k.name
    return None


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def _rules(x, fname, what="the argument"):
    """(key, value) pairs of an association or a list of rules."""
    if _is_assoc(x):
        return _assoc_items(x)
    if is_list(x) and all(_is_rule(r) for r in x.args):
        return [(r.args[0], KL.rule_value(r, ev)) for r in x.args]
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be an association or a list of rules, not {_show(x)}")


def _rules_deep(x, fname):
    """Pairs of an association or a (nested) list of rules and associations."""
    if _is_assoc(x):
        return _assoc_items(x)
    if is_list(x):
        out = []
        for e in x.args:
            if _is_rule(e):
                out.append((e.args[0], KL.rule_value(e, ev)))
            elif is_list(e) or _is_assoc(e):
                out.extend(_rules_deep(e, fname))
            else:
                raise TypeError(f"{fname}: {_show(e)} is not a rule")
        return out
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: the argument must be an association or a list of rules, not {_show(x)}")


def _show(x):
    try:
        from .. import formatting as F
        return F.input_form(x)
    except Exception:
        return repr(x)


def split_opts(args, names, fname):
    """Trailing options name -> value (name a symbol or a string in names); an unknown option raises ValueError."""
    args = list(args)
    opts = {}
    while args and _is_rule(args[-1]) and _key(args[-1].args[0]) is not None:
        k = _key(args[-1].args[0])
        if k not in names:
            raise ValueError(f"{fname}: {k} is not a known option (expected one of {', '.join(sorted(names))})")
        opts[k] = args.pop().args[1]
    return args, opts


def _str(x, fname, what="the argument"):
    if isinstance(x, str):
        return x
    if isinstance(x, Symbol) or (isinstance(x, Expr) and x.head not in ("List", "Association", "Rule")
                                 and not is_byte_array(x) and x.head not in ("URL", "File", "IPAddress")):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be a string, not {_show(x)}")


def _url_str(x, fname):
    """A URL given as a string or URL["..."]; CloudObject is not supported."""
    if isinstance(x, str):
        return x
    if isinstance(x, Expr) and x.head == "URL" and len(x.args) == 1 and isinstance(x.args[0], str):
        return x.args[0]
    if isinstance(x, Expr) and x.head == "CloudObject":
        raise NotImplementedError(f"{fname}: CloudObject URLs are not supported (there is no Wolfram Cloud)")
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a URL string or URL[\"...\"], not {_show(x)}")


def _bool_opt(v, fname, name):
    if v is True or v is False:
        return v
    raise ValueError(f"{fname}: {name} must be True or False, not {_show(v)}")


def _enc_opt(opts, fname, default="utf-8"):
    v = opts.get("CharacterEncoding")
    if v is None or is_sym(v, "Automatic"):
        return default
    return _codec(v, fname)


def _wl_number_string(x):
    """InputForm-style text of a machine number (1.3 -> "1.3", 1. -> "1.")."""
    if isinstance(x, float):
        r = repr(x)
        if "e" in r or "inf" in r or "nan" in r:
            raise ValueError(f"cannot encode {r}")
        return r[:-1] if r.endswith(".0") else r
    return str(x)


def _query_value(v, fname):
    """Query/encode value forms: strings, numbers, None/Null/Missing -> "", True/False, URL[...]; lists flatten."""
    if v is None or is_sym(v, "None") or is_sym(v, "Null") or is_sym(v, "Missing") or \
            (isinstance(v, Expr) and v.head == "Missing"):
        return [""]
    if v is True:
        return ["true"]
    if v is False:
        return ["false"]
    if isinstance(v, str):
        return [v]
    if isinstance(v, bool):
        return ["true" if v else "false"]
    if isinstance(v, int) or K.is_int(v):
        return [str(int(v))]
    if isinstance(v, float):
        return [_wl_number_string(v)]
    if isinstance(v, Expr) and v.head == "URL" and len(v.args) == 1 and isinstance(v.args[0], str):
        return [v.args[0]]
    if isinstance(v, Expr) and v.head == "CloudObject":
        raise NotImplementedError(f"{fname}: CloudObject values are not supported (there is no Wolfram Cloud)")
    if is_list(v):
        out = []
        for e in v.args:
            out.extend(_query_value(e, fname))
        return out
    from .core import R
    try:
        q = R(v)
    except Exception:
        q = None
    if q is not None and not isinstance(v, Symbol):
        import sympy as sp
        if isinstance(q, sp.Rational) and q.q == 1:
            return [str(q)]
        raise ValueError(f"{fname}: {_show(v)} is not a string, integer or machine number")
    if isinstance(v, Symbol) or isinstance(v, Expr):
        raise Unevaluated()
    raise TypeError(f"{fname}: cannot encode {_show(v)}")


def _flat_pairs(pairs, fname):
    out = []
    for k, v in pairs:
        ks = k if isinstance(k, str) else (_query_value(k, fname)[0] if not isinstance(k, Symbol) else None)
        if ks is None:
            raise Unevaluated()
        vals = _query_value(v, fname)
        if is_list(v) and not v.args:
            continue
        for s in vals:
            out.append((ks, s))
    return out


def _encode_query(pairs, enc, sort=False, remove_empty=False):
    if sort:
        pairs = sorted(pairs, key=lambda p: p[0])
    parts = []
    for k, v in pairs:
        if v == "":
            if remove_empty:
                continue
            parts.append(urllib.parse.quote_plus(k, safe="", encoding=enc))
        else:
            parts.append(urllib.parse.quote_plus(k, safe="", encoding=enc) + "=" +
                         urllib.parse.quote_plus(v, safe="", encoding=enc))
    return "&".join(parts)


def _decode_query(s, enc, fname):
    out = []
    if s == "":
        return out
    for part in s.split("&"):
        if part == "":
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        try:
            out.append((urllib.parse.unquote_plus(k, encoding=enc, errors="strict"),
                        urllib.parse.unquote_plus(v, encoding=enc, errors="strict")))
        except UnicodeDecodeError:
            raise ValueError(f"{fname}: the query string is not valid in the encoding {enc}")
    return out


def _qrules(pairs):
    return L(*(Expr("Rule", k, v) for k, v in pairs))


# ═══════════════════════════════════════ URL encoding ═══════════════════════════════════════

_QOPTS = {"CharacterEncoding", "SortQueries", "RemoveEmptyQueries"}


def _thread(x, f):
    if is_list(x):
        return L(*(_thread(e, f) for e in x.args))
    return f(x)


@fn("URLEncode", area=AREA, status="added",
    notes="URLEncode[s] percent-encodes every byte of the UTF-8 (or CharacterEncoding -> enc) form of s except "
          "letters, digits and -_.~ (a space becomes %20, as in Wolfram); integers and machine numbers are "
          "converted to their InputForm text, None/Null/Missing give \"\", True/False give \"true\"/\"false\", "
          "URL[\"u\"] encodes u; threads over lists. A symbol stays unevaluated; CloudObject raises "
          "NotImplementedError, other expressions raise TypeError/ValueError, and an unknown encoding ValueError. "
          "Check: reference (Wolfram documentation examples) and URLDecode round trips in tests.")
def _urlencode(x: Expr, *opts: Option) -> String | ListOf[String]:
    _, o = split_opts(opts, {"CharacterEncoding"}, "URLEncode")
    if _:
        raise TypeError("URLEncode: expected URLEncode[string, opts]")
    enc = _enc_opt(o, "URLEncode")

    def one(v):
        if isinstance(v, Expr) and v.head == "Association":
            raise TypeError("URLEncode: an association cannot be encoded (use URLQueryEncode)")
        s = _query_value(v, "URLEncode")
        if len(s) != 1:
            raise TypeError(f"URLEncode: cannot encode {_show(v)}")
        try:
            return urllib.parse.quote(s[0], safe="", encoding=enc, errors="strict")
        except UnicodeEncodeError as e:
            raise ValueError(f"URLEncode: {s[0][e.start]!r} cannot be represented in the encoding {enc}")
    return _thread(x, one)


@fn("URLDecode", area=AREA, status="added",
    notes="URLDecode[s] decodes %XX escapes (bytes interpreted in UTF-8 or CharacterEncoding -> enc) and turns "
          "+ into a space; threads over lists. Invalid bytes for the encoding raise ValueError, a non-string "
          "TypeError, a symbol stays unevaluated. Check: reference (Wolfram documentation examples) and "
          "URLEncode round trips in tests.")
def _urldecode(x: String | ListOf[String], *opts: Option) -> String | ListOf[String]:
    _, o = split_opts(opts, {"CharacterEncoding"}, "URLDecode")
    if _:
        raise TypeError("URLDecode: expected URLDecode[string, opts]")
    enc = _enc_opt(o, "URLDecode")

    def one(v):
        s = _str(v, "URLDecode")
        try:
            return urllib.parse.unquote_plus(s, encoding=enc, errors="strict")
        except UnicodeDecodeError:
            raise ValueError(f"URLDecode: {s!r} is not valid percent-encoded {enc}")
    return _thread(x, one)


def _qopts(o, fname):
    sort = _bool_opt(o.get("SortQueries", False), fname, "SortQueries")
    rem = _bool_opt(o.get("RemoveEmptyQueries", False), fname, "RemoveEmptyQueries")
    return sort, rem


@fn("URLQueryEncode", area=AREA, status="added",
    notes="URLQueryEncode[<|k1 -> v1, ...|>] or URLQueryEncode[{k1 -> v1, ...}] builds k1=v1&k2=v2 with keys and "
          "values form-encoded (a space becomes +); list values give repeated keys, None/Null/Missing give a bare "
          "key, True/False give true/false; options CharacterEncoding, \"SortQueries\" and \"RemoveEmptyQueries\". "
          "A non-rule argument raises TypeError, an unknown option ValueError. Check: reference (Wolfram "
          "documentation examples) and URLQueryDecode round trips in tests.")
def _urlqueryencode(x: Assoc | ListOf[Rule], *opts: Option) -> String:
    _, o = split_opts(opts, _QOPTS, "URLQueryEncode")
    if _:
        raise TypeError("URLQueryEncode: expected URLQueryEncode[rules, opts]")
    enc = _enc_opt(o, "URLQueryEncode")
    sort, rem = _qopts(o, "URLQueryEncode")
    pairs = _rules_deep(x, "URLQueryEncode")
    try:
        return _encode_query(_flat_pairs(pairs, "URLQueryEncode"), enc, sort, rem)
    except UnicodeEncodeError:
        raise ValueError(f"URLQueryEncode: a character cannot be represented in the encoding {enc}")


@fn("URLQueryDecode", area=AREA, status="added",
    notes="URLQueryDecode[\"k1=v1&k2=v2\"] gives {\"k1\" -> \"v1\", ...} (form decoding: + is a space, a key "
          "without = gets \"\"); URL[\"...\"] uses its query part; threads over (nested) lists; options "
          "CharacterEncoding, \"SortQueries\", \"RemoveEmptyQueries\". Bytes invalid in the encoding raise "
          "ValueError, a non-string TypeError. Check: reference (Wolfram documentation examples) and "
          "URLQueryEncode round trips in tests.")
def _urlquerydecode(x: String | ListOf[String], *opts: Option) -> ListOf[Rule]:
    _, o = split_opts(opts, _QOPTS, "URLQueryDecode")
    if _:
        raise TypeError("URLQueryDecode: expected URLQueryDecode[string, opts]")
    enc = _enc_opt(o, "URLQueryDecode")
    sort, rem = _qopts(o, "URLQueryDecode")

    def one(v):
        if isinstance(v, Expr) and v.head == "URL":
            s = urllib.parse.urlsplit(_url_str(v, "URLQueryDecode")).query
        else:
            s = _str(v, "URLQueryDecode")
        pairs = _decode_query(s, enc, "URLQueryDecode")
        if rem:
            pairs = [p for p in pairs if p[1] != ""]
        if sort:
            pairs = sorted(pairs, key=lambda p: p[0])
        return _qrules(pairs)
    return _thread(x, one)


# ═══════════════════════════════════════ URL parse / build ═══════════════════════════════════════

_COMPONENTS = ("Scheme", "User", "Domain", "Port", "Path", "Query", "Fragment")
_ALLCOMP = _COMPONENTS + ("PathString", "AbsoluteDomain", "AbsolutePath", "QueryString")


def _split_url(u, fname, enc="utf-8"):
    """Components of a URL string, Wolfram style (absent -> None, Path a list with "" for a leading /)."""
    p = urllib.parse.urlsplit(u)
    scheme = p.scheme or None
    netloc = p.netloc
    path = p.path
    if not scheme and not netloc and path and not path.startswith("/") and "." in path.split("/")[0] \
            and not u.startswith("."):
        # "www.wolfram.com/x": a bare host followed by a path
        first, _, rest = path.partition("/")
        netloc, path = first, ("/" + rest if _ else "")
    user, host, port = None, None, None
    if netloc:
        hostport = netloc
        if "@" in netloc:
            user, hostport = netloc.rsplit("@", 1)
            user = urllib.parse.unquote(user, encoding=enc)
        if hostport.startswith("["):
            host, _, rest = hostport[1:].partition("]")
            port = rest[1:] if rest.startswith(":") else None
        elif ":" in hostport:
            host, port = hostport.rsplit(":", 1)
        else:
            host = hostport
        if port is not None:
            port = int(port) if port.isdigit() else (port or None)
    segs = [] if path == "" else [urllib.parse.unquote(s, encoding=enc) for s in path.split("/")]
    q = _decode_query(p.query, enc, fname)
    return {"Scheme": scheme, "User": user, "Domain": host or None, "Port": port, "Path": segs,
            "Query": q, "Fragment": (urllib.parse.unquote(p.fragment, encoding=enc) if p.fragment else None),
            "PathString": path, "QueryString": p.query if p.query else None,
            "_netloc": netloc, "_scheme": p.scheme}


def _comp_value(c, name):
    v = c.get(name)
    if name == "Path":
        return L(*v)
    if name == "Query":
        return _qrules(v)
    if name == "AbsoluteDomain":
        if not c["_netloc"]:
            return NONE
        return (c["_scheme"] + "://" if c["_scheme"] else "") + c["_netloc"]
    if name == "AbsolutePath":
        dom = (c["_scheme"] + "://" if c["_scheme"] else "") + c["_netloc"]
        return dom + c["PathString"]
    if v is None:
        return NONE
    return v


@fn("URLParse", area=AREA, status="added",
    notes="URLParse[url] gives <|\"Scheme\", \"User\", \"Domain\", \"Port\", \"Path\", \"Query\", \"Fragment\"|> "
          "(absent components None, Port an integer, Path a list with \"\" first for an absolute path and last for "
          "a trailing /, Query decoded rules); URLParse[url, comp] and URLParse[url, {comp, ...}] also give "
          "\"PathString\", \"AbsoluteDomain\", \"AbsolutePath\", \"QueryString\"; URL[...] accepted; threads over a "
          "list of URLs; option CharacterEncoding. An unknown component raises ValueError, a non-string "
          "TypeError, CloudObject NotImplementedError. Check: reference (Wolfram documentation examples) and "
          "URLBuild[URLParse[u]] == u round trips in tests.")
def _urlparse(u: String | ListOf[String], *rest: Expr | Option) -> Assoc | Expr:
    args, o = split_opts(rest, {"CharacterEncoding"}, "URLParse")
    enc = _enc_opt(o, "URLParse")
    if len(args) > 1:
        raise TypeError("URLParse: expected URLParse[url] or URLParse[url, components]")
    comp = args[0] if args else None
    names = None
    if comp is not None:
        cl = comp.args if is_list(comp) else (comp,)
        for c in cl:
            if not isinstance(c, str):
                raise TypeError(f"URLParse: a component must be a string, not {_show(c)}")
            if c not in _ALLCOMP:
                raise ValueError(f"URLParse: unknown component {c!r} (known: {', '.join(_ALLCOMP)})")
        names = list(cl)

    def one(x):
        c = _split_url(_url_str(x, "URLParse"), "URLParse", enc)
        if names is None:
            return _assoc([(k, _comp_value(c, k)) for k in _COMPONENTS])
        vals = [_comp_value(c, k) for k in names]
        return L(*vals) if is_list(comp) else vals[0]
    if is_list(u):
        return L(*(one(x) for x in u.args))
    return one(u)


def _quote_seg(s, enc):
    return urllib.parse.quote(s, safe="!$&'()*+,;=:@", encoding=enc)


def _build_from_parts(scheme, user, domain, port, path, query, fragment, enc):
    out = ""
    if scheme:
        out += scheme + "://"
    if domain:
        if user:
            out += urllib.parse.quote(user, safe=":!$&'()*+,;=", encoding=enc) + "@"
        out += domain
        if port not in (None, ""):
            out += ":" + str(port)
    if path:
        if domain and not path.startswith("/"):
            path = "/" + path
        out += path
    if query:
        if domain and not path:
            out += "/"
        out += "?" + query
    if fragment:
        out += "#" + urllib.parse.quote(fragment, safe="!$&'()*+,;=:@/?", encoding=enc)
    return out


def _none(v):
    return v is None or is_sym(v, "None") or is_sym(v, "Automatic")


def _path_text(p, enc, fname):
    if isinstance(p, str):
        return p
    if is_list(p):
        segs = []
        for s in p.args:
            if not isinstance(s, (str, int)) or isinstance(s, bool):
                raise TypeError(f"{fname}: path elements must be strings, not {_show(s)}")
            segs.append(_quote_seg(str(s), enc))
        return "/".join(segs)
    raise TypeError(f"{fname}: a path must be a string or a list of strings, not {_show(p)}")


def _query_text(q, enc, fname):
    if _none(q):
        return []
    if isinstance(q, str):
        return _decode_query(q, enc, fname)
    return _flat_pairs(_rules(q, fname, "a query"), fname)


def build_url(spec, params, enc, sort, rem, fname):
    """URL string from a URL/path string, a path list or a components association plus extra query pairs."""
    extra = [] if params is None else _flat_pairs(_rules(params, fname, "the query parameters"), fname)
    if isinstance(spec, str) or (isinstance(spec, Expr) and spec.head in ("URL", "CloudObject")):
        u = _url_str(spec, fname)
        c = _split_url(u, fname, enc)
        base = u.split("#", 1)[0].split("?", 1)[0]
        q = c["Query"] + extra
        qs = _encode_query(q, enc, sort, rem)
        if qs:
            if c["_netloc"] and c["PathString"] == "":
                base += "/"
            base += "?" + qs
        if c["Fragment"]:
            base += "#" + urllib.parse.urlsplit(u).fragment
        return base
    if is_list(spec):
        parts = []
        for i, s in enumerate(spec.args):
            if not isinstance(s, str):
                raise TypeError(f"{fname}: path elements must be strings, not {_show(s)}")
            parts.append(s if i == 0 else _quote_seg(s, enc))
        base = "/".join(parts)
        qs = _encode_query(extra, enc, sort, rem)
        return base + ("?" + qs if qs else "")
    if _is_assoc(spec):
        d = {}
        for k, v in _assoc_items(spec):
            ks = _key(k)
            if ks not in _COMPONENTS:
                raise ValueError(f"{fname}: unknown URL component {_show(k)} (known: {', '.join(_COMPONENTS)})")
            d[ks] = v
        for k in ("Scheme", "User", "Domain", "Fragment"):
            if k in d and not _none(d[k]) and not isinstance(d[k], str):
                raise TypeError(f"{fname}: \"{k}\" must be a string, not {_show(d[k])}")
        port = d.get("Port")
        if _none(port):
            port = None
        elif not (isinstance(port, str) or (K.is_int(port) and not isinstance(port, bool))):
            raise TypeError(f"{fname}: \"Port\" must be an integer, not {_show(port)}")
        path = "" if _none(d.get("Path")) else _path_text(d["Path"], enc, fname)
        q = _query_text(d.get("Query"), enc, fname) + extra
        g = lambda k: None if _none(d.get(k)) else d.get(k)
        return _build_from_parts(g("Scheme"), g("User"), g("Domain"), port, path,
                                 _encode_query(q, enc, sort, rem), g("Fragment"), enc)
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a URL string, a path list or an association of components, "
                    f"not {_show(spec)}")


@fn("URLBuild", area=AREA, status="added",
    notes="URLBuild[path, params], URLBuild[{base, seg, ...}], URLBuild[assoc] and URLBuild[assoc, params] build a "
          "URL string: path segments after the first are percent-encoded and joined with /, query values follow "
          "URLQueryEncode (lists repeat the key, None gives a bare key), existing query parameters are kept, a "
          "domain with an empty path gets / before ?; association keys Scheme, User, Domain, Port, Path, Query, "
          "Fragment; options CharacterEncoding, \"SortQueries\", \"RemoveEmptyQueries\". Unknown keys or options "
          "raise ValueError, wrong kinds TypeError, CloudObject NotImplementedError. Check: reference (Wolfram "
          "documentation examples) and URLBuild[URLParse[u]] == u round trips in tests.")
def _urlbuild(spec: String | ListOf[String] | Assoc, *rest: Expr | Option) -> String:
    args, o = split_opts(rest, _QOPTS, "URLBuild")
    if len(args) > 1:
        raise TypeError("URLBuild: expected URLBuild[spec] or URLBuild[spec, params]")
    enc = _enc_opt(o, "URLBuild")
    sort, rem = _qopts(o, "URLBuild")
    return build_url(spec, args[0] if args else None, enc, sort, rem, "URLBuild")
