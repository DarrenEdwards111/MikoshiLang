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


# ═══════════════════════════════════════ session switches ═══════════════════════════════════════

def _sysvar(name, getter):
    E.SYSTEM_VARIABLES[name] = getter


def _own_or(name, default):
    st = K.state()
    if name in st.own:
        return ev(Symbol(name))
    return default


def internet_allowed():
    v = _own_or("$AllowInternet", True)
    if v is True or v is False:
        return v
    raise ValueError(f"$AllowInternet must be True or False, not {_show(v)}")


def _loopback(host):
    if host is None:
        return False
    h = host.strip("[]").lower()
    if h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


class Blocked(Exception):
    pass


def check_allowed(host, fname):
    if not _loopback(host) and not internet_allowed():
        raise Blocked(f"{fname}: internet access is disabled ($AllowInternet is False); only loopback hosts "
                      f"can be reached")


_sysvar("$AllowInternet", lambda: True)
record("$AllowInternet", AREA, "added",
       "$AllowInternet is True by default; setting it to False (e.g. Block[{$AllowInternet = False}, ...]) makes "
       "every W30 network head refuse non-loopback hosts before connecting (URL functions give "
       "Failure[\"ConnectionFailure\", ...], socket and lookup heads raise ValueError); 127.0.0.1/localhost stay "
       "reachable. A non-Boolean value raises ValueError when a network head reads it. The setting is not "
       "preserved between sessions. Check: reference (Wolfram documentation) and blocked/allowed cases in tests.")

_PROXY_KEYS = ("UseProxy", "HTTP", "HTTPS", "FTP", "Socks")


def _default_proxy_rules():
    return _assoc([("UseProxy", _sym("Automatic")), ("HTTP", NONE), ("HTTPS", NONE), ("FTP", NONE),
                   ("Socks", NONE)])


_sysvar("$DefaultProxyRules", _default_proxy_rules)
record("$DefaultProxyRules", AREA, "added",
       "$DefaultProxyRules is <|\"UseProxy\" -> Automatic, \"HTTP\" -> None, \"HTTPS\" -> None, \"FTP\" -> None, "
       "\"Socks\" -> None|> (the documented default); URLRead and friends honour it: UseProxy -> Automatic uses "
       "the system proxy settings (http_proxy/https_proxy/no_proxy), None or False uses none, Manual or True uses "
       "the \"HTTP\"/\"HTTPS\" entries {host, port}. Assign the whole association to change it. Socks proxies "
       "raise NotImplementedError; malformed settings raise ValueError. Check: reference (Wolfram documentation "
       "default value) and a proxied request to a local server in tests.")


def _proxy_for(scheme, host, fname):
    """(host, port) of the proxy for a request, or None."""
    rules = _own_or("$DefaultProxyRules", None)
    if rules is None:
        rules = _default_proxy_rules()
    if not _is_assoc(rules):
        raise ValueError(f"{fname}: $DefaultProxyRules must be an association")
    d = {_key(k): v for k, v in _assoc_items(rules)}
    use = d.get("UseProxy", _sym("Automatic"))
    if is_sym(use, "None") or use is False:
        return None
    if is_sym(use, "Automatic"):
        if _loopback(host) or urllib.request.proxy_bypass(host):
            return None
        p = urllib.request.getproxies().get(scheme)
        if not p:
            return None
        u = urllib.parse.urlsplit(p if "://" in p else "http://" + p)
        return (u.hostname, u.port or 80)
    if is_sym(use, "Manual") or use is True:
        if d.get("Socks") not in (None, NONE) and not is_sym(d.get("Socks"), "None"):
            raise NotImplementedError(f"{fname}: SOCKS proxies are not supported")
        spec = d.get("HTTPS" if scheme == "https" else "HTTP", NONE)
        if is_sym(spec, "None") or spec is None:
            return None
        if is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[0], str) and K.is_int(spec.args[1]):
            return (spec.args[0], int(spec.args[1]))
        if isinstance(spec, str):
            return (spec, 80)
        raise ValueError(f"{fname}: a proxy must be {{\"host\", port}}, not {_show(spec)}")
    raise ValueError(f"{fname}: \"UseProxy\" must be Automatic, None, Manual, True or False, not {_show(use)}")


_sysvar("$UserAgent", lambda: NONE)
_sysvar("$UserAgentString", lambda: NONE)
record("$UserAgent", AREA, "added",
       "$UserAgent is the user agent of the HTTP request that started the current evaluation; MikoshiLang "
       "evaluations are never started by an HTTP request, so it is None (Wolfram's documented value outside a "
       "web request). Check: exact (documented value) in tests.")
record("$UserAgentString", AREA, "added",
       "$UserAgentString is the user agent string of the HTTP request that started the current evaluation; "
       "None here (Wolfram's documented value outside a web request). The agent MikoshiLang itself sends is "
       "\"MikoshiLang HTTPClient <version>\". Check: exact (documented value) in tests.")


def _client_agent():
    from .. import __version__
    return f"MikoshiLang HTTPClient {__version__}"


# ═══════════════════════════════════════ cookies ═══════════════════════════════════════

_COOKIE_KEYS = ("Domain", "Path", "Name", "Content", "ExpirationDate", "CreationDate", "AllowSubdomains",
                "ConnectionType", "ScriptAccessible")


def _cookie_jar():
    v = _own_or("$Cookies", None)
    if v is None or not is_list(v):
        return []
    return list(v.args)


def _set_jar(items):
    K.state().own["$Cookies"] = L(*items)


_sysvar("$Cookies", lambda: L())


def _date_from_unix(t):
    d = _dt.datetime.fromtimestamp(t, _dt.timezone.utc)
    return Expr("DateObject", L(d.year, d.month, d.day, d.hour, d.minute, float(d.second)), "Instant",
                "Gregorian", 0.0)


def _unix_of(date, fname):
    if is_sym(date, "Automatic") or date is None:
        return None
    if isinstance(date, (int, float)) and not isinstance(date, bool):
        return float(date) - _EPOCH1900          # AbsoluteTime number
    if isinstance(date, Expr) and date.head == "DateObject":
        a = ev(Expr("AbsoluteTime", date))
        if isinstance(a, (int, float)):
            return float(a) - _EPOCH1900
        try:
            return float(K.to_float(a)) - _EPOCH1900 if hasattr(K, "to_float") else float(a) - _EPOCH1900
        except Exception:
            pass
    raise ValueError(f"{fname}: \"ExpirationDate\" must be a DateObject, an AbsoluteTime number or Automatic, "
                     f"not {_show(date)}")


def _cookie_dict(c, fname):
    if not _is_assoc(c):
        raise TypeError(f"{fname}: a cookie must be an association, not {_show(c)}")
    d = {}
    for k, v in _assoc_items(c):
        ks = _key(k)
        if ks not in _COOKIE_KEYS:
            raise ValueError(f"{fname}: unknown cookie element {_show(k)} (known: {', '.join(_COOKIE_KEYS)})")
        d[ks] = v
    for k in ("Name", "Content", "Path"):
        if k in d and not isinstance(d[k], str):
            raise TypeError(f"{fname}: cookie \"{k}\" must be a string, not {_show(d[k])}")
    if "Domain" in d and not (isinstance(d["Domain"], str) or is_sym(d["Domain"], "Automatic")):
        raise TypeError(f"{fname}: cookie \"Domain\" must be a string or Automatic, not {_show(d['Domain'])}")
    for k in ("AllowSubdomains", "ScriptAccessible"):
        if k in d and d[k] is not True and d[k] is not False and not is_sym(d[k], "Automatic"):
            raise ValueError(f"{fname}: cookie \"{k}\" must be True, False or Automatic")
    if "ConnectionType" in d and not (is_sym(d["ConnectionType"], "All") or d["ConnectionType"] == "HTTPS"):
        raise ValueError(f"{fname}: cookie \"ConnectionType\" must be All or \"HTTPS\"")
    if "ExpirationDate" in d:
        _unix_of(d["ExpirationDate"], fname)
    if "Name" not in d:
        raise ValueError(f"{fname}: a cookie needs a \"Name\"")
    return d


def _cookie_id(d):
    dom = d.get("Domain")
    return (dom if isinstance(dom, str) else "Automatic", d.get("Path", "/"), d.get("Name"))


def set_cookies(cookies, fname="SetCookies"):
    jar = _cookie_jar()
    for c in cookies:
        d = _cookie_dict(c, fname)
        cid = _cookie_id(d)
        jar = [j for j in jar if not (_is_assoc(j) and _cookie_id(_cookie_dict(j, fname)) == cid)]
        exp = d.get("ExpirationDate")
        t = _unix_of(exp, fname) if exp is not None else None
        if t is None or t > time.time():
            jar.append(c)
    _set_jar(jar)


@fn("SetCookies", area=AREA, status="added",
    notes="SetCookies[assoc] or SetCookies[{assoc, ...}] adds cookies (elements Domain, Path, Name, Content, "
          "ExpirationDate, CreationDate, AllowSubdomains, ConnectionType, ScriptAccessible) to $Cookies, replacing "
          "a cookie with the same Domain, Path and Name, and returns its argument; an already expired cookie "
          "removes the stored one. Cookies live in the session (Block[{$Cookies}, ...] localises them); there is "
          "no persistent $CookieStore. Unknown elements or bad values raise ValueError, non-associations "
          "TypeError, a missing Name ValueError. Check: reference (Wolfram documentation) and SetCookies/"
          "FindCookies/$Cookies round trips in tests.")
def _setcookies(c: Assoc | ListOf[Assoc]) -> Assoc | ListOf[Assoc]:
    if isinstance(c, Symbol):
        raise Unevaluated()
    items = list(c.args) if is_list(c) else [c]
    set_cookies(items)
    return c


def _str_match(s, pat):
    if isinstance(pat, str):
        if "*" in pat or "@" in pat:
            import fnmatch
            return isinstance(s, str) and fnmatch.fnmatchcase(s, pat.replace("@", "[!A-Z]*"))
        return s == pat
    return isinstance(s, str) and ev(Expr("StringMatchQ", s, pat)) is True


@fn("FindCookies", area=AREA, status="added",
    notes="FindCookies[] gives $Cookies; FindCookies[domain] the cookies whose Domain matches domain (a string, "
          "a * wildcard pattern as in FileNames, or a string pattern); FindCookies[assoc] those whose elements "
          "match every entry of assoc (string entries by pattern, others by equality). Unknown elements raise "
          "ValueError, other argument kinds TypeError. Check: reference (Wolfram documentation) and SetCookies "
          "round trips in tests.")
def _findcookies(spec: Expr = None) -> ListOf[Assoc]:
    jar = _cookie_jar()
    if spec is None:
        return L(*jar)
    if _is_assoc(spec):
        want = {}
        for k, v in _assoc_items(spec):
            ks = _key(k)
            if ks not in _COOKIE_KEYS:
                raise ValueError(f"FindCookies: unknown cookie element {_show(k)}")
            want[ks] = v
    elif isinstance(spec, str) or (isinstance(spec, Expr) and spec.head in ("StringExpression", "RegularExpression",
                                                                         "Alternatives")):
        want = {"Domain": spec}
    elif isinstance(spec, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"FindCookies: expected a domain or an association, not {_show(spec)}")
    out = []
    for j in jar:
        d = {_key(k): v for k, v in _assoc_items(j)}
        ok = True
        for k, v in want.items():
            have = d.get(k)
            if isinstance(v, str) or (isinstance(v, Expr) and v.head in ("StringExpression", "RegularExpression",
                                                                         "Alternatives")):
                ok = _str_match(have, v)
            else:
                ok = K.same(have, v) if have is not None else False
            if not ok:
                break
        if ok:
            out.append(j)
    return L(*out)


record("$Cookies", AREA, "added",
       "$Cookies is the session's list of cookie associations ({} initially). SetCookies and URLRead (with "
       "CookieFunction -> Automatic) append to it, requests with \"Cookies\" -> Automatic send the matching "
       "unexpired ones (domain, AllowSubdomains, path prefix and ConnectionType -> \"HTTPS\" respected), and "
       "assigning a list or Block[{$Cookies}, ...] replaces or localises it. Check: reference (Wolfram "
       "documentation) and cookies sent to and received from a local server in tests.")


def _cookies_for(scheme, host, path):
    out = []
    now = time.time()
    for j in _cookie_jar():
        if not _is_assoc(j):
            continue
        d = {_key(k): v for k, v in _assoc_items(j)}
        dom = d.get("Domain")
        if isinstance(dom, str):
            cd = dom.lstrip(".").lower()
            sub = d.get("AllowSubdomains", False)
            if not (host == cd or (sub is True and host.endswith("." + cd))):
                continue
        p = d.get("Path", "/")
        if isinstance(p, str) and not (path == p or path.startswith(p.rstrip("/") + "/") or p == "/"):
            continue
        if d.get("ConnectionType") == "HTTPS" and scheme != "https":
            continue
        exp = d.get("ExpirationDate")
        if exp is not None and not is_sym(exp, "Automatic"):
            t = _unix_of(exp, "URLRead")
            if t is not None and t <= now:
                continue
        out.append((d.get("Name"), d.get("Content", "")))
    return out


def _parse_set_cookie(value, host, path):
    sc = http.cookies.SimpleCookie()
    try:
        sc.load(value)
    except http.cookies.CookieError:
        return []
    out = []
    for name, m in sc.items():
        dom = m["domain"]
        if dom:
            domain, sub = dom, True
        else:
            domain, sub = host, False
        cpath = m["path"] or (path.rsplit("/", 1)[0] or "/") if (m["path"] or path) else "/"
        exp = _sym("Automatic")
        if m["max-age"]:
            try:
                exp = _date_from_unix(time.time() + int(m["max-age"]))
            except ValueError:
                pass
        elif m["expires"]:
            import email.utils
            try:
                exp = _date_from_unix(email.utils.parsedate_to_datetime(m["expires"]).timestamp())
            except (TypeError, ValueError):
                pass
        out.append(_assoc([("Domain", domain), ("Path", cpath), ("Name", name), ("Content", m.value),
                           ("ExpirationDate", exp), ("AllowSubdomains", sub),
                           ("ConnectionType", "HTTPS" if m["secure"] else _sym("All")),
                           ("ScriptAccessible", not bool(m["httponly"]))]))
    return out


# ═══════════════════════════════════════ HTTP requests ═══════════════════════════════════════

_REQ_KEYS = ("Scheme", "User", "Domain", "Port", "Path", "Query", "Fragment", "Method", "Body", "ContentType",
             "Username", "Password", "Cookies", "UserAgent", "Headers")
_REQ_OPTS = {"CharacterEncoding", "CookieFunction", "FollowRedirects", "Interactive", "TimeConstraint",
             "VerifySecurityCertificates"}
_READ_OPTS = _REQ_OPTS | {"Authentication", "ConnectionSettings"}


def _req_split(req, fname):
    """(url spec, element dict, option dict) of a request given as URL, assoc or HTTPRequest[...]."""
    if isinstance(req, Expr) and req.head == "HTTPRequest" and req.args:
        a = list(req.args)
        spec = a[0]
        rest, opts = split_opts(a[1:], _REQ_OPTS, fname)
        elems = rest[0] if rest else None
        if len(rest) > 1:
            raise TypeError(f"{fname}: malformed HTTPRequest")
    else:
        spec, elems, opts = req, None, {}
    d = {}
    if _is_assoc(spec):
        for k, v in _assoc_items(spec):
            ks = _key(k)
            if ks not in _REQ_KEYS:
                raise ValueError(f"{fname}: unknown request element {_show(k)} (known: {', '.join(_REQ_KEYS)})")
            d[ks] = v
        base = None
    else:
        base = _url_str(spec, fname)
    if elems is not None:
        if not _is_assoc(elems):
            raise TypeError(f"{fname}: request elements must be an association, not {_show(elems)}")
        for k, v in _assoc_items(elems):
            ks = _key(k)
            if ks not in _REQ_KEYS:
                raise ValueError(f"{fname}: unknown request element {_show(k)} (known: {', '.join(_REQ_KEYS)})")
            d[ks] = v
    return base, d, opts


def _req_url(base, d, fname, extra_query=None):
    comps = {k: d[k] for k in _COMPONENTS if k in d}
    if base is not None:
        c = _split_url(base, fname)
        for k in _COMPONENTS:
            if k not in comps and c.get(k) not in (None, [], ""):
                v = c[k]
                comps[k] = L(*v) if k == "Path" else (_qrules(v) if k == "Query" else v)
    if extra_query:
        q = _query_text(comps.get("Query"), "utf-8", fname) + extra_query
        comps["Query"] = _qrules(q)
    if not comps.get("Domain"):
        raise ValueError(f"{fname}: the request has no host")
    if not comps.get("Scheme"):
        comps["Scheme"] = "http"
    return build_url(_assoc(list(comps.items())), None, "utf-8", False, False, fname)


def _content_charset(ct):
    if not isinstance(ct, str):
        return None
    for part in ct.split(";")[1:]:
        k, _, v = part.strip().partition("=")
        if k.lower() == "charset" and v:
            return v.strip('"')
    return None


def _is_textual(ct):
    if not isinstance(ct, str):
        return False
    t = ct.split(";")[0].strip().lower()
    return t.startswith("text/") or t in ("application/json", "application/xml", "application/javascript",
                                          "application/x-www-form-urlencoded") or t.endswith("+xml") \
        or t.endswith("+json")


def _file_path(x):
    if isinstance(x, Expr) and x.head == "File" and len(x.args) == 1 and isinstance(x.args[0], str):
        return os.path.expanduser(x.args[0])
    return None


def _form_body(pairs, fname):
    """Body bytes and content type for a list of form rules (urlencoded or multipart)."""
    multipart = any(_file_path(v) is not None or _is_assoc(v) for _, v in pairs)
    if not multipart:
        return _encode_query(_flat_pairs(pairs, fname), "utf-8").encode("ascii"), \
            "application/x-www-form-urlencoded"
    boundary = "----MikoshiLangBoundary" + uuid.uuid4().hex
    out = []
    for k, v in pairs:
        if not isinstance(k, str):
            raise TypeError(f"{fname}: a form field name must be a string, not {_show(k)}")
        name, mime, filename = k, None, None
        if _is_assoc(v):
            spec = {_key(a): b for a, b in _assoc_items(v)}
            bad = set(spec) - {"Content", "Name", "MIMEType"}
            if bad:
                raise ValueError(f"{fname}: unknown multipart element(s) {sorted(bad)}")
            content = spec.get("Content", "")
            filename = spec.get("Name")
            mime = spec.get("MIMEType")
        else:
            content = v
        fp = _file_path(content)
        if fp is not None:
            with open(fp, "rb") as f:
                data = f.read()
            filename = filename or os.path.basename(fp)
            mime = mime or "application/octet-stream"
        elif is_byte_array(content):
            data = ba_bytes(content, fname)
            mime = mime or "application/octet-stream"
        else:
            vals = _query_value(content, fname)
            data = (vals[0] if len(vals) == 1 else ",".join(vals)).encode("utf-8")
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        head = f"--{boundary}\r\nContent-Disposition: {disp}\r\n" + \
            (f"Content-Type: {mime}\r\n" if mime else "") + "\r\n"
        out.append(head.encode("utf-8") + data + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def _body_bytes(body, ct, enc_opt, fname):
    """(bytes, content type) of a request "Body" setting."""
    if body is None or body == "":
        return b"", None
    if isinstance(body, str):
        if enc_opt is not None and is_sym(enc_opt, "None"):
            try:
                return body.encode("latin-1"), None
            except UnicodeEncodeError:
                raise ValueError(f"{fname}: with CharacterEncoding -> None the body must contain only "
                                 f"character codes below 256")
        if enc_opt is None or is_sym(enc_opt, "Automatic"):
            codec = _content_charset(ct) or "utf-8"
        else:
            codec = _codec(enc_opt, fname)
        try:
            return body.encode(codec), None
        except (UnicodeEncodeError, LookupError) as e:
            raise ValueError(f"{fname}: the body cannot be encoded as {codec}: {e}")
    if is_byte_array(body):
        return ba_bytes(body, fname), None
    fp = _file_path(body)
    if fp is not None:
        with open(fp, "rb") as f:
            return f.read(), None
    if is_list(body) and body.args and all(K.is_int(b) and not isinstance(b, bool) for b in body.args):
        bs = [int(b) for b in body.args]
        if any(b < 0 or b > 255 for b in bs):
            raise ValueError(f"{fname}: body bytes must be integers 0..255")
        return bytes(bs), None
    if _is_assoc(body) or (is_list(body) and all(_is_rule(r) for r in body.args)):
        return _form_body(_rules(body, fname), fname)
    if isinstance(body, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: unsupported \"Body\" setting {_show(body)}")


def _headers_list(h, fname):
    if h is None or is_sym(h, "None") or is_sym(h, "Automatic"):
        return []
    out = []
    if is_list(h) and all(is_list(p) and len(p.args) == 2 for p in h.args):
        pairs = [(p.args[0], p.args[1]) for p in h.args]
    else:
        pairs = _rules(h, fname, "\"Headers\"")
    for k, v in pairs:
        if not isinstance(k, str):
            raise TypeError(f"{fname}: a header name must be a string, not {_show(k)}")
        vals = _query_value(v, fname)
        out.append((k, ", ".join(vals)))
    return out


class Request:
    """A fully resolved HTTP request."""

    def __init__(self, url, method, headers, body, cookies, agent, auth):
        self.url, self.method, self.headers, self.body = url, method, headers, body
        self.cookies, self.agent, self.auth = cookies, agent, auth


def resolve_request(req, fname, extra_query=None, body_override=None, method_override=None):
    base, d, ropts = _req_split(req, fname)
    url = _req_url(base, d, fname, extra_query)
    method = d.get("Method", "GET")
    if method_override:
        method = method_override
    if not isinstance(method, str):
        raise TypeError(f"{fname}: Method must be a string such as \"GET\", not {_show(method)}")
    method = method.upper()
    ct = d.get("ContentType")
    if ct is not None and not (isinstance(ct, str) or is_sym(ct, "Automatic")):
        raise TypeError(f"{fname}: \"ContentType\" must be a string")
    ct = ct if isinstance(ct, str) else None
    body, auto_ct = _body_bytes(body_override if body_override is not None else d.get("Body"), ct,
                                ropts.get("CharacterEncoding"), fname)
    headers = _headers_list(d.get("Headers"), fname)
    if ct or auto_ct:
        headers = [h for h in headers if h[0].lower() != "content-type"] + [("Content-Type", ct or auto_ct)]
    user, pw = d.get("Username", ""), d.get("Password", "")
    for v, n in ((user, "Username"), (pw, "Password")):
        if not isinstance(v, str):
            raise TypeError(f"{fname}: \"{n}\" must be a string")
    auth = (user, pw) if user or pw else None
    agent = d.get("UserAgent", _sym("Automatic"))
    if is_sym(agent, "Automatic"):
        agent = _client_agent()
    elif not isinstance(agent, str) and not is_sym(agent, "None"):
        raise TypeError(f"{fname}: \"UserAgent\" must be a string")
    cookies = d.get("Cookies", _sym("Automatic"))
    return Request(url, method, headers, body, cookies, agent, auth), ropts


def _timeouts(v, fname):
    if v is None or is_sym(v, "Infinity") or (isinstance(v, Expr) and v.head == "DirectedInfinity"):
        return None, None
    if _is_assoc(v):
        d = {_key(k): x for k, x in _assoc_items(v)}
        bad = set(d) - {"Connecting", "Reading"}
        if bad:
            raise ValueError(f"{fname}: TimeConstraint keys must be \"Connecting\" and \"Reading\"")
        return _timeouts(d.get("Connecting"), fname)[0], _timeouts(d.get("Reading"), fname)[0]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v <= 0:
            raise ValueError(f"{fname}: TimeConstraint must be positive")
        return float(v), float(v)
    raise ValueError(f"{fname}: TimeConstraint must be a positive number, Infinity or an association")


def _speed(v, fname, what):
    if v is None or is_sym(v, "Automatic") or is_sym(v, "Infinity"):
        return None
    if K.is_int(v) and not isinstance(v, bool) and int(v) > 0:
        return int(v)
    if isinstance(v, float) and v > 0:
        return v
    if isinstance(v, Expr) and v.head == "Quantity":
        raise NotImplementedError(f"{fname}: Quantity speeds are not supported (give bytes per second)")
    raise ValueError(f"{fname}: \"{what}\" must be a positive number of bytes per second or Automatic")


class Transfer:
    """Outcome of one URL request (after redirects)."""

    def __init__(self):
        self.status = None
        self.reason = ""
        self.headers = []
        self.body = b""
        self.cookies = []
        self.url = None
        self.times = {}


def _failure(fname, url, msg, req_expr=None):
    items = [("MessageTemplate", "`1`"), ("MessageParameters", L(msg)), ("Message", msg), ("URL", url)]
    if req_expr is not None:
        items.append(("HTTPRequest", req_expr))
    return Expr("Failure", "ConnectionFailure", _assoc(items))


class ConnectionFailed(Exception):
    pass


def perform(r, opts, fname, follow=None, max_redirects=20):
    """Execute a resolved Request with http.client; raises ConnectionFailed/Blocked on network errors."""
    if follow is None:
        follow = _bool_opt(opts.get("FollowRedirects", True), fname, "FollowRedirects")
    verify = _bool_opt(opts.get("VerifySecurityCertificates", True), fname, "VerifySecurityCertificates")
    _bool_opt(opts.get("Interactive", True), fname, "Interactive")
    tc, tr = _timeouts(opts.get("TimeConstraint"), fname)
    cf = opts.get("CookieFunction", _sym("Automatic"))
    cs = opts.get("ConnectionSettings", _sym("Automatic"))
    down = up = None
    if not is_sym(cs, "Automatic"):
        if not _is_assoc(cs):
            raise ValueError(f"{fname}: ConnectionSettings must be an association or Automatic")
        csd = {_key(k): v for k, v in _assoc_items(cs)}
        bad = set(csd) - {"MaxDownloadSpeed", "MaxUploadSpeed"}
        if bad:
            raise ValueError(f"{fname}: unknown ConnectionSettings key(s) {sorted(bad)}")
        down = _speed(csd.get("MaxDownloadSpeed"), fname, "MaxDownloadSpeed")
        up = _speed(csd.get("MaxUploadSpeed"), fname, "MaxUploadSpeed")
    auth = opts.get("Authentication", _sym("Automatic"))
    extra_headers, extra_query = [], []
    if not (is_sym(auth, "Automatic") or is_sym(auth, "None")):
        if not _is_assoc(auth):
            if isinstance(auth, Expr) and auth.head in ("PermissionsKey", "SecuredAuthenticationKey"):
                raise NotImplementedError(f"{fname}: {auth.head} authentication needs the Wolfram Cloud and is "
                                          f"not supported")
            raise ValueError(f"{fname}: Authentication must be Automatic, None or an association")
        ad = {_key(k): v for k, v in _assoc_items(auth)}
        bad = set(ad) - {"Username", "Password", "Headers", "Query"}
        if bad:
            raise NotImplementedError(f"{fname}: Authentication element(s) {sorted(bad)} are not supported")
        if "Username" in ad or "Password" in ad:
            u, p = ad.get("Username", ""), ad.get("Password", "")
            if not isinstance(u, str) or not isinstance(p, str):
                raise TypeError(f"{fname}: Authentication Username and Password must be strings")
            r.auth = (u, p)
        if "Headers" in ad:
            extra_headers = _headers_list(ad["Headers"], fname)
        if "Query" in ad:
            extra_query = _flat_pairs(_rules(ad["Query"], fname, "the Authentication query"), fname)
    url, method, body = r.url, r.method, r.body
    if extra_query:
        url = build_url(url, _qrules(extra_query), "utf-8", False, False, fname)
    t = Transfer()
    t0 = time.perf_counter()
    for hop in range(max_redirects + 1):
        p = urllib.parse.urlsplit(url)
        scheme = (p.scheme or "http").lower()
        if scheme not in ("http", "https"):
            raise NotImplementedError(f"{fname}: the {scheme!r} scheme is not supported (only http and https)")
        host = p.hostname
        if not host:
            raise ValueError(f"{fname}: {url!r} has no host")
        check_allowed(host, fname)
        port = p.port or (443 if scheme == "https" else 80)
        path = (p.path or "/") + ("?" + p.query if p.query else "")
        proxy = _proxy_for(scheme, host, fname)
        if proxy is not None:
            check_allowed(proxy[0], fname)
        hdrs = {}
        if r.agent is not None and not is_sym(r.agent, "None"):
            hdrs["User-Agent"] = r.agent
        for k, v in r.headers + extra_headers:
            hdrs[k] = v
        if r.auth:
            tok = base64.b64encode(f"{r.auth[0]}:{r.auth[1]}".encode("utf-8")).decode("ascii")
            hdrs["Authorization"] = "Basic " + tok
        if is_sym(r.cookies, "Automatic"):
            ck = _cookies_for(scheme, host.lower(), p.path or "/")
        elif is_sym(r.cookies, "None"):
            ck = []
        elif is_list(r.cookies):
            ck = []
            for c in r.cookies.args:
                d = _cookie_dict(c, fname)
                ck.append((d["Name"], d.get("Content", "")))
        else:
            raise ValueError(f"{fname}: \"Cookies\" must be Automatic, None or a list of cookie associations")
        if ck:
            hdrs["Cookie"] = "; ".join(f"{n}={v}" for n, v in ck)
        try:
            ta = time.perf_counter()
            if scheme == "https":
                import ssl
                ctx = ssl.create_default_context() if verify else ssl._create_unverified_context()
                if proxy:
                    conn = http.client.HTTPSConnection(proxy[0], proxy[1], timeout=tc, context=ctx)
                    conn.set_tunnel(host, port)
                else:
                    conn = http.client.HTTPSConnection(host, port, timeout=tc, context=ctx)
            else:
                if proxy:
                    conn = http.client.HTTPConnection(proxy[0], proxy[1], timeout=tc)
                    path = f"http://{p.netloc.rsplit('@', 1)[-1]}{path}"
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=tc)
            try:
                conn.connect()
                tb = time.perf_counter()
                if tr is not None and conn.sock is not None:
                    conn.sock.settimeout(tr)
                conn.putrequest(method, path, skip_accept_encoding=True)
                for k, v in hdrs.items():
                    conn.putheader(k, v)
                if body or method in ("POST", "PUT", "PATCH"):
                    conn.putheader("Content-Length", str(len(body)))
                conn.endheaders()
                if body:
                    if up:
                        for i in range(0, len(body), 4096):
                            s0 = time.perf_counter()
                            conn.send(body[i:i + 4096])
                            dt = len(body[i:i + 4096]) / up - (time.perf_counter() - s0)
                            if dt > 0:
                                time.sleep(dt)
                    else:
                        conn.send(body)
                resp = conn.getresponse()
                tfirst = time.perf_counter()
                chunks = []
                while True:
                    s0 = time.perf_counter()
                    chunk = resp.read(65536 if not down else max(1, min(65536, int(down) // 4 or 1)))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if down:
                        dt = len(chunk) / down - (time.perf_counter() - s0)
                        if dt > 0:
                            time.sleep(dt)
                data = b"".join(chunks)
                tend = time.perf_counter()
            finally:
                conn.close()
        except (OSError, http.client.HTTPException) as e:
            raise ConnectionFailed(f"{fname}: could not read {url}: {e}")
        t.times = {"HostConnection": tb - t0, "TransferInitiation": tfirst - t0, "TransactionTotal": tend - t0,
                   "PreTransfer": tb - t0, "HopStart": ta - t0}
        hdr_list = list(resp.getheaders())
        received = []
        for k, v in hdr_list:
            if k.lower() == "set-cookie":
                received.extend(_parse_set_cookie(v, host.lower(), p.path or "/"))
        _apply_cookie_function(cf, received, fname)
        t.cookies.extend(received)
        t.status, t.reason, t.headers, t.body, t.url = resp.status, resp.reason, hdr_list, data, url
        loc = resp.getheader("Location")
        if follow and resp.status in (301, 302, 303, 307, 308) and loc:
            url = urllib.parse.urljoin(url, loc)
            if resp.status == 303 or (resp.status in (301, 302) and method not in ("GET", "HEAD")):
                method, body = "GET", b""
                r.headers = [h for h in r.headers if h[0].lower() not in ("content-type",)]
            continue
        return t
    raise ConnectionFailed(f"{fname}: more than {max_redirects} redirects")


def _apply_cookie_function(cf, cookies, fname):
    if not cookies or is_sym(cf, "None"):
        return
    if is_sym(cf, "Automatic") or is_sym(cf, "SetCookies"):
        set_cookies(cookies, fname)
        return
    for c in cookies:
        apply(cf, c)


# ═══════════════════════════════════════ HTTPRequest object ═══════════════════════════════════════

def _req_props(req, fname="HTTPRequest"):
    base, d, opts = _req_split(req, fname)
    try:
        url = _req_url(base, d, fname)
    except ValueError:
        url = base
    c = _split_url(url, fname) if isinstance(url, str) else {}
    body_spec = d.get("Body", "")
    body, auto_ct = _body_bytes(body_spec, d.get("ContentType") if isinstance(d.get("ContentType"), str) else None,
                                opts.get("CharacterEncoding"), fname)
    props = {}
    for k in _COMPONENTS:
        props[k] = _comp_value(c, k) if c else NONE
    props["Method"] = d.get("Method", "GET")
    props["Body"] = body.decode("utf-8", errors="strict") if _decodable(body) else body.decode("latin-1")
    props["BodyByteArray"] = byte_array(body)
    props["BodyBytes"] = L(*body)
    if _is_assoc(body_spec) or (is_list(body_spec) and body_spec.args and all(_is_rule(r) for r in body_spec.args)):
        props["FormRules"] = _qrules(_flat_pairs(_rules(body_spec, fname), fname)) if auto_ct and \
            auto_ct.startswith("application/x-www") else L(*(Expr("Rule", k, v) for k, v in _rules(body_spec, fname)))
    else:
        props["FormRules"] = L()
    props["PathString"] = c.get("PathString", "") if c else ""
    props["QueryString"] = c.get("QueryString") or "" if c else ""
    props["AbsolutePath"] = _comp_value(c, "AbsolutePath") if c else NONE
    props["AbsoluteDomain"] = _comp_value(c, "AbsoluteDomain") if c else NONE
    props["ContentType"] = d.get("ContentType", auto_ct if auto_ct else _sym("Automatic"))
    props["Username"] = d.get("Username", "")
    props["Password"] = d.get("Password", "")
    props["Cookies"] = d.get("Cookies", _sym("Automatic"))
    props["UserAgent"] = d.get("UserAgent", _sym("Automatic"))
    props["Headers"] = L(*(Expr("Rule", k, v) for k, v in _headers_list(d.get("Headers"), fname)))
    props["URL"] = url if url is not None else NONE
    return props


def _decodable(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _prop_key(p):
    if isinstance(p, Symbol) and p.name == "Method":
        return "Method", p
    if isinstance(p, str):
        return p, p
    return None, p


def _props_lookup(props, args, fname, order=None):
    names = order or list(props)
    if not args:
        return _assoc([(k, props[k]) for k in names])
    if len(args) != 1:
        return K.UNHANDLED
    a = args[0]
    if is_list(a):
        out = []
        for p in a.args:
            k, orig = _prop_key(p)
            if k not in props:
                raise ValueError(f"{fname}: unknown property {_show(p)}")
            out.append((orig, props[k]))
        return _assoc(out)
    k, orig = _prop_key(a)
    if k == "Properties":
        return L(*names)
    if k == "PropertyAssociation":
        return _assoc([(n, props[n]) for n in names])
    if k is None or k not in props:
        raise ValueError(f"{fname}: unknown property {_show(a)}")
    return props[k]


def _req_hook(head, args, ev_):
    if not head.args:
        return K.UNHANDLED
    return _props_lookup(_req_props(head), args, "HTTPRequest")


KL.COMPOUND_HEAD_HOOKS["HTTPRequest"] = _req_hook


@fn("HTTPRequest", area=AREA, status="added",
    notes="HTTPRequest[url], HTTPRequest[assoc], HTTPRequest[url, assoc] and trailing options (CharacterEncoding, "
          "CookieFunction, FollowRedirects, Interactive, TimeConstraint, VerifySecurityCertificates) are a "
          "symbolic request (nothing is sent); HTTPRequest[url] is shown as HTTPRequest[url, <||>]. Elements: URL "
          "components plus Method (\"GET\"), Body (string, ByteArray, File, byte list, form rules or "
          "multipart specs), ContentType, Username, Password, Cookies, UserAgent, Headers. req[\"prop\"], "
          "req[{props}] and req[] give components, elements, \"Body\", \"BodyByteArray\", \"BodyBytes\", "
          "\"FormRules\", \"PathString\", \"QueryString\", \"AbsolutePath\", \"AbsoluteDomain\", \"URL\", "
          "\"Properties\", \"PropertyAssociation\". Unknown elements, options or properties raise ValueError, "
          "wrong kinds TypeError, CloudObject NotImplementedError. Check: reference (Wolfram documentation "
          "examples) and requests echoed back by a local server in tests.")
def _httprequest(spec: String | Assoc, *rest: Assoc | Option) -> Expr:
    if isinstance(spec, Symbol):
        raise Unevaluated()
    args, _o = split_opts(rest, _REQ_OPTS, "HTTPRequest")
    if len(args) > 1:
        raise TypeError("HTTPRequest: expected HTTPRequest[url], HTTPRequest[assoc] or HTTPRequest[url, assoc]")
    req = Expr("HTTPRequest", spec, *rest)
    _req_props(req)                       # validates every element
    if not _is_assoc(spec) and not args:
        return Expr("HTTPRequest", spec, Expr("Association"), *rest)
    raise Unevaluated()                   # already canonical: stays as it is


# ═══════════════════════════════════════ HTTPResponse / HTTPRedirect ═══════════════════════════════════════

_RESP_META = ("StatusCode", "ContentType", "Cookies", "Headers", "CharacterEncoding")
_RESP_PROPS = ("StatusCode", "ContentType", "Cookies", "Headers", "Body", "BodyByteArray", "BodyBytes",
               "CharacterEncoding", "StatusCodeDescription")
_WL_ENC = {"utf-8": "UTF8", "utf8": "UTF8", "iso-8859-1": "ISOLatin1", "latin-1": "ISOLatin1", "latin1": "ISOLatin1",
           "us-ascii": "ASCII", "ascii": "ASCII", "windows-1252": "WindowsANSI", "cp1252": "WindowsANSI"}


def _py_codec(name, fname):
    import codecs
    try:
        return codecs.lookup(_codec(name, fname) if name in _WL_TO_PY else name).name
    except (LookupError, ValueError):
        raise ValueError(f"{fname}: unknown character encoding {name!r}")


_WL_TO_PY = {"UTF8", "UTF-8", "ISOLatin1", "ISO8859-1", "ASCII", "WindowsANSI", "ShiftJIS", "MacintoshRoman"}


def _status_text(code):
    import http as _h
    try:
        return _h.HTTPStatus(code).phrase
    except ValueError:
        return ""


def _resp_parts(head, fname="HTTPResponse"):
    a = list(head.args)
    if not a:
        raise TypeError(f"{fname}: expected {fname}[body] or {fname}[body, metadata]")
    body = a[0]
    rest, opts = split_opts(a[1:], {"CharacterEncoding"}, fname)
    meta = rest[0] if rest else Expr("Association")
    if len(rest) > 1 or not _is_assoc(meta):
        raise TypeError(f"{fname}: the metadata must be an association")
    m = {}
    for k, v in _assoc_items(meta):
        ks = _key(k)
        if ks not in _RESP_META:
            raise ValueError(f"{fname}: unknown metadata element {_show(k)} (known: {', '.join(_RESP_META)})")
        m[ks] = v
    code = m.get("StatusCode", 200)
    if not (K.is_int(code) and not isinstance(code, bool) and 100 <= int(code) <= 599):
        raise ValueError(f"{fname}: \"StatusCode\" must be an integer from 100 to 599, not {_show(code)}")
    ct = m.get("ContentType", "text/html")
    if not (isinstance(ct, str) or is_sym(ct, "None")):
        raise TypeError(f"{fname}: \"ContentType\" must be a string")
    hdrs = m.get("Headers", L())
    hl = _headers_list(hdrs, fname)
    cookies = m.get("Cookies", L())
    if not is_list(cookies):
        raise TypeError(f"{fname}: \"Cookies\" must be a list of cookie associations")
    for c in cookies.args:
        _cookie_dict(c, fname)
    enc = opts.get("CharacterEncoding", m.get("CharacterEncoding", _sym("Automatic")))
    if isinstance(body, str):
        raw = None
    elif is_byte_array(body):
        raw = ba_bytes(body, fname)
    elif is_list(body) and all(K.is_int(b) and not isinstance(b, bool) and 0 <= int(b) <= 255 for b in body.args):
        raw = bytes(int(b) for b in body.args)
    elif isinstance(body, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"{fname}: the body must be a string, a ByteArray or a list of bytes, not {_show(body)}")
    if is_sym(enc, "Automatic"):
        cs = _content_charset(ct)
        codec = cs if cs else ("utf-8" if _is_textual(ct) else None)
    elif is_sym(enc, "None"):
        codec = None
    elif isinstance(enc, str):
        codec = enc
    else:
        raise ValueError(f"{fname}: CharacterEncoding must be Automatic, None or an encoding name")
    if codec is not None:
        codec = _py_codec(codec, fname)
    if raw is None:
        try:
            raw = body.encode(codec) if codec else body.encode("latin-1")
        except UnicodeEncodeError:
            raise ValueError(f"{fname}: the body cannot be represented in the encoding {codec or 'None'}")
    return int(code), ct, hl, cookies, raw, codec


def _resp_props(head, fname="HTTPResponse"):
    code, ct, hl, cookies, raw, codec = _resp_parts(head, fname)
    if codec:
        try:
            text = raw.decode(codec)
        except UnicodeDecodeError:
            raise ValueError(f"{fname}: the body is not valid {codec}; use \"BodyByteArray\"")
    else:
        text = raw.decode("latin-1")
    return {"StatusCode": code, "ContentType": ct, "Cookies": cookies,
            "Headers": L(*(Expr("Rule", k, v) for k, v in hl)), "Body": text, "BodyByteArray": byte_array(raw),
            "BodyBytes": L(*raw), "CharacterEncoding": _WL_ENC.get(codec, codec) if codec else NONE,
            "StatusCodeDescription": _status_text(code)}


def _resp_hook(head, args, ev_):
    if not head.args:
        return K.UNHANDLED
    return _props_lookup(_resp_props(head), args, "HTTPResponse", list(_RESP_PROPS))


KL.COMPOUND_HEAD_HOOKS["HTTPResponse"] = _resp_hook


@fn("HTTPResponse", area=AREA, status="added",
    notes="HTTPResponse[body] and HTTPResponse[body, <|\"StatusCode\" -> 200, \"ContentType\" -> \"text/html\", "
          "\"Cookies\" -> {}, \"Headers\" -> {}|>] (body a string, ByteArray or byte list; option "
          "CharacterEncoding) represent an HTTP response; URLRead returns one with a ByteArray body. "
          "resp[\"prop\"], resp[{props}] and resp[] give the metadata plus \"Body\" (decoded with the charset of "
          "ContentType, UTF-8 for text types, byte-per-character otherwise), \"BodyByteArray\", \"BodyBytes\", "
          "\"CharacterEncoding\", \"StatusCodeDescription\" (http.HTTPStatus phrase) and \"Properties\". Bad "
          "metadata raises ValueError/TypeError, a body that is invalid in its charset raises ValueError. "
          "Check: reference (Wolfram documentation and RFC 9110 status phrases) and local-server responses in "
          "tests.")
def _httpresponse(body: String | Expr, *rest: Assoc | Option) -> Expr:
    _resp_parts(Expr("HTTPResponse", body, *rest))
    raise Unevaluated()                   # validated; the object stays as given


_REDIR_META = ("StatusCode", "ContentType", "Cookies", "Headers")


def _redir_props(head):
    a = list(head.args)
    if not a or len(a) > 2:
        raise TypeError("HTTPRedirect: expected HTTPRedirect[uri] or HTTPRedirect[uri, metadata]")
    uri = _url_str(a[0], "HTTPRedirect")
    meta = a[1] if len(a) > 1 else Expr("Association")
    if not _is_assoc(meta):
        raise TypeError("HTTPRedirect: the metadata must be an association")
    m = {}
    for k, v in _assoc_items(meta):
        ks = _key(k)
        if ks not in _REDIR_META:
            raise ValueError(f"HTTPRedirect: unknown metadata element {_show(k)} (known: {', '.join(_REDIR_META)})")
        m[ks] = v
    code = m.get("StatusCode", 302)
    if not (K.is_int(code) and not isinstance(code, bool) and 300 <= int(code) <= 399):
        raise ValueError(f"HTTPRedirect: \"StatusCode\" must be a 3xx integer, not {_show(code)}")
    hdrs = m.get("Headers", _sym("Automatic"))
    hl = _headers_list(hdrs, "HTTPRedirect")
    if not any(k.lower() == "location" for k, _ in hl):
        hl.append(("Location", uri))
    cookies = m.get("Cookies", L())
    if not is_list(cookies):
        raise TypeError("HTTPRedirect: \"Cookies\" must be a list")
    ct = m.get("ContentType", "text/html")
    if not isinstance(ct, str):
        raise TypeError("HTTPRedirect: \"ContentType\" must be a string")
    return {"StatusCode": int(code), "ContentType": ct, "Cookies": cookies,
            "Headers": L(*(Expr("Rule", k, v) for k, v in hl)), "Location": uri,
            "StatusCodeDescription": _status_text(int(code))}


KL.COMPOUND_HEAD_HOOKS["HTTPRedirect"] = lambda head, args, ev_: (
    _props_lookup(_redir_props(head), args, "HTTPRedirect") if head.args else K.UNHANDLED)


@fn("HTTPRedirect", area=AREA, status="added",
    notes="HTTPRedirect[uri] and HTTPRedirect[uri, <|\"StatusCode\" -> 302, \"ContentType\", \"Cookies\", "
          "\"Headers\"|>] represent a redirect response (validated, otherwise inert, since there is no cloud "
          "deployment); r[\"prop\"] gives the metadata, \"Location\" and \"StatusCodeDescription\", and "
          "\"Headers\" gains Location -> uri when not given. A non-3xx status or unknown element raises "
          "ValueError, CloudObject NotImplementedError. Check: reference (Wolfram documentation defaults, RFC 9110 "
          "phrases) in tests.")
def _httpredirect(uri: String, *meta: Assoc) -> Expr:
    if isinstance(uri, Symbol):
        raise Unevaluated()
    _redir_props(Expr("HTTPRedirect", uri, *meta))
    raise Unevaluated()


# ═══════════════════════════════════════ URLRead / URLExecute / URLDownload ═══════════════════════════════════════

_ELEMS = ("Body", "BodyByteArray", "BodyBytes", "Headers", "Cookies", "ContentType", "StatusCode",
          "StatusCodeDescription")


def _response_expr(t):
    ct = None
    for k, v in t.headers:
        if k.lower() == "content-type":
            ct = v
    meta = [("StatusCode", t.status), ("ContentType", ct if ct is not None else NONE),
            ("Headers", L(*(Expr("Rule", k, v) for k, v in t.headers))), ("Cookies", L(*t.cookies))]
    return Expr("HTTPResponse", byte_array(t.body), _assoc(meta))


def _request_expr(req):
    if isinstance(req, Expr) and req.head == "HTTPRequest":
        return req
    if _is_assoc(req):
        return Expr("HTTPRequest", req)
    return Expr("HTTPRequest", req, Expr("Association"))


def _read_one(req, opts, fname, **kw):
    """(Transfer, None) or (None, Failure expression)."""
    r, ropts = resolve_request(req, fname, **{k: v for k, v in kw.items() if k != "follow"})
    o = dict(ropts)
    o.update(opts)
    try:
        return perform(r, o, fname, follow=kw.get("follow")), None
    except (ConnectionFailed, Blocked) as e:
        return None, _failure(fname, r.url, str(e), _request_expr(req))


def _elem(t, resp, e, fname):
    if e is Symbol("Automatic") or is_sym(e, "Automatic"):
        return resp
    if is_sym(e, "All"):
        return _assoc([(k, _elem(t, resp, k, fname)) for k in _ELEMS])
    if is_list(e):
        return _assoc([(k, _elem(t, resp, k, fname)) for k in e.args])
    if e == "Stream":
        raise NotImplementedError(f"{fname}: the \"Stream\" element is not supported (read \"BodyByteArray\")")
    if not isinstance(e, str) or e not in _ELEMS:
        raise ValueError(f"{fname}: unknown element {_show(e)} (known: {', '.join(_ELEMS)}, All, Automatic)")
    return _resp_props(resp, fname)[e]


def _is_req(x):
    return isinstance(x, str) or _is_assoc(x) or (isinstance(x, Expr) and x.head in ("HTTPRequest", "URL",
                                                                                   "CloudObject"))


@fn("URLRead", area=AREA, status="added",
    notes="URLRead[req], URLRead[req, elem], URLRead[req, {elems}] and URLRead[{req, ...}, ...] send HTTP(S) "
          "requests (req a URL, URL[...], association or HTTPRequest) with http.client and give an HTTPResponse "
          "(ByteArray body) or the elements \"Body\", \"BodyByteArray\", \"BodyBytes\", \"Headers\", \"Cookies\", "
          "\"ContentType\", \"StatusCode\", \"StatusCodeDescription\", All, Automatic; options Authentication "
          "(Automatic/None or <|\"Username\", \"Password\", \"Headers\", \"Query\"|>), CharacterEncoding, "
          "ConnectionSettings, CookieFunction, FollowRedirects (True), Interactive (no dialogs are ever shown), "
          "TimeConstraint, VerifySecurityCertificates. Lists are read one after another. A connection failure, "
          "too many redirects or $AllowInternet = False for a non-loopback host gives the documented "
          "Failure[\"ConnectionFailure\", <|..., \"URL\", \"HTTPRequest\"|>]. \"Stream\", non-HTTP schemes and "
          "cloud authentication raise NotImplementedError; unknown elements/options ValueError. Check: reference "
          "(Wolfram documentation) and round trips against a local http.server in tests.")
def _urlread(req: Expr, *rest: Expr | Option) -> Expr:
    args, opts = split_opts(rest, _READ_OPTS, "URLRead")
    if len(args) > 1:
        raise TypeError("URLRead: expected URLRead[req] or URLRead[req, elements]")
    elem = args[0] if args else _sym("Automatic")
    if isinstance(req, Symbol):
        raise Unevaluated()
    if is_list(req):
        return L(*(_urlread(q, *rest) for q in req.args))
    if not _is_req(req):
        raise TypeError(f"URLRead: expected a URL, an association or an HTTPRequest, not {_show(req)}")
    t, fail = _read_one(req, opts, "URLRead")
    if fail is not None:
        return fail
    resp = _response_expr(t)
    return _elem(t, resp, elem, "URLRead")


_MIME_FORMATS = {"application/json": "JSON", "text/json": "JSON", "text/plain": "Text", "text/csv": "CSV",
                 "text/tab-separated-values": "TSV", "application/xml": "XML", "text/xml": "XML",
                 "application/x-wl": "Package", "application/vnd.wolfram.wl": "Package"}


def _import_body(raw, ct, fmt, fname):
    if fmt is not None:
        return ev(Expr("ImportByteArray", byte_array(raw), fmt))
    mime = ct.split(";")[0].strip().lower() if isinstance(ct, str) else ""
    f = _MIME_FORMATS.get(mime)
    if mime in ("text/html", "application/xhtml+xml"):
        raise NotImplementedError(f"{fname}: importing HTML pages is not supported (Wolfram gives the page's plain "
                                  f"text); give a format such as \"String\", or use URLRead[url, \"Body\"]")
    if f is None and mime.startswith("text/"):
        f = "Text"
    if f is None:
        return ev(Expr("ImportByteArray", byte_array(raw)))
    if f == "Text":
        cs = _content_charset(ct) or "utf-8"
        try:
            return raw.decode(cs)
        except (UnicodeDecodeError, LookupError):
            raise ValueError(f"{fname}: the response is not valid {cs} text")
    return ev(Expr("ImportByteArray", byte_array(raw), f))


def _simple_params(pairs):
    for _, v in pairs:
        if _file_path(v) is not None or _is_assoc(v) or is_byte_array(v):
            return False
        if isinstance(v, str) and len(v) > 512:
            return False
    return True


@fn("URLExecute", area=AREA, status="added",
    notes="URLExecute[url], URLExecute[url, params], URLExecute[url, params, fmt] (url a URL, association or "
          "HTTPRequest; params rules or an association) send the request (simple parameters go in the query "
          "string with GET, files, byte arrays and long values are POSTed as a form) and Import the body: by "
          "the given format, else by content type (JSON -> \"JSON\", text/* -> string, CSV, TSV, XML; HTML raises "
          "NotImplementedError since the plain-text extraction is not reproduced), else by sniffing the bytes. Options as for URLRead. A redirect left unfollowed (FollowRedirects -> False) and "
          "connection failures give Failure[\"ConnectionFailure\", ...] (documented); an HTTP status >= 400 "
          "raises ValueError, an unrecognisable body raises from Import. Check: reference (Wolfram "
          "documentation) and local-server round trips in tests.")
def _urlexecute(req: Expr, *rest: Expr | Option) -> Expr:
    args, opts = split_opts(rest, _READ_OPTS, "URLExecute")
    if isinstance(req, Symbol):
        raise Unevaluated()
    if not _is_req(req):
        raise TypeError(f"URLExecute: expected a URL, an association or an HTTPRequest, not {_show(req)}")
    if len(args) > 2:
        raise TypeError("URLExecute: expected URLExecute[url, params, format]")
    params = args[0] if args else None
    fmt = args[1] if len(args) > 1 else None
    if fmt is not None and not isinstance(fmt, str):
        raise TypeError("URLExecute: the format must be a string")
    kw = {}
    if params is not None and not (is_list(params) and not params.args):
        pairs = _rules(params, "URLExecute", "the parameters")
        if _simple_params(pairs):
            kw["extra_query"] = _flat_pairs(pairs, "URLExecute")
        else:
            kw["body_override"] = params
            kw["method_override"] = "POST"
    t, fail = _read_one(req, opts, "URLExecute", **kw)
    if fail is not None:
        return fail
    if 300 <= t.status < 400:
        return _failure("URLExecute", t.url, f"URLExecute: the server answered with redirect status {t.status} "
                        f"and FollowRedirects is False", _request_expr(req))
    if t.status >= 400:
        raise ValueError(f"URLExecute: the server returned status {t.status} ({_status_text(t.status)}) for {t.url}")
    ct = None
    for k, v in t.headers:
        if k.lower() == "content-type":
            ct = v
    return _import_body(t.body, ct, fmt, "URLExecute")


_DL_ELEMS = ("File", "Headers", "Cookies", "ContentType", "StatusCode", "StatusCodeDescription")


def _dl_name(t):
    for k, v in t.headers:
        if k.lower() == "content-disposition" and "filename=" in v:
            n = v.split("filename=", 1)[1].split(";")[0].strip().strip('"')
            n = os.path.basename(n)
            if n:
                return n
    n = os.path.basename(urllib.parse.urlsplit(t.url).path)
    return n or "index.html"


def _unique(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{root}-{i}{ext}"):
        i += 1
    return f"{root}-{i}{ext}"


def _tmpdir():
    v = ev(Symbol("$TemporaryDirectory"))
    return v if isinstance(v, str) and os.path.isdir(v) else tempfile.gettempdir()


def _download_one(req, target, target_is_dir, opts, elem):
    t, fail = _read_one(req, opts, "URLDownload")
    if fail is not None:
        return fail
    if t.status >= 400:
        raise ValueError(f"URLDownload: the server returned status {t.status} ({_status_text(t.status)}) for {t.url}")
    if target is None:
        root, ext = os.path.splitext(_dl_name(t))
        path = os.path.join(_tmpdir(), f"{root}-{uuid.uuid4()}{ext}")
    elif target_is_dir:
        path = _unique(os.path.join(target, _dl_name(t)))
    else:
        path = target
    with open(path, "wb") as f:
        f.write(t.body)
    fobj = Expr("File", path)
    if elem is None:
        return fobj
    resp = _response_expr(t)

    def one(e):
        if e == "File":
            return fobj
        if not isinstance(e, str) or e not in _DL_ELEMS:
            raise ValueError(f"URLDownload: unknown element {_show(e)} (known: {', '.join(_DL_ELEMS)})")
        return _resp_props(resp, "URLDownload")[e]
    if is_list(elem):
        return _assoc([(e, one(e)) for e in elem.args])
    return one(elem)


@fn("URLDownload", area=AREA, status="added",
    notes="URLDownload[url] saves the body to a new file in $TemporaryDirectory and gives File[path]; "
          "URLDownload[url, file] writes file (an existing directory gets the server's or the URL's file name, "
          "with a -n suffix if taken); URLDownload[{url, ...}] / [{url, ...}, dir] download into a new temporary "
          "directory or dir and give a list; URLDownload[req, loc, elem] gives \"File\", \"Headers\", \"Cookies\", "
          "\"ContentType\", \"StatusCode\", \"StatusCodeDescription\" or a list of them. Requests and options as "
          "for URLRead. Connection failures give Failure[\"ConnectionFailure\", ...]; an HTTP status >= 400 raises "
          "ValueError and writes nothing. Check: reference (Wolfram documentation) and local-server downloads "
          "compared byte for byte in tests.")
def _urldownload(req: Expr, *rest: Expr | Option) -> Expr:
    args, opts = split_opts(rest, _READ_OPTS, "URLDownload")
    if isinstance(req, Symbol):
        raise Unevaluated()
    if len(args) > 2:
        raise TypeError("URLDownload: expected URLDownload[url, location, element]")
    loc = args[0] if args else None
    elem = args[1] if len(args) > 1 else None
    if loc is not None and not (isinstance(loc, str) or _file_path(loc) is not None):
        raise TypeError(f"URLDownload: the location must be a file name or File[...], not {_show(loc)}")
    locp = None if loc is None else (loc if isinstance(loc, str) else _file_path(loc))
    if is_list(req):
        for q in req.args:
            if not _is_req(q):
                raise TypeError(f"URLDownload: expected URLs or requests, not {_show(q)}")
        d = locp or tempfile.mkdtemp(dir=_tmpdir())
        if not os.path.isdir(d):
            raise ValueError(f"URLDownload: {d} is not a directory")
        return L(*(_download_one(q, d, True, opts, elem) for q in req.args))
    if not _is_req(req):
        raise TypeError(f"URLDownload: expected a URL, an association or an HTTPRequest, not {_show(req)}")
    return _download_one(req, locp, locp is not None and os.path.isdir(locp), opts, elem)


_RT_PROPS = ("NameLookup", "HostConnection", "SSLHandshake", "PreTransfer", "TransferInitiation", "HTTPRedirect",
             "TransactionTotal")


def _seconds(t):
    return Expr("Quantity", float(t), "Seconds")


@fn("URLResponseTime", area=AREA, status="added",
    notes="URLResponseTime[url] and URLResponseTime[url, prop] time one request for the first byte (GET with "
          "Range: bytes=0-0, no redirects) of url (a URL, URL[...] or bare host name, port 80 assumed) and give "
          "Quantity[t, \"Seconds\"], cumulative from the start like curl: \"NameLookup\", \"HostConnection\", "
          "\"SSLHandshake\" (0 for http), \"PreTransfer\", \"TransferInitiation\", \"TransactionTotal\" (the "
          "default), \"HTTPRedirect\" (0, redirects are not followed), \"ConnectionTimes\" or All for an "
          "association; option TimeConstraint (default 60). Deviation: an unreachable URL raises ValueError "
          "(Wolfram returns $Failed). The timings are verified to be non-decreasing and an ArithmeticError is "
          "raised otherwise. Check: self-check (monotone timing chain) and local-server timings bounded by "
          "AbsoluteTiming in tests.")
def _urlresponsetime(url: Expr, *rest: Expr | Option) -> Expr:
    args, opts = split_opts(rest, {"TimeConstraint"}, "URLResponseTime")
    if isinstance(url, Symbol):
        raise Unevaluated()
    if len(args) > 1:
        raise TypeError("URLResponseTime: expected URLResponseTime[url] or URLResponseTime[url, prop]")
    prop = args[0] if args else "TransactionTotal"
    if isinstance(url, Expr) and url.head == "IPAddress" and url.args and isinstance(url.args[0], str):
        u = "http://" + url.args[0]
    else:
        u = _url_str(url, "URLResponseTime")
    if "://" not in u:
        u = "http://" + u
    tc = opts.get("TimeConstraint", 60)
    if not (isinstance(tc, (int, float)) and not isinstance(tc, bool) and tc > 0):
        raise ValueError("URLResponseTime: TimeConstraint must be a positive number of seconds")
    p = urllib.parse.urlsplit(u)
    host = p.hostname
    if not host:
        raise ValueError(f"URLResponseTime: {u!r} has no host")
    try:
        check_allowed(host, "URLResponseTime")
    except Blocked as e:
        raise ValueError(str(e))
    t0 = time.perf_counter()
    try:
        socket.getaddrinfo(host, p.port or 80)
    except OSError as e:
        raise ValueError(f"URLResponseTime: cannot resolve {host}: {e}")
    t_dns = time.perf_counter() - t0
    r = Request(u, "GET", [("Range", "bytes=0-0")], b"", _sym("None"), _client_agent(), None)
    try:
        t = perform(r, {"TimeConstraint": tc, "CookieFunction": _sym("None")}, "URLResponseTime", follow=False)
    except (ConnectionFailed, Blocked) as e:
        raise ValueError(str(e))
    tm = t.times
    off = t_dns
    vals = {"NameLookup": t_dns, "HostConnection": off + tm["HostConnection"],
            "SSLHandshake": (off + tm["HostConnection"]) if p.scheme == "https" else 0.0,
            "PreTransfer": off + tm["PreTransfer"], "TransferInitiation": off + tm["TransferInitiation"],
            "HTTPRedirect": 0.0, "TransactionTotal": off + tm["TransactionTotal"]}
    chain = [vals["NameLookup"], vals["HostConnection"], vals["PreTransfer"], vals["TransferInitiation"],
             vals["TransactionTotal"]]
    if any(b < a for a, b in zip(chain, chain[1:])):
        raise ArithmeticError("URLResponseTime: result failed verification (timings are not cumulative)")
    conn = ("NameLookup", "HostConnection", "SSLHandshake", "PreTransfer")
    if is_sym(prop, "All"):
        return _assoc([(k, _seconds(vals[k])) for k in _RT_PROPS])
    if prop == "ConnectionTimes":
        return _assoc([(k, _seconds(vals[k])) for k in conn])
    if not isinstance(prop, str) or prop not in vals:
        raise ValueError(f"URLResponseTime: unknown property {_show(prop)} (known: {', '.join(_RT_PROPS)}, "
                         f"\"ConnectionTimes\", All)")
    return _seconds(vals[prop])


# ═══════════════════════════════════════ option symbols ═══════════════════════════════════════

for _n, _d in {
    "FollowRedirects": "option of URLRead, URLExecute, URLDownload and HTTPRequest (default True): True follows "
                       "301/302/303/307/308 redirects (at most 20; 303, and 301/302 after a non-GET, switch to "
                       "GET), False gives the redirect response itself (URLExecute then gives Failure). A "
                       "non-Boolean raises ValueError. Check: reference (Wolfram documentation) and redirects "
                       "from a local server in tests.",
    "ConnectionSettings": "option of URLRead and related functions: Automatic or <|\"MaxDownloadSpeed\" -> b, "
                          "\"MaxUploadSpeed\" -> b|> with b bytes per second (approximate throttling, as "
                          "documented) or Automatic. Quantity speeds raise NotImplementedError, unknown keys or "
                          "bad values ValueError. Check: reference (Wolfram documentation) and a throttled local "
                          "download timed in tests.",
    "CookieFunction": "option of URLRead, URLExecute, URLDownload and HTTPRequest: Automatic (= SetCookies, the "
                      "default) stores received cookies in $Cookies, None ignores them, f applies f to each cookie "
                      "association (keys Domain, Path, Name, Content, ExpirationDate, AllowSubdomains, "
                      "ConnectionType, ScriptAccessible). Check: reference (Wolfram documentation) and cookies "
                      "from a local server in tests.",
    "Authentication": "option of URLRead, URLExecute and URLDownload: Automatic or None (no credentials), or an "
                      "association with \"Username\"/\"Password\" (HTTP Basic), \"Headers\" and \"Query\" "
                      "entries added to the request; PermissionsKey, SecuredAuthenticationKey, SSH keys and other "
                      "cloud/SSH settings raise NotImplementedError. Credentials are never read from disk. "
                      "Check: reference (Wolfram documentation) and the Authorization header seen by a local "
                      "server in tests.",
}.items():
    record(_n, AREA, "added", _d)


# ═══════════════════════════════════════ hosts and addresses ═══════════════════════════════════════

def _ip_text(x, fname):
    if isinstance(x, Expr) and x.head == "IPAddress" and len(x.args) == 1 and isinstance(x.args[0], str):
        return x.args[0]
    return None


@fn("IPAddress", area=AREA, status="added",
    notes="IPAddress[\"address\"] is a validated symbolic IPv4 or IPv6 address (checked with Python's ipaddress "
          "module; IPv6 text is kept as given); used by HostLookup, SocketConnect, URLResponseTime and socket "
          "properties. A malformed address raises ValueError, a non-string TypeError, a symbol stays "
          "unevaluated. Check: exact (ipaddress validation) and HostLookup round trips in tests.")
def _ipaddress(a: String) -> Expr:
    s = _str(a, "IPAddress")
    try:
        ipaddress.ip_address(s)
    except ValueError:
        raise ValueError(f"IPAddress: {s!r} is not a valid IPv4 or IPv6 address")
    raise Unevaluated()


def _host_of(x, fname):
    ip = _ip_text(x, fname)
    if ip is not None:
        return ip
    s = _url_str(x, fname)
    if "://" in s:
        h = urllib.parse.urlsplit(s).hostname
        if not h:
            raise ValueError(f"{fname}: {s!r} has no host")
        return h
    return s.strip("[]")


_HL_PROPS = ("FullName", "IPv4Addresses", "IPv6Addresses")


@fn("HostLookup", area=AREA, status="added",
    notes="HostLookup[name] gives the IPv4 addresses of a host as IPAddress objects (IPv6 if it has no IPv4 "
          "address); HostLookup[address] (an IP string or IPAddress) gives the host name; HostLookup[spec, prop] "
          "gives \"FullName\", \"IPv4Addresses\", \"IPv6Addresses\" or All (an association); URLs are reduced to "
          "their host; lists thread. Resolution uses the system resolver (socket.getaddrinfo/gethostbyaddr), so "
          "localhost and literal addresses need no network. An unknown host gives Missing[\"NotFound\"] "
          "(documented); with $AllowInternet = False only loopback names are looked up (others raise ValueError). "
          "Check: reference (Wolfram documentation localhost examples) and name/address round trips in tests.")
def _hostlookup(spec: Expr, prop: Expr = None) -> Expr:
    if isinstance(spec, Symbol):
        raise Unevaluated()
    if is_list(spec):
        return L(*(_hostlookup(x, prop) if prop is not None else _hostlookup(x) for x in spec.args))
    if prop is not None and not (is_sym(prop, "All") or (isinstance(prop, str) and prop in _HL_PROPS)):
        raise ValueError(f"HostLookup: unknown property {_show(prop)} (known: {', '.join(_HL_PROPS)}, All)")
    host = _host_of(spec, "HostLookup")
    is_ip = True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        is_ip = False
    if not _loopback(host) and not internet_allowed() and not is_ip:
        raise ValueError("HostLookup: internet access is disabled ($AllowInternet is False); only loopback names "
                         "can be looked up")
    try:
        if is_ip:
            full = socket.gethostbyaddr(host)[0] if (internet_allowed() or _loopback(host)) else host
        else:
            full = socket.getfqdn(host) if host != "localhost" else "localhost"
        infos = socket.getaddrinfo(host, None)
    except (socket.herror, socket.gaierror, OSError, UnicodeError):
        return Expr("Missing", "NotFound")
    v4, v6 = [], []
    for fam, _t, _p, _c, sa in infos:
        a = sa[0]
        if fam == socket.AF_INET and a not in v4:
            v4.append(a)
        elif fam == socket.AF_INET6 and a not in v6:
            v6.append(a.split("%")[0])
    v6 = list(dict.fromkeys(v6))
    props = {"FullName": full, "IPv4Addresses": L(*(Expr("IPAddress", a) for a in v4)),
             "IPv6Addresses": L(*(Expr("IPAddress", a) for a in v6))}
    if prop is None:
        if is_ip:
            return full
        return props["IPv4Addresses"] if v4 else props["IPv6Addresses"]
    if is_sym(prop, "All"):
        return _assoc([(k, props[k]) for k in _HL_PROPS])
    return props[prop]


def _linux_iface_addrs():
    import fcntl
    import struct
    out = []
    for _i, name in socket.if_nameindex():
        if name == "lo":
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            res = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))   # SIOCGIFADDR
            out.append(socket.inet_ntoa(res[20:24]))
        except OSError:
            pass
        finally:
            s.close()
    return out


def _machine_addresses():
    try:
        addrs = _linux_iface_addrs()
    except (ImportError, OSError, AttributeError):
        addrs = []
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 9))           # TEST-NET-1: no packet is sent by a UDP connect
            a = s.getsockname()[0]
            if not a.startswith("127."):
                addrs.append(a)
        except OSError:
            pass
        finally:
            s.close()
    return L(*addrs)


_sysvar("$MachineAddresses", _machine_addresses)
record("$MachineAddresses", AREA, "added",
       "$MachineAddresses lists the IPv4 addresses of this computer's non-loopback network interfaces as strings "
       "({} when offline), read from the interfaces (Linux SIOCGIFADDR) or, elsewhere, the source address of the "
       "default route (a UDP connect sends no packet). Check: exact (every entry is a valid non-loopback IPv4 "
       "address that a local socket can bind to) in tests.")


def _default_iface():
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000":
                    return parts[0]
        return NONE
    except OSError:
        raise NotImplementedError("$DefaultNetworkInterface: only available on Linux (/proc/net/route)")


_sysvar("$DefaultNetworkInterface", _default_iface)
record("$DefaultNetworkInterface", AREA, "added",
       "$DefaultNetworkInterface is the name of the interface carrying the default route (e.g. \"eth0\"), read "
       "from /proc/net/route, or None when there is no default route; on systems without /proc/net/route it "
       "raises NotImplementedError. Check: exact (the name is one of socket.if_nameindex()) in tests.")


# ═══════════════════════════════════════ sockets ═══════════════════════════════════════

import threading
import select
import contextvars

SOCKETS = {}               # uuid -> _Sock
LISTENERS = {}             # id -> _Listener
_LOCK = threading.RLock()


class _Sock:
    def __init__(self, sock, kind, server=None):
        self.sock, self.kind, self.server = sock, kind, server     # kind: "Client" | "Server" | "Remote"
        self.uuid = f"TCP-{uuid.uuid4()}"
        self.clients = []
        self.closed = False
        self.local = sock.getsockname()
        try:
            self.peer = sock.getpeername() if kind != "Server" else self.local
        except OSError:
            self.peer = self.local
        self.listener = None
        SOCKETS[self.uuid] = self

    def expr(self):
        return Expr("SocketObject", self.uuid)

    def accept_pending(self):
        """Accept queued connections of a server socket (non-blocking)."""
        if self.kind != "Server" or self.closed:
            return
        while True:
            r, _, _ = select.select([self.sock], [], [], 0)
            if not r:
                return
            try:
                c, _addr = self.sock.accept()
            except OSError:
                return
            self.clients.append(_Sock(c, "Remote", self))

    def readable(self):
        """The underlying sockets this object reads from."""
        if self.kind == "Server":
            self.accept_pending()
            return [c.sock for c in self.clients if not c.closed]
        return [] if self.closed else [self.sock]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for c in self.clients:
            c.close()
        try:
            self.sock.close()
        finally:
            SOCKETS.pop(self.uuid, None)


def close_socket(x):
    """Close a SocketObject (for the stream functions Close/DeleteObject); gives "host:port"."""
    s = sock_obj(x, "Close")
    host, port = s.peer[0], s.peer[1]
    s.close()
    return f"{host}:{port}"


def write_socket(x, data):
    """Write bytes to a SocketObject (for WriteString/BinaryWrite); a server writes to every connected client."""
    s = sock_obj(x, "SocketWriteMessage")
    targets = [c for c in s.clients if not c.closed] if s.kind == "Server" else [s]
    if s.kind == "Server":
        s.accept_pending()
        targets = [c for c in s.clients if not c.closed]
    for t in targets:
        t.sock.sendall(data)


def sock_obj(x, fname):
    if isinstance(x, Expr) and x.head == "SocketObject" and len(x.args) == 1 and isinstance(x.args[0], str):
        s = SOCKETS.get(x.args[0])
        if s is None:
            raise ValueError(f"{fname}: {_show(x)} is not an open socket")
        return s
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a SocketObject, not {_show(x)}")


def _protocol(p, fname):
    if p is None or p == "TCP":
        return "TCP"
    if p in ("ZMQ", "WS", "WSS") or (is_list(p) and p.args and p.args[0] == "ZMQ"):
        raise NotImplementedError(f"{fname}: the {_show(p)} protocol is not supported (only raw \"TCP\"; ZeroMQ "
                                  f"and WebSocket need libraries MikoshiLang does not bundle)")
    raise ValueError(f"{fname}: unknown protocol {_show(p)} (\"TCP\", \"ZMQ\", \"WS\", \"WSS\")")


def _port(p, fname):
    if K.is_int(p) and not isinstance(p, bool):
        if not 0 <= int(p) <= 65535:
            raise ValueError(f"{fname}: port {p} is out of range")
        return int(p)
    if isinstance(p, str):
        if p.isdigit():
            return _port(int(p), fname)
        try:
            return socket.getservbyname(p)
        except OSError:
            raise ValueError(f"{fname}: unknown service name {p!r}")
    if is_sym(p, "Automatic"):
        return 0
    if isinstance(p, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: a port must be an integer or a service name, not {_show(p)}")


def _address(a, fname, local):
    """(host, port) of a socket address specification."""
    if is_sym(a, "Automatic") and local:
        return "127.0.0.1", 0
    if K.is_int(a) and not isinstance(a, bool):
        return "127.0.0.1", _port(a, fname)
    if is_list(a) and len(a.args) == 2:
        h = a.args[0]
        host = _ip_text(h, fname) or (h if isinstance(h, str) else None)
        if host is None:
            if isinstance(h, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: a host must be a string or IPAddress, not {_show(h)}")
        return host, _port(a.args[1], fname)
    ip = _ip_text(a, fname)
    if ip is not None:
        raise ValueError(f"{fname}: an IPAddress needs a port: give {{IPAddress[...], port}}")
    if isinstance(a, Expr) and a.head == "URL":
        a = _url_str(a, fname)
    if isinstance(a, str):
        if "://" in a:
            u = urllib.parse.urlsplit(a)
            port = u.port or _port(u.scheme, fname)
            return u.hostname, port
        if a.startswith("["):
            h, _, rest = a[1:].partition("]")
            return h, _port(rest.lstrip(":"), fname)
        if ":" in a:
            h, p = a.rsplit(":", 1)
            return h.strip(), _port(p.strip(), fname)
        raise ValueError(f"{fname}: {a!r} needs a port (\"host:port\" or {{host, port}})")
    if isinstance(a, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: unsupported socket address {_show(a)}")


def _family(host):
    try:
        return socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET
    except ValueError:
        return socket.AF_INET


@fn("SocketConnect", area=AREA, status="added",
    notes="SocketConnect[addr] and SocketConnect[addr, \"TCP\"] open a TCP connection and give SocketObject[...]; "
          "addr is {host, port}, {host, \"service\"}, \"host:port\", a local port, URL[\"url\"] or \"scheme://host\" "
          "(port from the scheme's service), {IPAddress[...], port}, or a SocketObject from SocketOpen. \"ZMQ\", "
          "{\"ZMQ\", type}, \"WS\" and \"WSS\" raise NotImplementedError. A refused or unreachable address, or a "
          "non-loopback host with $AllowInternet = False, raises ValueError (Wolfram returns $Failed). Reading "
          "uses SocketReadMessage; writing uses SocketWriteMessage (Close/WriteString/BinaryWrite on sockets are "
          "wired to the W27 stream functions at integration through wl_external.close_socket/write_socket). "
          "Check: reference (Wolfram documentation) and echo round trips with a local TCP server in tests.")
def _socketconnect(addr: Expr, proto: Expr = None) -> Expr:
    _protocol(proto, "SocketConnect")
    if isinstance(addr, Expr) and addr.head == "SocketObject":
        srv = sock_obj(addr, "SocketConnect")
        if srv.kind != "Server":
            raise ValueError("SocketConnect: only a SocketObject opened with SocketOpen can be connected to")
        host, port = srv.local[0], srv.local[1]
    else:
        host, port = _address(addr, "SocketConnect", False)
    if port == 0:
        raise ValueError("SocketConnect: a destination port is required")
    try:
        check_allowed(host, "SocketConnect")
    except Blocked as e:
        raise ValueError(str(e))
    try:
        s = socket.create_connection((host, port), timeout=30)
        s.settimeout(None)
    except OSError as e:
        raise ValueError(f"SocketConnect: cannot connect to {host}:{port}: {e}")
    with _LOCK:
        obj = _Sock(s, "Client")
        if not isinstance(addr, Expr) or addr.head != "SocketObject":
            return obj.expr()
        srv.accept_pending()
        return obj.expr()


@fn("SocketOpen", area=AREA, status="added",
    notes="SocketOpen[port], SocketOpen[{host, port}], SocketOpen[\"host:port\"], SocketOpen[{host, Automatic}] "
          "and SocketOpen[Automatic] (an available port on 127.0.0.1), with optional \"TCP\", open a listening "
          "TCP socket on this machine and give SocketObject[...]; incoming connections become socket[\"ConnectedClients\"]. "
          "Only local addresses can be opened (a non-local address raises ValueError, as the OS refuses it); "
          "\"ZMQ\" protocols raise NotImplementedError; a port in use raises ValueError. Check: reference "
          "(Wolfram documentation) and SocketConnect round trips in tests.")
def _socketopen(addr: Expr, proto: Expr = None) -> Expr:
    _protocol(proto, "SocketOpen")
    host, port = _address(addr, "SocketOpen", True)
    if is_list(addr) and len(addr.args) == 2 and is_sym(addr.args[1], "Automatic"):
        port = 0
    try:
        s = socket.socket(_family(host), socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(16)
    except OSError as e:
        try:
            s.close()
        except Exception:
            pass
        raise ValueError(f"SocketOpen: cannot open {host}:{port}: {e}")
    with _LOCK:
        return _Sock(s, "Server").expr()


def _sock_props(s):
    sh, sp_ = s.local[0], s.local[1]
    dh, dp = s.peer[0], s.peer[1]
    if s.kind == "Remote":
        sh, sp_, dh, dp = dh, dp, sh, sp_
    props = {"SourceIPAddress": Expr("IPAddress", sh), "SourcePort": sp_,
             "DestinationIPAddress": Expr("IPAddress", dh), "DestinationPort": dp, "Protocol": "TCP",
             "UUID": s.uuid.split("-", 1)[1], "Type": s.kind}
    if s.kind == "Server":
        s.accept_pending()
        props["ConnectedClients"] = L(*(c.expr() for c in s.clients if not c.closed))
    return props


def _sock_hook(head, args, ev_):
    if not (len(head.args) == 1 and isinstance(head.args[0], str)) or len(args) != 1:
        return K.UNHANDLED
    s = sock_obj(head, "SocketObject")
    props = _sock_props(s)
    a = args[0]
    if a == "Properties":
        return L(*props)
    if not isinstance(a, str) or a not in props:
        raise ValueError(f"SocketObject: unknown property {_show(a)} (known: {', '.join(props)})")
    return props[a]


KL.COMPOUND_HEAD_HOOKS["SocketObject"] = _sock_hook


@fn("SocketObject", area=AREA, status="added",
    notes="SocketObject[\"TCP-uuid\"] represents an open TCP socket made by SocketConnect, SocketOpen or accepted "
          "by an open server; socket[\"prop\"] gives \"SourceIPAddress\", \"SourcePort\", \"DestinationIPAddress\", "
          "\"DestinationPort\", \"Protocol\", \"UUID\", \"Type\" (\"Client\", \"Server\" or \"Remote\"), "
          "\"ConnectedClients\" (servers) and \"Properties\". A SocketObject that is not open raises ValueError "
          "when used; constructing one by hand with an unknown id raises ValueError. Check: exact (properties read "
          "from the OS socket) and cross-checked between the two ends of a local connection in tests.")
def _socketobject(x: String) -> Expr:
    if not isinstance(x, str):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"SocketObject: expected a socket id string, not {_show(x)}")
    if x not in SOCKETS:
        raise ValueError(f"SocketObject: {x!r} is not an open socket")
    raise Unevaluated()


def _wait(socks, timeout):
    if not socks:
        if timeout:
            time.sleep(min(timeout, 0.05))
        return []
    r, _, _ = select.select(socks, [], [], timeout)
    return r


def _read_any(s, fname, block=True):
    """Bytes currently available on a socket object (a server reads its first ready client)."""
    deadline = None
    while True:
        socks = s.readable()
        r = _wait(socks, 0.05 if s.kind == "Server" or not block else None)
        if r:
            data = r[0].recv(65536)
            return data
        if not block:
            return None
        if s.closed:
            raise ValueError(f"{fname}: the socket is closed")
        if deadline is not None and time.time() > deadline:
            return None


@fn("SocketReadMessage", area=AREA, status="added",
    notes="SocketReadMessage[socket] waits for data on a TCP SocketObject and gives the bytes currently in the "
          "operating-system buffer as a ByteArray (an empty ByteArray when the peer has closed the connection); "
          "on a server socket it reads the first connected client with data. The \"MultipartComplete\" option "
          "(web sockets only) is accepted and has no effect on TCP. A socket that is not open raises ValueError. "
          "Check: reference (Wolfram documentation) and echo round trips with a local TCP server in tests.")
def _socketreadmessage(s: Expr, *opts: Option) -> Expr:
    _, o = split_opts(opts, {"MultipartComplete"}, "SocketReadMessage")
    if _:
        raise TypeError("SocketReadMessage: expected SocketReadMessage[socket]")
    so = sock_obj(s, "SocketReadMessage")
    if so.listener is not None:
        raise ValueError("SocketReadMessage: the socket is being read by a SocketListener")
    return byte_array(_read_any(so, "SocketReadMessage"))


@fn("SocketWriteMessage", area=AREA, status="added",
    notes="Partial. SocketWriteMessage[socket, ba] and SocketWriteMessage[socket, \"string\"] (UTF-8) write the bytes "
          "to a TCP SocketObject (a server writes to every connected client) and return Null. Deviation: "
          "Wolfram documents this head for web sockets only, which are not supported (\"WS\"/\"WSS\" sockets cannot "
          "be created); the <|\"DataByteArray\", \"Type\", \"MultipartComplete\"|> form raises NotImplementedError. "
          "A closed socket raises ValueError, other data TypeError. Check: reference (the bytes arrive unchanged "
          "at a local echo server) in tests.")
def _socketwritemessage(s: Expr, data: Expr) -> Expr:
    so = sock_obj(s, "SocketWriteMessage")
    if isinstance(data, str):
        b = data.encode("utf-8")
    elif is_byte_array(data):
        b = ba_bytes(data, "SocketWriteMessage")
    elif _is_assoc(data):
        raise NotImplementedError("SocketWriteMessage: web-socket message associations are not supported")
    elif isinstance(data, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"SocketWriteMessage: expected a ByteArray or a string, not {_show(data)}")
    try:
        write_socket(so.expr(), b)
    except OSError as e:
        raise ValueError(f"SocketWriteMessage: cannot write to the socket: {e}")
    return None


def _seconds_arg(t, fname):
    if isinstance(t, (int, float)) and not isinstance(t, bool) and t >= 0:
        return float(t)
    if isinstance(t, Expr) and t.head == "Quantity" and len(t.args) == 2 and t.args[1] in ("Seconds", "Second") \
            and isinstance(t.args[0], (int, float)):
        return float(t.args[0])
    if isinstance(t, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: the time must be a non-negative number of seconds, not {_show(t)}")


@fn("SocketReadyQ", area=AREA, status="added",
    notes="SocketReadyQ[socket] gives True if data can be read from the TCP SocketObject immediately (for a "
          "server: from any connected client), without blocking; SocketReadyQ[socket, t] waits up to t seconds "
          "(a number or Quantity[t, \"Seconds\"]). A socket that is not open raises ValueError, a negative time "
          "ValueError. Check: reference (Wolfram documentation) and before/after-write cases with a local server "
          "in tests.")
def _socketreadyq(s: Expr, t: Expr = 0) -> Boolean:
    so = sock_obj(s, "SocketReadyQ")
    wait = _seconds_arg(t, "SocketReadyQ")
    deadline = time.time() + wait
    while True:
        socks = so.readable()
        if _wait(socks, 0 if not socks else min(0.05, max(0.0, deadline - time.time()))):
            return True
        if time.time() >= deadline:
            return False
        if not socks:
            time.sleep(0.01)


@fn("SocketWaitAll", area=AREA, status="added",
    notes="SocketWaitAll[{s1, s2, ...}] waits until data is ready to read on every TCP SocketObject and gives "
          "{s1, s2, ...}; an empty list is returned at once. Sockets that are not open raise ValueError, "
          "non-lists TypeError. Check: reference (Wolfram documentation) and SocketReadyQ on the result in tests.")
def _socketwaitall(ss: ListOf[Expr]) -> ListOf[Expr]:
    if not is_list(ss):
        if isinstance(ss, Symbol):
            raise Unevaluated()
        raise TypeError("SocketWaitAll: expected a list of SocketObjects")
    objs = [sock_obj(x, "SocketWaitAll") for x in ss.args]
    pending = list(objs)
    while pending:
        pending = [o for o in pending if not _wait(o.readable(), 0.05)]
        for o in pending:
            if o.closed:
                raise ValueError("SocketWaitAll: a socket was closed while waiting")
    return ss


@fn("SocketWaitNext", area=AREA, status="added",
    notes="SocketWaitNext[{s1, s2, ...}] waits until data is ready on any of the TCP SocketObjects and gives that "
          "socket (the first ready one in list order). An empty list or sockets that are not open raise "
          "ValueError, non-lists TypeError. Check: reference (Wolfram documentation) and the socket that received "
          "data in tests.")
def _socketwaitnext(ss: ListOf[Expr]) -> Expr:
    if not is_list(ss):
        if isinstance(ss, Symbol):
            raise Unevaluated()
        raise TypeError("SocketWaitNext: expected a list of SocketObjects")
    if not ss.args:
        raise ValueError("SocketWaitNext: the list of sockets is empty")
    objs = [sock_obj(x, "SocketWaitNext") for x in ss.args]
    while True:
        for x, o in zip(ss.args, objs):
            if _wait(o.readable(), 0):
                return x
        time.sleep(0.01)


_SOCK_SPECS = ("Remote", "Local", "Server", "Client", "TCP", "ZMQ", "WSS", "WS")


@fn("Sockets", area=AREA, status="added",
    notes="Sockets[] gives the open SocketObjects of the session (servers, clients and accepted remote ends); "
          "Sockets[spec] filters by \"Local\" (opened here: servers and clients), \"Remote\" (accepted "
          "connections), \"Server\", \"Client\", a protocol (\"TCP\"; \"ZMQ\", \"WS\", \"WSS\" give {}) or All. "
          "An unknown spec raises ValueError. Check: exact (registry contents) and before/after open cases in "
          "tests.")
def _sockets(spec: Expr = None) -> ListOf[Expr]:
    for s in list(SOCKETS.values()):
        s.accept_pending()
    socks = [s for s in SOCKETS.values() if not s.closed]
    if spec is None or is_sym(spec, "All"):
        return L(*(s.expr() for s in socks))
    if is_list(spec) and spec.args and spec.args[0] == "ZMQ":
        return L()
    if not isinstance(spec, str) or spec not in _SOCK_SPECS:
        raise ValueError(f"Sockets: unknown specification {_show(spec)} (known: {', '.join(_SOCK_SPECS)}, All)")
    sel = {"Remote": lambda s: s.kind == "Remote", "Local": lambda s: s.kind != "Remote",
           "Server": lambda s: s.kind == "Server", "Client": lambda s: s.kind == "Client",
           "TCP": lambda s: True}.get(spec, lambda s: False)
    return L(*(s.expr() for s in socks if sel(s)))


# ───────────────────────────── listeners ─────────────────────────────

_LISTEN_OPTS = {"CharacterEncoding", "HandlerFunctionsKeys", "HandlerFunctions", "RecordSeparators"}
_EVENT_KEYS = ("Timestamp", "Socket", "SourceSocket", "Data", "DataBytes", "DataByteArray", "MultipartComplete")
_DEFAULT_KEYS = ("Timestamp", "Socket", "SourceSocket", "Data")


class _Listener:
    def __init__(self, sock, fun, keys, codec, seps, owns):
        self.sock, self.fun, self.keys, self.codec, self.seps, self.owns = sock, fun, keys, codec, seps, owns
        self.id = f"TCP-{uuid.uuid4()}"
        self.errors = []
        self.events = 0
        self.stop = threading.Event()
        self.ctx = contextvars.copy_context()
        self.buffers = {}
        sock.listener = self
        LISTENERS[self.id] = self
        self.thread = threading.Thread(target=self.ctx.run, args=(self.loop,), daemon=True)
        self.thread.start()

    def expr(self):
        return Expr("SocketListener", self.id)

    def loop(self):
        while not self.stop.is_set() and not self.sock.closed:
            try:
                with _LOCK:
                    socks = self.sock.readable()
                    owners = {id(c.sock): c for c in self.sock.clients} if self.sock.kind == "Server" else \
                        {id(self.sock.sock): self.sock}
                r = _wait(socks, 0.05)
            except (OSError, ValueError):
                return
            for rs in r:
                try:
                    data = rs.recv(65536)
                except OSError:
                    data = b""
                src = owners.get(id(rs))
                if not data:
                    if src is not None and src.kind == "Remote":
                        src.close()
                    elif src is self.sock:
                        return
                    continue
                for chunk in self.split(src, data):
                    self.dispatch(src, chunk)

    def split(self, src, data):
        if not self.seps:
            return [data]
        buf = self.buffers.get(id(src), b"") + data
        out = []
        while True:
            hits = [(buf.find(s), s) for s in self.seps if buf.find(s) >= 0]
            if not hits:
                break
            i, s = min(hits)
            out.append(buf[:i])
            buf = buf[i + len(s):]
        self.buffers[id(src)] = buf
        return out

    def dispatch(self, src, chunk):
        vals = {"Timestamp": _date_from_unix(time.time()), "Socket": self.sock.expr(),
                "SourceSocket": src.expr() if src is not None else self.sock.expr(),
                "DataBytes": L(*chunk), "DataByteArray": byte_array(chunk), "MultipartComplete": True}
        try:
            vals["Data"] = chunk.decode(self.codec)
        except UnicodeDecodeError:
            vals["Data"] = chunk.decode("latin-1")
        event = _assoc([(k, vals[k]) for k in self.keys])
        with _LOCK:
            self.events += 1
            if self.fun is None:
                return
            try:
                apply(self.fun, event)
            except Exception as e:           # recorded, readable as listener["HandlerErrors"]
                self.errors.append(f"{type(e).__name__}: {e}")

    def delete(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sock.listener = None
        if self.owns:
            self.sock.close()
        LISTENERS.pop(self.id, None)


def _handler_fun(fun, opts, fname):
    h = opts.get("HandlerFunctions")
    if h is not None and not is_sym(h, "Automatic"):
        if not _is_assoc(h):
            raise ValueError(f"{fname}: HandlerFunctions must be an association")
        d = {_key(k): v for k, v in _assoc_items(h)}
        bad = set(d) - {"DataReceived"}
        if bad:
            raise ValueError(f"{fname}: unknown event(s) {sorted(bad)} (only \"DataReceived\")")
        if "DataReceived" in d:
            return d["DataReceived"]
    return fun


@fn("SocketListen", area=AREA, status="added",
    notes="SocketListen[socket, fun], SocketListen[port, fun], SocketListen[\"host:port\", fun] and "
          "SocketListen[Automatic] start a background thread that applies fun to an event association for every "
          "chunk of data received on a TCP socket (for a server, on each connected client), and give a "
          "SocketListener; keys \"Timestamp\", \"Socket\", \"SourceSocket\", \"Data\" by default, also "
          "\"DataBytes\", \"DataByteArray\", \"MultipartComplete\" via HandlerFunctionsKeys; options "
          "CharacterEncoding (UTF-8), HandlerFunctions (<|\"DataReceived\" -> f|>), RecordSeparators (None or "
          "strings: fun gets each delimited record). fun runs in the session that started the listener, one event "
          "at a time; its errors are collected in listener[\"HandlerErrors\"]. A socket made by SocketListen is "
          "closed with the listener. Unknown keys/options raise ValueError; ZMQ/web sockets NotImplementedError. "
          "Check: reference (Wolfram documentation) and events from a local client in tests.")
def _socketlisten(spec: Expr, *rest: Expr | Option) -> Expr:
    args, opts = split_opts(rest, _LISTEN_OPTS, "SocketListen")
    if len(args) > 1:
        raise TypeError("SocketListen: expected SocketListen[socket, fun]")
    fun = _handler_fun(args[0] if args else None, opts, "SocketListen")
    keys = opts.get("HandlerFunctionsKeys", _sym("Default"))
    if is_sym(keys, "Default") or is_sym(keys, "Automatic"):
        kl = list(_DEFAULT_KEYS)
    elif is_list(keys) and all(isinstance(k, str) for k in keys.args):
        kl = list(keys.args)
        bad = [k for k in kl if k not in _EVENT_KEYS]
        if bad:
            raise ValueError(f"SocketListen: unknown HandlerFunctionsKeys {bad} (known: {', '.join(_EVENT_KEYS)})")
    else:
        raise ValueError("SocketListen: HandlerFunctionsKeys must be a list of strings or Default")
    enc = opts.get("CharacterEncoding")
    codec = "utf-8" if enc is None or is_sym(enc, "Automatic") or is_sym(enc, "$CharacterEncoding") else \
        _codec(enc, "SocketListen")
    seps = opts.get("RecordSeparators", NONE)
    if is_sym(seps, "None") or is_sym(seps, "Automatic"):
        sl = []
    elif isinstance(seps, str):
        sl = [seps.encode(codec)]
    elif is_list(seps) and all(isinstance(x, str) and x for x in seps.args):
        sl = [x.encode(codec) for x in seps.args]
    else:
        raise ValueError("SocketListen: RecordSeparators must be None, a string or a list of strings")
    if isinstance(spec, Expr) and spec.head == "SocketObject":
        so, owns = sock_obj(spec, "SocketListen"), False
    else:
        so, owns = sock_obj(_socketopen(spec), "SocketListen"), True
    if so.listener is not None:
        raise ValueError("SocketListen: the socket already has a listener")
    return _Listener(so, fun, kl, codec, sl, owns).expr()


def _listener_obj(x, fname):
    if isinstance(x, Expr) and x.head == "SocketListener" and len(x.args) == 1 and isinstance(x.args[0], str):
        lst = LISTENERS.get(x.args[0])
        if lst is None:
            raise ValueError(f"{fname}: {_show(x)} is not an active listener")
        return lst
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a SocketListener, not {_show(x)}")


def delete_listener(x):
    """Stop a SocketListener (for DeleteObject)."""
    _listener_obj(x, "DeleteObject").delete()
    return None


def _listener_hook(head, args, ev_):
    if not (len(head.args) == 1 and isinstance(head.args[0], str)) or len(args) != 1:
        return K.UNHANDLED
    lst = _listener_obj(head, "SocketListener")
    props = {"Socket": lst.sock.expr(), "HandlerFunctionsKeys": L(*lst.keys),
             "HandlerFunctions": _assoc([("DataReceived", lst.fun)]) if lst.fun is not None else Expr("Association"),
             "RecordSeparators": L(*(s.decode(lst.codec) for s in lst.seps)) if lst.seps else NONE,
             "CharacterEncoding": "UTF-8" if lst.codec == "utf-8" else lst.codec,
             "EventCount": lst.events, "HandlerErrors": L(*lst.errors)}
    a = args[0]
    if a == "Properties":
        return L(*props)
    if not isinstance(a, str) or a not in props:
        raise ValueError(f"SocketListener: unknown property {_show(a)} (known: {', '.join(props)})")
    return props[a]


KL.COMPOUND_HEAD_HOOKS["SocketListener"] = _listener_hook


@fn("SocketListener", area=AREA, status="added",
    notes="SocketListener[\"TCP-uuid\"] represents an active listener made by SocketListen; listener[\"prop\"] gives "
          "\"Socket\", \"HandlerFunctions\", \"HandlerFunctionsKeys\", \"RecordSeparators\", "
          "\"CharacterEncoding\", \"EventCount\" (events seen) and \"HandlerErrors\" (messages of handler calls "
          "that raised), and \"Properties\". DeleteObject stops it (wired at integration through "
          "wl_external.delete_listener). An id that is not active raises ValueError. Check: exact (listener "
          "state) and event counts from a local client in tests.")
def _socketlistener(x: String) -> Expr:
    if not isinstance(x, str):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError("SocketListener: expected a listener id string")
    if x not in LISTENERS:
        raise ValueError(f"SocketListener: {x!r} is not an active listener")
    raise Unevaluated()
