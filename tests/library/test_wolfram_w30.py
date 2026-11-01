"""Differential tests for Wolfram batch W30 (external interfaces: URL/HTTP, cookies, hosts, TCP sockets,
compilation and foreign functions; modules wl_external, wl_external_compile, wl_external_ffi).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): the documented examples and
  defaults, e.g. ``URLEncode["Hello, World!"] == "Hello%2C%20World%21"``, the URLQueryEncode/URLQueryDecode,
  URLParse and URLBuild examples, ``$DefaultProxyRules``, ``HTTPRequest[url]`` shown as ``HTTPRequest[url, <||>]``,
  HTTPRedirect's default status 302, ``Compile[{{x}}, x^2 + Sin[x^2], CompilationTarget -> "C"][10] == 99.4936``,
  ``FunctionCompile`` examples (AddTwo, squareFun, polyFunc, VarArgsAdd, addRank, fooInteger, the
  UnsignedInteger8 overflow with CompilerRuntimeErrorAction and the documented Failure["ArgumentCount"],
  Failure["ArgumentType"] and Failure["RuntimeError"] objects), ``TypeOf`` examples, ``IfCompiled``,
  ``$TargetSystems``, the RawMemory* examples (``RawMemoryExport[-1, "CInt"]`` reads back -1, writing 65 into
  "a string" gives "A string", the bytes {65, 32, 115, ...} of "A string"), NullRawPointerQ, ManagedObject and
  UnmanageObject examples.
* Python standard library behaviour as the independent reference: urllib.parse quoting (RFC 3986 unreserved
  characters), http.HTTPStatus reason phrases (RFC 9110, "I'm a Teapot" for 418), base64 for HTTP Basic
  authentication (RFC 7617: w30user:w30pass -> dzMwdXNlcjp3MzBwYXNz), ipaddress validation, and the C standard
  library through ctypes (abs, labs, toupper, strlen, atoi, strtod, strchr, qsort).
* Numerical references: math.sin etc. for compiled bodies (Sin[2] = 0.9092974268256817, Sum 1/k! for k <= 20).

Safety: every network case talks only to servers started by the module fixture on 127.0.0.1 (``_w30_servers``) and
stopped afterwards; ``{P}`` (HTTP), ``{E}`` (TCP echo) and ``{F}`` (a port with no listener) are substituted at run
time so test IDs do not contain ports. Non-loopback names used for $AllowInternet/proxy checks are never resolved
(blocked before lookup, or sent to the local proxy). No mail is sent, no credentials are read, and foreign
functions use only the C standard library. Symbols assigned in cases are prefixed ``w30``.
"""
import tempfile

import pytest

from _harness import Pred, Raises, check, ids
from _w30_servers import EchoServer, HTTPServer, free_port
from mikoshilang.expr import Expr


def _failure_tag(r):
    return r.args[0] if isinstance(r, Expr) and r.head == "Failure" and r.args else None


def _ev(src):
    from _harness import ev
    return ev(src)


def _file_name(r):
    import os
    return os.path.basename(r.args[0]) if isinstance(r, Expr) and r.head == "File" else ""


def _file_bytes(r):
    """Bytes of a downloaded File[...] (the file is removed afterwards)."""
    import os
    if not (isinstance(r, Expr) and r.head == "File"):
        return None
    with open(r.args[0], "rb") as f:
        data = f.read()
    os.remove(r.args[0])
    return data


def _cleanup_tmp(name):
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), name)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _failure_msg(r):
    if _failure_tag(r) is None:
        return ""
    for rule in r.args[1].args:
        if rule.args[0] == "Message":
            return rule.args[1]
    return ""


# ---------------------------------------------------------------- URL encoding (Wolfram reference examples; RFC 3986)
NET = [
    ('URLEncode["Hello, World!"]', '"Hello%2C%20World%21"'),
    ('URLEncode[{"Kurt Gödel", "Paul Erdős"}]', '{"Kurt%20G%C3%B6del", "Paul%20Erd%C5%91s"}'),
    ('URLEncode[None]', '""'),
    ('URLEncode[Missing["KeyAbsent", "MissingKey"]]', '""'),
    ('URLEncode[1]', '"1"'),
    ('URLEncode[1.3]', '"1.3"'),
    ('URLEncode[URL["http://www.wolfram.com"]]', '"http%3A%2F%2Fwww.wolfram.com"'),
    ('URLEncode["Eötvös Loránd University", CharacterEncoding -> "MacintoshRoman"]', '"E%9Atv%9As%20Lor%87nd%20University"'),
    ('URLEncode[w30sym]', 'URLEncode[w30sym]'),
    ('URLEncode[CloudObject["x"]]', Raises(NotImplementedError)),
    ('URLEncode["a", CharacterEncoding -> "NoSuchEncoding"]', Raises(ValueError)),
    ('URLEncode["-_.~ az09"]', '"-_.~%20az09"'),
    ('URLDecode["Hello%2C+World%21"]', '"Hello, World!"'),
    ('URLDecode[{"Kurt+G%C3%B6del", "Paul+Erd%C5%91s"}]', '{"Kurt Gödel", "Paul Erdős"}'),
    ('URLDecode["E%9Atv%9As+Lor%87nd+University", CharacterEncoding -> "MacintoshRoman"]', '"Eötvös Loránd University"'),
    ('URLDecode[URLEncode["a b/c?d=é&"]]', '"a b/c?d=é&"'),
    ('URLDecode[w30sym]', 'URLDecode[w30sym]'),
    ('URLDecode["%FF%FE"]', Raises(ValueError)),
    ('URLDecode[12]', Raises(TypeError)),
    ('URLQueryEncode[{"name" -> "John Doe", "title" -> "Reg. Manager: Northwest"}]', '"name=John+Doe&title=Reg.+Manager%3A+Northwest"'),
    ('URLQueryEncode[<|"user" -> Missing[], "number" -> 3, "other" -> None|>]', '"user&number=3&other"'),
    ('URLQueryEncode[<|"number" -> {1, 2, 3}, "name" -> {"jack", "john", "bob"}|>]', '"number=1&number=2&number=3&name=jack&name=john&name=bob"'),
    ('URLQueryEncode[{"url" -> URL["http://www.wolfram.com"]}]', '"url=http%3A%2F%2Fwww.wolfram.com"'),
    ('URLQueryEncode[{"university" -> "Eötvös Loránd University"}, CharacterEncoding -> "MacintoshRoman"]', '"university=E%9Atv%9As+Lor%87nd+University"'),
    ('URLQueryEncode[{"c" -> 4, "z" -> 1, "a" -> 2}, "SortQueries" -> True]', '"a=2&c=4&z=1"'),
    ('URLQueryEncode[{"c" -> 4, "z" -> 1, "a" -> 2}]', '"c=4&z=1&a=2"'),
    ('URLQueryEncode[{"a" -> 2, "c" -> 4, "z" -> None}, "RemoveEmptyQueries" -> True]', '"a=2&c=4"'),
    ('URLQueryEncode[<|"value" -> 1, "value" -> 2|>]', '"value=2"'),
    ('URLQueryEncode[{"t" -> True, "f" -> False}]', '"t=true&f=false"'),
    ('URLQueryEncode["notrules"]', Raises(TypeError)),
    ('URLQueryEncode[{"a" -> 1}, "Bogus" -> True]', Raises(ValueError)),
    ('URLQueryDecode["name=smiley&symbol=%3A%29"]', '{"name" -> "smiley", "symbol" -> ":)"}'),
    ('URLQueryDecode[{"p1=a%2Bb", "p2=a%2Ab"}]', '{{"p1" -> "a+b"}, {"p2" -> "a*b"}}'),
    ('URLQueryDecode[URL["http://www.wolfram.com/?i=1"]]', '{"i" -> "1"}'),
    ('URLQueryDecode["university=E%9Atv%9As+Lor%87nd+University", CharacterEncoding -> "MacintoshRoman"]', '{"university" -> "Eötvös Loránd University"}'),
    ('URLQueryDecode["z=1&a=2&c=4", "SortQueries" -> True]', '{"a" -> "2", "c" -> "4", "z" -> "1"}'),
    ('URLQueryDecode["z=1&a&c=4&z=", "RemoveEmptyQueries" -> True]', '{"z" -> "1", "c" -> "4"}'),
    ('URLQueryDecode["z=1&a&c=4&z="]', '{"z" -> "1", "a" -> "", "c" -> "4", "z" -> ""}'),
    ('URLQueryEncode[URLQueryDecode[{"zed=4&anna=1", {"sara=1&", "paul=5"}}]]', '"zed=4&anna=1&sara=1&paul=5"'),
    ('URLQueryDecode[5]', Raises(TypeError)),
    ('URLParse["http://www.wolfram.com/solutions"]', '<|"Scheme" -> "http", "User" -> None, "Domain" -> "www.wolfram.com", "Port" -> None, "Path" -> {"", "solutions"}, "Query" -> {}, "Fragment" -> None|>'),
    ('URLParse["http://www.wolframalpha.com/input?i=100+USD+in+EUR", {"Path", "Query"}]', '{{"", "input"}, {"i" -> "100 USD in EUR"}}'),
    ('URLParse["http://www.wolframalpha.com/examples", "Domain"]', '"www.wolframalpha.com"'),
    ('URLParse["ssh://user:pass@host:22/base2?par1=val1#fragment", {"User", "Port", "Fragment"}]', '{"user:pass", 22, "fragment"}'),
    ('URLParse["http://www.wolfram.com/solutions/industry/astronomy", "PathString"]', '"/solutions/industry/astronomy"'),
    ('URLParse["http://www.wolframalpha.com/input/?i=tom+brady+vs+peyton+manning+touchdowns&a=_*NFLPassingTouchdowns-", "QueryString"]', '"i=tom+brady+vs+peyton+manning+touchdowns&a=_*NFLPassingTouchdowns-"'),
    ('URLParse["http://www.wolframalpha.com/input/?i=tom+brady&a=x", {"AbsolutePath", "Query"}]', '{"http://www.wolframalpha.com/input/", {"i" -> "tom brady", "a" -> "x"}}'),
    ('URLParse[URL["http://www.wolfram.com"], "Domain"]', '"www.wolfram.com"'),
    ('URLParse["http://a.b/c", "Nope"]', Raises(ValueError)),
    ('URLParse[42]', Raises(TypeError)),
    ('URLBuild[{"www.wolfram.com", "solutions"}]', '"www.wolfram.com/solutions"'),
    ('URLBuild[{"https://www.google.com", "search"}, {"q" -> "Mathematica"}]', '"https://www.google.com/search?q=Mathematica"'),
    ('URLBuild[<|"Scheme" -> "https", "Domain" -> "www.wolframalpha.com", "Path" -> "examples"|>]', '"https://www.wolframalpha.com/examples"'),
    ('URLBuild[{"https://www.wolframalpha.com", "input"}, {"i" -> "Doppler Shift 300Hz at 75 mph"}]', '"https://www.wolframalpha.com/input?i=Doppler+Shift+300Hz+at+75+mph"'),
    ('URLBuild["https://example.com", <|"n" -> {1, 2, 3}|>]', '"https://example.com/?n=1&n=2&n=3"'),
    ('URLBuild["https://example.com/?a=2", <|"n" -> {1, 2, 3}|>]', '"https://example.com/?a=2&n=1&n=2&n=3"'),
    ('URLBuild[URL["https://www.example.com/?i=1"], <|"x" -> 2|>]', '"https://www.example.com/?i=1&x=2"'),
    ('URLBuild[<|"Scheme" -> "https", "User" -> "user:info", "Domain" -> "www.domain.com", "Port" -> "80", "Path" -> {"path", "to", "resource"}, "Query" -> {"param1" -> "val2", "param2" -> "val2"}, "Fragment" -> "fragment"|>]', '"https://user:info@www.domain.com:80/path/to/resource?param1=val2&param2=val2#fragment"'),
    ('URLBuild[<|"Path" -> "path/to/dir", "Query" -> {"x" -> 3}|>]', '"path/to/dir?x=3"'),
    ('URLBuild[URLParse["https://u@h.example:8080/a/b?x=1&y=2#f"]]', '"https://u@h.example:8080/a/b?x=1&y=2#f"'),
    ('URLBuild[<|"Bogus" -> 1|>]', Raises(ValueError)),
    ('URLBuild[17]', Raises(TypeError)),
    # ------------------------------------------------------------ session switches (documented defaults)
    ('$AllowInternet', 'True'),
    ('Block[{$AllowInternet = False}, $AllowInternet]', 'False'),
    ('Block[{$AllowInternet = False}, URLRead["http://w30-blocked.example/"]]', Pred(lambda r: _failure_tag(r) == "ConnectionFailure" and "AllowInternet" in _failure_msg(r), "ConnectionFailure naming $AllowInternet")),
    ('Block[{$AllowInternet = False}, URLRead["http://127.0.0.1:{P}/text", "Body"]]', '"hello from w30"'),
    ('Block[{$AllowInternet = False}, HostLookup["w30-blocked.example"]]', Raises(ValueError, "AllowInternet")),
    ('Block[{$AllowInternet = 3}, URLRead["http://w30-blocked.example/"]]', Raises(ValueError)),
    ('$DefaultProxyRules', '<|"UseProxy" -> Automatic, "HTTP" -> None, "HTTPS" -> None, "FTP" -> None, "Socks" -> None|>'),
    ('Block[{$DefaultProxyRules = <|"UseProxy" -> Manual, "HTTP" -> {"127.0.0.1", {P}}|>}, URLRead["http://w30-proxied.example/echo", "Body"]]', Pred(lambda r: '"path": "http://w30-proxied.example/echo"' in r, "request sent to the local proxy with an absolute URL")),
    ('Block[{$DefaultProxyRules = <|"UseProxy" -> Manual, "Socks" -> {"127.0.0.1", 1080}|>}, URLRead["http://w30-proxied.example/"]]', Raises(NotImplementedError)),
    ('Block[{$DefaultProxyRules = <|"UseProxy" -> Manual, "HTTP" -> 12|>}, URLRead["http://w30-proxied.example/"]]', Raises(ValueError)),
    ('$UserAgent', 'None'),
    ('$UserAgentString', 'None'),
    ('{$UserAgent, $UserAgentString} === {None, None}', 'True'),
    ('StringStartsQ[URLRead["http://127.0.0.1:{P}/echo", "Body"], "{"] && StringContainsQ[URLRead["http://127.0.0.1:{P}/echo", "Body"], "MikoshiLang HTTPClient"] && $UserAgent === None', 'True'),
    ('$UserAgentString === None && StringQ[URLRead["http://127.0.0.1:{P}/text", "Body"]]', 'True'),
    ('Block[{$Cookies = {}}, $Cookies]', '{}'),
    ('Block[{$Cookies = {}}, URLRead["http://127.0.0.1:{P}/cookie"]; Sort[Lookup[$Cookies, "Name"]]]', '{"w30a", "w30b"}'),
    ('Block[{$Cookies = {}}, SetCookies[<|"Domain" -> "127.0.0.1", "Path" -> "/", "Name" -> "w30c", "Content" -> "v1"|>]; Lookup[$Cookies, "Content"]]', '{"v1"}'),
    ('Block[{$Cookies = {}}, SetCookies[<|"Domain" -> "127.0.0.1", "Path" -> "/", "Name" -> "w30c", "Content" -> "v1"|>]; StringContainsQ[URLRead["http://127.0.0.1:{P}/echo", "Body"], "w30c=v1"]]', 'True'),
    ('Block[{$Cookies = {}}, SetCookies[{<|"Domain" -> "a.example", "Path" -> "/", "Name" -> "n1", "Content" -> "1"|>, <|"Domain" -> "a.example", "Path" -> "/", "Name" -> "n1", "Content" -> "2"|>}]; Lookup[$Cookies, "Content"]]', '{"2"}'),
    ('SetCookies[<|"Domain" -> "a.example", "Bogus" -> 1, "Name" -> "x"|>]', Raises(ValueError)),
    ('SetCookies[5]', Raises(TypeError)),
    ('SetCookies[<|"Domain" -> "a.example", "Content" -> "1"|>]', Raises(ValueError)),
    ('Block[{$Cookies = {}}, SetCookies[{<|"Domain" -> ".wolfram.com", "Path" -> "/", "Name" -> "WR_SID", "Content" -> "a"|>, <|"Domain" -> "user.example", "Path" -> "/", "Name" -> "JSESSIONID", "Content" -> "b"|>}]; Lookup[FindCookies[".wolfram.com"], "Name"]]', '{"WR_SID"}'),
    ('Block[{$Cookies = {}}, SetCookies[{<|"Domain" -> ".wolfram.com", "Path" -> "/", "Name" -> "WR_SID", "Content" -> "a"|>, <|"Domain" -> "user.example", "Path" -> "/", "Name" -> "JSESSIONID", "Content" -> "b"|>}]; Lookup[FindCookies[<|"Name" -> "JSESSIONID"|>], "Content"]]', '{"b"}'),
    ('Block[{$Cookies = {}}, SetCookies[<|"Domain" -> "x.example", "Path" -> "/", "Name" -> "k", "Content" -> "v"|>]; FindCookies[] === $Cookies]', 'True'),
    ('FindCookies[<|"Bogus" -> 1|>]', Raises(ValueError)),
    ('FindCookies[12]', Raises(TypeError)),
    # ------------------------------------------------------------ HTTP (local http.server only)
    ('HTTPRequest["http://www.wolframalpha.com/"]', 'HTTPRequest["http://www.wolframalpha.com/", <||>]'),
    ('HTTPRequest["http://www.wolframalpha.com/"][{"Scheme", "Domain", "Method"}]', '<|"Scheme" -> "http", "Domain" -> "www.wolframalpha.com", "Method" -> "GET"|>'),
    ('HTTPRequest["http://example.com", <|"Query" -> {"a" -> {"1", "2"}}|>]["URL"]', '"http://example.com/?a=1&a=2"'),
    ('HTTPRequest["http://example.com", <|"Method" -> "POST", "Body" -> "x=1"|>]["Body"]', '"x=1"'),
    ('HTTPRequest["http://example.com", <|"Bogus" -> 1|>]', Raises(ValueError)),
    ('URLRead[HTTPRequest["http://127.0.0.1:{P}/text", <|"Method" -> 5|>]]', Raises(TypeError)),
    ('w30r = URLRead["http://127.0.0.1:{P}/text"]; {Head[w30r], w30r["StatusCode"], w30r["Body"], w30r["ContentType"]}', '{HTTPResponse, 200, "hello from w30", "text/plain; charset=utf-8"}'),
    ('HTTPResponse["Hello <strong>world</strong>"]["Body"]', '"Hello <strong>world</strong>"'),
    ('HTTPResponse["x", <|"StatusCode" -> 404|>][{"StatusCode", "StatusCodeDescription"}]', '<|"StatusCode" -> 404, "StatusCodeDescription" -> "Not Found"|>'),
    ('HTTPResponse["x"]["ContentType"]', '"text/html"'),
    ('HTTPResponse["x", <|"StatusCode" -> "abc"|>]', Raises((ValueError, TypeError))),
    ('URLRead["http://127.0.0.1:{P}/latin", "Body"]', '"café"'),
    ('HTTPRedirect["http://www.wolfram.com/"]["StatusCode"]', '302'),
    ('HTTPRedirect["http://www.wolfram.com/", <|"StatusCode" -> 301|>][{"Location", "StatusCodeDescription"}]', '<|"Location" -> "http://www.wolfram.com/", "StatusCodeDescription" -> "Moved Permanently"|>'),
    ('MemberQ[HTTPRedirect["http://www.wolfram.com/"]["Headers"], "Location" -> "http://www.wolfram.com/"]', 'True'),
    ('HTTPRedirect["http://www.wolfram.com/", <|"StatusCode" -> 200|>]', Raises(ValueError)),
    ('URLRead["http://127.0.0.1:{P}/json", "Body"]', '"{\\"a\\": 1, \\"b\\": [1, 2, 3], \\"c\\": \\"x\\"}"'),
    ('URLRead["http://127.0.0.1:{P}/bin", "BodyBytes"] == Range[0, 255]', 'True'),
    ('URLRead[{"http://127.0.0.1:{P}/text", "http://127.0.0.1:{P}/status/404"}, "StatusCode"]', '{200, 404}'),
    ('URLRead["http://127.0.0.1:{P}/status/418", {"StatusCode", "StatusCodeDescription"}]', '<|"StatusCode" -> 418, "StatusCodeDescription" -> "I\'m a Teapot"|>'),
    ('URLRead[HTTPRequest["http://127.0.0.1:{P}/echo", <|"Method" -> "PUT", "Body" -> "w30 body"|>], "Body"]', Pred(lambda r: '"method": "PUT"' in r and '"body": "w30 body"' in r, "echoed PUT with its body")),
    ('URLRead["http://127.0.0.1:{P}/text", "Nope"]', Raises(ValueError)),
    ('URLRead["http://127.0.0.1:{F}/"]', Pred(lambda r: _failure_tag(r) == "ConnectionFailure", "Failure[\"ConnectionFailure\", ...]")),
    ('URLRead["ftp://127.0.0.1/"]', Raises(NotImplementedError)),
    ('URLRead["http://127.0.0.1:{P}/redirect", "Body"]', '"hello from w30"'),
    ('URLRead["http://127.0.0.1:{P}/redirect", "StatusCode", FollowRedirects -> False]', '302'),
    ('URLRead["http://127.0.0.1:{P}/redirect303", "Body"]', Pred(lambda r: '"method": "GET"' in r and '"path": "/echo"' in r, "303 continues with GET")),
    ('URLRead["http://127.0.0.1:{P}/loop"]', Pred(lambda r: _failure_tag(r) == "ConnectionFailure" and "redirect" in _failure_msg(r), "too many redirects")),
    ('URLRead["http://127.0.0.1:{P}/text", FollowRedirects -> 3]', Raises(ValueError)),
    ('URLRead["http://127.0.0.1:{P}/echo", "Body", Authentication -> <|"Username" -> "w30user", "Password" -> "w30pass"|>]', Pred(lambda r: '"authorization": "Basic dzMwdXNlcjp3MzBwYXNz"' in r, "Basic w30user:w30pass (base64 per RFC 7617)")),
    ('URLRead["http://127.0.0.1:{P}/echo", "Body", Authentication -> <|"Headers" -> {"X-W30" -> "h"}, "Query" -> {"tok" -> "q"}|>]', Pred(lambda r: '"x-w30": "h"' in r and "tok=q" in r, "header and query added")),
    ('URLRead["http://127.0.0.1:{P}/text", Authentication -> <|"PermissionsKey" -> "k"|>]', Raises(NotImplementedError)),
    ('URLRead["http://127.0.0.1:{P}/text", Authentication -> <|"Bogus" -> 1|>]', Raises(NotImplementedError)),
    ('URLRead["http://127.0.0.1:{P}/big", "BodyBytes", ConnectionSettings -> <|"MaxDownloadSpeed" -> 1000000|>] // Length', '20000'),
    ('First[AbsoluteTiming[URLRead["http://127.0.0.1:{P}/big", ConnectionSettings -> <|"MaxDownloadSpeed" -> 40000|>]]] > 0.2', 'True'),
    ('URLRead["http://127.0.0.1:{P}/text", ConnectionSettings -> <|"MaxDownloadSpeed" -> -5|>]', Raises(ValueError)),
    ('URLRead["http://127.0.0.1:{P}/text", ConnectionSettings -> <|"Bogus" -> 1|>]', Raises(ValueError)),
    ('Block[{$Cookies = {}}, URLRead["http://127.0.0.1:{P}/cookie", CookieFunction -> None]; $Cookies]', '{}'),
    ('Block[{$Cookies = {}, w30seen = {}}, URLRead["http://127.0.0.1:{P}/cookie", CookieFunction -> ((AppendTo[w30seen, #["Name"]]) &)]; {Sort[w30seen], $Cookies}]', '{{"w30a", "w30b"}, {}}'),
    ('Block[{$Cookies = {}}, URLRead["http://127.0.0.1:{P}/cookie", CookieFunction -> Automatic]; Lookup[FindCookies[<|"Name" -> "w30b"|>], "ScriptAccessible"]]', '{False}'),
    ('URLExecute["http://127.0.0.1:{P}/json"]', '{"a" -> 1, "b" -> {1, 2, 3}, "c" -> "x"}'),
    ('URLExecute["http://127.0.0.1:{P}/csv"]', '{{1, 2}, {3, 4}}'),
    ('URLExecute["http://127.0.0.1:{P}/text"]', '"hello from w30"'),
    ('URLExecute["http://127.0.0.1:{P}/echo", {"q" -> "a b"}, "RawJSON"]["path"]', '"/echo?q=a+b"'),
    ('URLExecute["http://127.0.0.1:{P}/status/500"]', Raises(ValueError)),
    ('URLExecute["http://127.0.0.1:{P}/redirect", FollowRedirects -> False]', Pred(lambda r: _failure_tag(r) == "ConnectionFailure", "unfollowed redirect gives Failure")),
    ('URLDownload["http://127.0.0.1:{P}/bin"]', Pred(lambda r: _file_bytes(r) == bytes(range(256)), "File with the 256 bytes served")),
    ('URLDownload["http://127.0.0.1:{P}/named", "{T}"]', Pred(lambda r: _file_name(r).startswith("w30") and _file_name(r).endswith(".txt") and _file_bytes(r) == b"named file", "server-given file name w30.txt in the directory")),
    ('URLDownload["http://127.0.0.1:{P}/text", "{T}/w30-elements.txt", {"StatusCode", "ContentType"}]', Pred(lambda r: _cleanup_tmp("w30-elements.txt") and str(r) == str(_ev('<|"StatusCode" -> 200, "ContentType" -> "text/plain; charset=utf-8"|>')), "StatusCode and ContentType of the download")),
    ('URLDownload["http://127.0.0.1:{P}/status/404"]', Raises(ValueError)),
    ('URLDownload["http://127.0.0.1:{F}/x"]', Pred(lambda r: _failure_tag(r) == "ConnectionFailure", "Failure[\"ConnectionFailure\", ...]")),
    ('URLResponseTime["http://127.0.0.1:{P}/text"] // Head', 'Quantity'),
    ('Keys[URLResponseTime["http://127.0.0.1:{P}/text", "ConnectionTimes"]]', '{"NameLookup", "HostConnection", "SSLHandshake", "PreTransfer"}'),
    ('w30t = URLResponseTime["http://127.0.0.1:{P}/text", All]; w30t["TransactionTotal"][[1]] >= w30t["TransferInitiation"][[1]] >= w30t["PreTransfer"][[1]] >= 0', 'True'),
    ('URLResponseTime["http://127.0.0.1:{F}/"]', Raises(ValueError)),
    ('URLResponseTime["http://127.0.0.1:{P}/", "Nope"]', Raises(ValueError)),
    ('URLExpand["http://127.0.0.1:{P}/redirect"]', '"http://127.0.0.1:{P}/text"'),
    ('URLExpand[{URL["http://127.0.0.1:{P}/redirect303"], "http://127.0.0.1:{P}/text"}]', '{"http://127.0.0.1:{P}/echo", "http://127.0.0.1:{P}/text"}'),
    ('URLExpand["http://127.0.0.1:{P}/loop"]', Pred(lambda r: _failure_tag(r) == "ConnectionFailure", "too many redirects")),
    ('URLExpand["http://127.0.0.1:{P}/text", Interactive -> 3]', Raises(ValueError)),
    ('URLExpand[w30sym]', 'URLExpand[w30sym]'),
    # ------------------------------------------------------------ hosts (no DNS: localhost, literals, IDNA failure)
    ('IPAddress["127.0.0.1"]', 'IPAddress["127.0.0.1"]'),
    ('IPAddress["::1"]', 'IPAddress["::1"]'),
    ('IPAddress["300.1.1.1"]', Raises(ValueError)),
    ('IPAddress[5]', Raises(TypeError)),
    ('HostLookup["localhost"]', Pred(lambda r: "127.0.0.1" in str(r) or "::1" in str(r), "localhost resolves to a loopback IPAddress")),
    ('HostLookup[IPAddress["127.0.0.1"], "IPv4Addresses"]', '{IPAddress["127.0.0.1"]}'),
    ('HostLookup["127.0.0.1", "FullName"] // StringQ', 'True'),
    ('HostLookup["a..b"]', 'Missing["NotFound"]'),
    ('HostLookup["localhost", "Nope"]', Raises(ValueError)),
    ('VectorQ[$MachineAddresses, StringQ]', 'True'),
    ('AllTrue[$MachineAddresses, !StringStartsQ[#, "127."] &]', 'True'),
    ('w30ma = $MachineAddresses; w30ma === {} || AllTrue[w30ma, Head[IPAddress[#]] === IPAddress &]', 'True'),
    ('$DefaultNetworkInterface === None || StringQ[$DefaultNetworkInterface]', 'True'),
    ('w30ni = $DefaultNetworkInterface; w30ni =!= ""', 'True'),
    ('Head[$DefaultNetworkInterface] =!= List', 'True'),
    # ------------------------------------------------------------ sockets (TCP, local echo server)
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; SocketWriteMessage[w30s, "hello"]; ByteArrayToString[SocketReadMessage[w30s]]', '"HELLO"'),
    ('w30s = SocketConnect["127.0.0.1:{E}", "TCP"]; {w30s["DestinationPort"], w30s["Protocol"], w30s["Type"]}', '{{E}, "TCP", "Client"}'),
    ('w30s = SocketConnect[{IPAddress["127.0.0.1"], {E}}]; SocketWriteMessage[w30s, ByteArray[{97, 98}]]; Normal[SocketReadMessage[w30s]]', '{65, 66}'),
    ('SocketConnect[{"127.0.0.1", {F}}]', Raises(ValueError)),
    ('SocketConnect[{"127.0.0.1", {E}}, "ZMQ"]', Raises(NotImplementedError)),
    ('Block[{$AllowInternet = False}, SocketConnect[{"w30-blocked.example", 80}]]', Raises(ValueError, "AllowInternet")),
    ('w30srv = SocketOpen[Automatic]; {w30srv["Type"], w30srv["SourceIPAddress"]}', '{"Server", IPAddress["127.0.0.1"]}'),
    ('w30srv = SocketOpen[Automatic]; w30c = SocketConnect[w30srv]; SocketWriteMessage[w30c, "ping"]; ByteArrayToString[SocketReadMessage[w30srv]]', '"ping"'),
    ('w30srv = SocketOpen[{"127.0.0.1", Automatic}, "TCP"]; IntegerQ[w30srv["SourcePort"]] && w30srv["SourcePort"] > 0', 'True'),
    ('SocketOpen[{"203.0.113.9", 4000}]', Raises(ValueError)),
    ('SocketOpen[Automatic, "ZMQ"]', Raises(NotImplementedError)),
    ('w30srv = SocketOpen[Automatic]; w30c = SocketConnect[w30srv]; SocketWriteMessage[w30c, "x"]; SocketReadMessage[w30srv]; w30rm = First[w30srv["ConnectedClients"]]; {w30rm["Type"], w30rm["SourcePort"] == w30c["SourcePort"], MemberQ[Sockets["Remote"], w30rm]}', '{"Remote", True, True}'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; Head[SocketObject[w30s[[1]]]]', 'SocketObject'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; w30s["Nope"]', Raises(ValueError)),
    ('SocketObject["TCP-w30-not-a-socket"]', Raises(ValueError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; SocketWriteMessage[w30s, "abc"]; ByteArrayToString[SocketReadMessage[w30s]]', '"ABC"'),
    ('SocketReadMessage[SocketObject[w30sym]]', Raises(TypeError)),
    ('SocketReadMessage[5]', Raises(TypeError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; {SocketReadyQ[w30s], SocketWriteMessage[w30s, "x"]; SocketReadyQ[w30s, 5]}', '{False, True}'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; SocketReadyQ[w30s, Quantity[0.05, "Seconds"]]', 'False'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; SocketReadyQ[w30s, -1]', Raises(ValueError)),
    ('w30a = SocketConnect[{"127.0.0.1", {E}}]; w30b = SocketConnect[{"127.0.0.1", {E}}]; SocketWriteMessage[w30a, "1"]; SocketWriteMessage[w30b, "2"]; SocketWaitAll[{w30a, w30b}] === {w30a, w30b}', 'True'),
    ('SocketWaitAll[{}]', '{}'),
    ('SocketWaitAll[5]', Raises(TypeError)),
    ('w30a = SocketConnect[{"127.0.0.1", {E}}]; w30b = SocketConnect[{"127.0.0.1", {E}}]; SocketWriteMessage[w30b, "2"]; SocketWaitNext[{w30a, w30b}] === w30b', 'True'),
    ('SocketWaitNext[{}]', Raises(ValueError)),
    ('SocketWaitNext[7]', Raises(TypeError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; SocketWriteMessage[w30s, "héllo"]; ByteArrayToString[SocketReadMessage[w30s]]', '"HéLLO"'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; SocketWriteMessage[w30s, <|"DataByteArray" -> ByteArray[{1}]|>]', Raises(NotImplementedError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; SocketWriteMessage[w30s, 42]', Raises(TypeError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; MemberQ[Sockets[], w30s]', 'True'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; {MemberQ[Sockets["Client"], w30s], MemberQ[Sockets["Server"], w30s], Sockets["ZMQ"]}', '{True, False, {}}'),
    ('Sockets["Nope"]', Raises(ValueError)),
    ('w30srv = SocketOpen[Automatic]; w30got = {}; w30l = SocketListen[w30srv, (AppendTo[w30got, #["Data"]]) &]; w30c = SocketConnect[w30srv]; SocketWriteMessage[w30c, "evt"]; Pause[0.5]; {w30got, w30l["EventCount"]}', '{{"evt"}, 1}'),
    ('w30got = {}; w30l = SocketListen[Automatic, (AppendTo[w30got, #["Data"]]) &, RecordSeparators -> {";"}]; w30c = SocketConnect[w30l["Socket"]]; SocketWriteMessage[w30c, "a;b;"]; Pause[0.5]; w30got', '{"a", "b"}'),
    ('w30l = SocketListen[Automatic, (StringLength[5]; Null) &]; w30c = SocketConnect[w30l["Socket"]]; SocketWriteMessage[w30c, "x"]; Pause[0.5]; {Length[w30l["HandlerErrors"]], w30l["EventCount"]}', '{1, 1}'),
    ('SocketListen[Automatic, Print, "Bogus" -> 1]', Raises(ValueError)),
    ('w30l = SocketListen[Automatic, Identity]; {Head[w30l], Head[w30l["Socket"]], w30l["EventCount"]}', '{SocketListener, SocketObject, 0}'),
    ('w30l = SocketListen[Automatic, Identity, HandlerFunctionsKeys -> {"Data", "DataBytes"}]; w30l["HandlerFunctionsKeys"]', '{"Data", "DataBytes"}'),
    ('w30l = SocketListen[Automatic, Identity]; w30l["Nope"]', Raises(ValueError)),
    ('SocketListener["TCP-w30-not-active"]', Raises(ValueError)),
    ('MatchQ[SocketListen[Automatic, Identity], _SocketListener]', 'True'),
    ('w30l = SocketListen[Automatic, Identity]; SocketListener @@ w30l === w30l', 'True'),
    ('SocketListener[5]', Raises(TypeError)),
    # W27 stream functions on sockets (wired at integration to wl_external.close_socket/write_socket/delete_listener)
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; WriteString[w30s, "ws"]; ByteArrayToString[SocketReadMessage[w30s]]', '"WS"'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; BinaryWrite[w30s, {97, 98}] === w30s', 'True'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; BinaryWrite[w30s, {97, 98}]; Normal[SocketReadMessage[w30s]]', '{65, 66}'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; Close[w30s] == "127.0.0.1:{E}"', 'True'),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; Close[w30s]; SocketObject[w30s[[1]]]', Raises(ValueError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; Close[w30s]; Close[w30s]', Raises(ValueError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; Close[w30s]; WriteString[w30s, "x"]', Raises(ValueError)),
    ('w30l = SocketListen[Automatic, Identity]; DeleteObject[w30l] === Null', 'True'),
    ('w30l = SocketListen[Automatic, Identity]; DeleteObject[w30l]; SocketListener @@ w30l', Raises(ValueError)),
    ('w30l = SocketListen[Automatic, Identity]; DeleteObject[w30l]; DeleteObject[w30l]', Raises(ValueError)),
    ('w30s = SocketConnect[{"127.0.0.1", {E}}]; DeleteObject[w30s]; SocketObject[w30s[[1]]]', Raises(ValueError)),
    ('DeleteObject[5]', Raises(NotImplementedError)),
    ('DeleteObject[w30notanobject]', 'DeleteObject[w30notanobject]'),
]
COMP = [
    # ------------------------------------------------------------ Compile (Wolfram reference examples; N of the uncompiled expression)
    ('Compile[{{x, _Real}}, Sin[x] + x^2 - 1/(1 + x)][2]', 0.9092974268256817 + 4 - 1/3, 1e-12),
    ('Compile[{x, y}, Sqrt[x y]][1, 2]', 2 ** 0.5),
    ('Compile[{x, y}, Sqrt[x y]][w30sa, 2]', 'Sqrt[2] Sqrt[w30sa]'),
    ('Compile[{x}, x^2, RuntimeAttributes -> {Listable}][{-5, 0, 5}]', '{25., 0., 25.}'),
    ('Compile[{{n, _Integer}}, n/2][3]', 1.5),
    ('Compile[{{n, _Integer}}, n + 1][41]', '42'),
    ('Compile[{{x, _Integer}}, x + 1][2.5]', 3.5),
    ('Compile[{{x, _Real}, {n, _Integer}}, Module[{sum, inc}, sum = 1.0; inc = 1.0; Do[inc = inc*x/i; sum = sum + inc, {i, n}]; sum]][1., 20]', 2.718281828459045, 1e-13),
    ('Compile[{{m, _Real, 2}}, Total[m, 2]][{{1, 2}, {3, 4}}]', '10.'),
    ('Compile[{{b, True | False}}, If[b, 1, 0]][True]', '1'),
    ('Compile[{{z, _Complex}}, z^2][1 + I]', 2j),
    ('Compile[{x}, x^2][1, 2]', Raises(ValueError)),
    ('Compile[x, x^2]', Raises(TypeError)),
    ('Compile[{{x, _String}}, x]', Raises(ValueError)),
    ('Compile[{{x, _Real, -1}}, x]', Raises(ValueError)),
    ('Compile[{x, x}, x]', Raises(ValueError)),
    ('Head[Compile[{x}, x^2]]', 'CompiledFunction'),
    ('w30cf = Compile[{x}, x^2]; w30cf[[2]]', '{{_Real, 0}}'),
    ('CompiledFunction[1, 2]', Raises(TypeError)),
    ('Compile[{{x}}, x^2 + Sin[x^2], CompilationTarget -> "C"][10]', 99.4936343588902, 1e-12),
    ('Compile[{{x}}, x^2, CompilationTarget -> "WVM"][3]', '9.'),
    ('Compile[{{x}}, x, CompilationTarget -> "LLVM"]', Raises(ValueError)),
    ('Block[{$CompilationTarget = "C"}, Compile[{{x}}, x^2 + Sin[x^2]]][10]', 99.4936343588902, 1e-12),
    ('$CompilationTarget', '"WVM"'),
    ('Block[{$CompilationTarget = "C"}, Compile[{{x}}, x][[4]]["CompilationTarget"]]', '"C"'),
    ('Block[{$CompilationTarget = "Fortran"}, Compile[{{x}}, x]]', Raises(ValueError)),
    ('Compile[{{x}}, x^2 + Sin[x^2], CompilationOptions -> {"ExpressionOptimization" -> False}][10]', 99.4936343588902, 1e-12),
    ('Compile[{{x}}, x^2, CompilationOptions -> {"InlineCompiledFunctions" -> True, "InlineExternalDefinitions" -> Automatic}][2]', '4.'),
    ('Compile[{{x}}, x, CompilationOptions -> {"Bogus" -> True}]', Raises(ValueError)),
    ('Compile[{{x}}, x, CompilationOptions -> {"ExpressionOptimization" -> 3}]', Raises(ValueError)),
    ('Compile[{{x}}, x^2, Parallelization -> True, RuntimeAttributes -> {Listable}][{1, 2}]', '{1., 4.}'),
    ('Compile[{{x}}, x, RuntimeAttributes -> {Orderless}]', Raises(ValueError)),
    # ------------------------------------------------------------ FunctionCompile (Wolfram reference examples)
    ('FunctionCompile[Function[Typed[arg, "MachineInteger"], arg + 1]][20]', '21'),
    ('Head[FunctionCompile[Function[Typed[arg, "MachineInteger"], arg + 1]]]', 'CompiledCodeFunction'),
    ('FunctionCompile[Function[Typed[arg, "MachineInteger"], arg + 1]][True, 1.5][[1]]', '"ArgumentCount"'),
    ('FunctionCompile[Function[Typed[arg, "MachineInteger"], arg + 1]][10.5][[1]]', '"ArgumentType"'),
    ('FunctionCompile[Function[Typed[arg, "MachineInteger"], arg + 1]][10.5][[2]]["MessageParameters"]', '{1}'),
    ('FunctionCompile[Function[Typed[len, "MachineInteger"], Table[i, {i, len}]]][5]', '{1, 2, 3, 4, 5}'),
    ('FunctionCompile[Function[Typed[n, "MachineInteger"], Nest[Sin, 10.1, n]]][200]', -0.11989743948479191, 1e-12),
    ('FunctionCompile[Function[Typed[x, "Real64"], Sqrt[x]]][10.5]', 10.5 ** 0.5),
    ('FunctionCompile[Function[Typed[x, "Real64"], Sqrt[x]]][-10.2]', 1j * 10.2 ** 0.5),
    ('FunctionCompile[Function[Typed[arg, "Complex128"], arg + 2]][20.2 + 3.4 I]', 22.2 + 3.4j),
    ('FunctionCompile[{Function[Typed[x, "Integer64"], x + 1], Function[Typed[x, "Integer64"], x - 1]}][[2]][10]', '9'),
    ('FunctionCompile[<|"inc" -> Function[Typed[x, "Integer64"], x + 1]|>]["inc"][1]', '2'),
    ('FunctionCompile[Function[x, x + 1]]', Raises(ValueError)),
    ('FunctionCompile[Function[Typed[x, "NoSuchType"], x]]', Raises(ValueError)),
    ('FunctionCompile[#1 + 1 &]', Raises(ValueError)),
    ('FunctionCompile[5]', Raises(TypeError)),
    ('FunctionCompile[Function[Typed[x, "String"], StringLength[x]]]["abcd"]', '4'),
    ('FunctionCompile[Function[Typed[b, "Boolean"], !b]][True]', 'False'),
    ('FunctionCompile[Function[Typed[x, "Integer64"], x], CompilerOptions -> 5]', Raises(ValueError)),
    ('FunctionCompile[Function[Typed[x, "Integer64"], x + 1], CompilerOptions -> <|"AbortHandling" -> False|>][1]', '2'),
    ('FunctionCompile[Function[Typed[x, "Integer64"], x + 1], CompilerOptions -> Automatic][41]', '42'),
    ('FunctionCompile[Function[{Typed[x, "UnsignedInteger8"], Typed[y, "UnsignedInteger8"]}, x + y]][250, 250]', '500'),
    ('FunctionCompile[Function[{Typed[x, "UnsignedInteger8"], Typed[y, "UnsignedInteger8"]}, x + y], CompilerRuntimeErrorAction -> None][250, 250][[2]]["MessageParameters"]', '{"IntegerOverflow"}'),
    ('FunctionCompile[Function[{Typed[x, "UnsignedInteger8"], Typed[y, "UnsignedInteger8"]}, x + y], CompilerRuntimeErrorAction -> (w30h[#[[1]]] &)][250, 250]', 'w30h["RuntimeError"]'),
    ('FunctionCompile[Function[{Typed[x, "UnsignedInteger8"], Typed[y, "UnsignedInteger8"]}, x + y], CompilerRuntimeErrorAction -> "Evaluate"][250, 250]', '500'),
    ('FunctionCompile[Function[{Typed[x, "UnsignedInteger8"], Typed[y, "UnsignedInteger8"]}, x + y], CompilerRuntimeErrorAction -> None][100, 100]', '200'),
    ('FunctionCompile[Function[Typed[x, "UnsignedInteger8"], x]][256][[1]]', '"ArgumentType"'),
    ('FunctionCompile[Function[Typed[x, "Integer64"], x], Bogus -> 1]', Raises(ValueError)),
    ('w30ccf = FunctionCompile[Function[Typed[x, "Integer64"], x^2]]; w30ccf[[1]]["Signature"]', 'TypeSpecifier[{"Integer64"} -> "Integer64"]'),
    ('CompiledCodeFunction[5]', Raises(TypeError)),
    ('CompiledCodeFunction[<|"Signature" -> 1|>]', Raises(ValueError)),
    ('FunctionCompile[Function[Typed[x, TypeSpecifier["PackedArray"]["Real64", 1]], Total[x]]][{1., 2., 3.5}]', '6.5'),
    ('FunctionCompile[Function[Typed[x, TypeSpecifier["PackedArray"]["Real64", 1]], Total[x]]][{{1.}}][[1]]', '"ArgumentType"'),
    ('FunctionCompile[Function[Typed[x, TypeSpecifier["ListVector"]["String"]], StringJoin[x]]][{"a", "b"}]', '"ab"'),
    ('FunctionCompile[Function[Typed[x, TypeSpecifier["Matrix"]["Real64"]], x]]', Raises(NotImplementedError)),
    # declarations
    ('FunctionCompile[FunctionDeclaration[w30AddTwo, Typed[{"Integer64"} -> "Integer64"]@Function[arg, 2 + arg]], Function[Typed[arg, "Integer64"], w30AddTwo[arg]]][200]', '202'),
    ('FunctionCompile[FunctionDeclaration[w30sq, Typed[{"Real64"} -> "Real64"]@Function[{arg}, arg^2]], w30sq][1.5]', '2.25'),
    ('FunctionCompile[{FunctionDeclaration[w30ov, Typed[{"Integer64"} -> "Integer64"]@Function[a, a + 1]], FunctionDeclaration[w30ov, Typed[{"String"} -> "String"]@Function[a, StringJoin[a, "!"]]]}, Function[{Typed[n, "Integer64"], Typed[s, "String"]}, {w30ov[n], w30ov[s]}]][1, "hi"]', '{2, "hi!"}'),
    ('FunctionDeclaration[w30f, 5]', Raises(TypeError)),
    ('FunctionDeclaration[w30f, Typed[{"Integer64"} -> "Integer64"]@Function[a, a], "Inline" -> "Sometimes"]', Raises(ValueError)),
    ('FunctionDeclaration[w30f, Typed[{"Integer64"} -> "Integer64"]@Function[a, a], "Inline" -> "Always"] // Head', 'FunctionDeclaration'),
    ('FunctionDeclaration[w30f, Typed[{"Integer64"} -> "Integer64"]@KernelFunction[w30g]]', Raises(NotImplementedError)),
    ('FunctionCompile[{FunctionDeclaration[w30poly, Typed[ForAllType[t, {TypeSpecifier["ListVector"][t]} -> t]]@Function[list, First[list]]]}, Function[Typed[list, TypeSpecifier["ListVector"]["MachineInteger"]], w30poly[list]]][{7, 3, 6, 4}]', '7'),
    ('FunctionCompile[{FunctionDeclaration[w30va, Typed[ForAllType[t, {"MachineInteger", SequenceType[t]} -> "MachineInteger"]]@Function[{e1, seq}, e1 + w30va[seq]]], FunctionDeclaration[w30va, Typed[{"MachineInteger"} -> "MachineInteger"]@Function[{e1}, e1]]}, Function[{Typed[x, "MachineInteger"], Typed[y, "MachineInteger"], Typed[z, "MachineInteger"]}, w30va[x, y, z]]][1, 2, 3]', '6'),
    ('FunctionCompile[{FunctionDeclaration[w30rk, Typed[ForAllType[r, {TypeSpecifier["PackedArray"]["MachineInteger", r]} -> TypeSpecifier["PackedArray"]["MachineInteger", TypeEvaluate[r + 1]]]]@Function[array, {array}]]}, Function[Typed[arg, TypeSpecifier["PackedArray"]["MachineInteger", 2]], w30rk[arg]]][{{1, 2, 3}, {4, 5, 6}}]', '{{{1, 2, 3}, {4, 5, 6}}}'),
    ('FunctionCompile[FunctionDeclaration[w30at, Typed[ForAllType[t, Element[t, "Integers"], {t} -> t]]@Function[arg, arg + Typed[2, TypeOf[arg]]]], Function[Typed[arg, "Integer32"], w30at[arg]]][10]', '12'),
    ('FunctionCompile[FunctionDeclaration[w30at2, Typed[ForAllType[t, Element[t, "Integers"], {t} -> t]]@Function[arg, arg + 2]], Function[Typed[arg, "Real64"], w30at2[arg]]][1.5]', Raises(TypeError)),
    ('FunctionCompile[FunctionDeclaration[w30rt, Typed[{"Integer64"} -> "Integer64"]@Function[a, a/2]], Function[Typed[a, "Integer64"], w30rt[a]]][3]', Raises(TypeError)),
    ('FunctionCompile[TypeDeclaration["Macro", "w30fooInteger", "MachineInteger"], Function[Typed[arg, "w30fooInteger"], arg + 2]][5]', '7'),
    ('FunctionCompile[TypeDeclaration["Alias", "w30Int", "Integer64"], Function[Typed[arg, "w30Int"], arg * 3]][5]', '15'),
    # X01 implemented product types (CreateTypeInstance); this used to raise NotImplementedError
    ('FunctionCompile[TypeDeclaration["Product", "w30Prod", <|"a" -> "MachineInteger"|>], Function[Typed[x, "Integer64"], x]][4]', '4'),
    ('TypeDeclaration["Product", "w30Prod", <|"field1" -> "MachineInteger", "field2" -> "Boolean"|>] // Head', 'TypeDeclaration'),
    ('TypeDeclaration["Struct", "w30x", "Integer64"]', Raises(ValueError)),
    ('TypeDeclaration["Macro", 5, "Integer64"]', Raises(TypeError)),
    ('OperationDeclaration["w30SV", "Increment", Typed[{"w30SV"} -> "MachineInteger"]@Function[{arg}, 1]] // Head', 'OperationDeclaration'),
    ('FunctionCompile[{TypeDeclaration["Product", "w30SV", <|"value" -> "MachineInteger"|>], OperationDeclaration["w30SV", "Increment", Typed[{"w30SV"} -> "MachineInteger"]@Function[{arg}, 1]]}, Function[Typed[x, "Integer64"], x]]', Raises(NotImplementedError)),
    ('OperationDeclaration["w30SV", 5, 6]', Raises(TypeError)),
    # environments
    ('w30env = CreateCompilerEnvironment[]; CompilerEnvironmentAppendTo[w30env, {FunctionDeclaration[w30AddTwo, Typed[{"Integer64"} -> "Integer64"]@Function[arg, 2 + arg]]}]; FunctionCompile[Function[{Typed[arg, "Integer64"]}, w30AddTwo[arg]], CompilerEnvironment -> w30env][10]', '12'),
    ('Block[{$CompilerEnvironment = CreateCompilerEnvironment[]}, CompilerEnvironmentAppendTo[FunctionDeclaration[w30AddThree, Typed[{"Integer64"} -> "Integer64"]@Function[arg, 3 + arg]]]; FunctionCompile[Function[{Typed[arg, "Integer64"]}, w30AddThree[arg]]][1]]', '4'),
    ('w30env2 = CreateCompilerEnvironment[]; CompilerEnvironmentAppendTo[w30env2, FunctionDeclaration[w30k, Typed[{"Integer64"} -> "Integer64"]@Function[a, a]]] === w30env2', 'True'),
    ('CompilerEnvironmentAppendTo[CreateCompilerEnvironment[], {5}]', Raises(TypeError)),
    ('CompilerEnvironmentAppendTo[CreateCompilerEnvironment[], FunctionDeclaration[w30k, Typed[{"NoType"} -> "Integer64"]@Function[a, a]]]', Raises(ValueError)),
    ('Head[CreateCompilerEnvironment[]]', 'CompilerEnvironmentObject'),
    ('CreateCompilerEnvironment[TargetSystem -> "Windows-x86-64"][TargetSystem]', '{"Windows-x86-64"}'),
    ('CreateCompilerEnvironment[TargetSystem -> {"MacOSX-x86-64", "MacOSX-ARM64"}][TargetSystem]', '{"MacOSX-x86-64", "MacOSX-ARM64"}'),
    ('CreateCompilerEnvironment[TargetSystem -> "Amiga"]', Raises(ValueError)),
    ('CreateCompilerEnvironment[Bogus -> 1]', Raises(ValueError)),
    ('CreateCompilerEnvironment[TargetSystem -> All][TargetSystem]', 'All'),
    ('$TargetSystems', '{"Linux-ARM", "Linux-x86-64", "MacOSX-ARM64", "MacOSX-x86-64", "Windows-x86-64"}'),
    ('SubsetQ[$TargetSystems, {"Linux-x86-64", "Windows-x86-64"}]', 'True'),
    ('Length[$TargetSystems]', '5'),
    ('$CompilerEnvironment // Head', 'CompilerEnvironmentObject'),
    ('Block[{$CompilerEnvironment = CreateCompilerEnvironment[]}, $CompilerEnvironment["Declarations"]]', '{}'),
    ('FunctionCompile[Function[{Typed[arg, "Integer64"]}, arg + 1], CompilerEnvironment -> $CompilerEnvironment][1]', '2'),
    ('w30env3 = CreateCompilerEnvironment[]; CompilerEnvironmentAppendTo[w30env3, {TypeDeclaration["Macro", "w30I", "Integer64"]}]; Length[w30env3["Declarations"]]', '1'),
    ('CompilerEnvironmentObject[<|"ID" -> "w30-none"|>]', Raises(ValueError)),
    ('CreateCompilerEnvironment[]["Nope"]', Raises(ValueError)),
    ('FunctionCompile[Function[Typed[x, "Integer64"], x], CompilerEnvironment -> 5]', Raises(ValueError)),
    # types
    ('Function[{Typed[x, "Real64"]}, x + 1]["two"]', '"two" + 1'),
    ('Function[Typed[arg, "MachineInteger"], arg + 1][10]', '11'),
    ('Typed[{"Integer64"} -> "Integer64"][Function[a, a]]', 'Typed[Function[a, a], {"Integer64"} -> "Integer64"]'),
    ('Typed[w30x, w30f[1]]', Raises(ValueError)),
    ('Typed[w30x, "w30DeclaredType"]', 'Typed[w30x, "w30DeclaredType"]'),
    ('Function[{x}, x + Typed[1, "Real64"]]["two"]', '"two" + Typed[1, "Real64"]'),
    ('TypeSpecifier["PackedArray"]["Real64", 1] // Head', 'TypeSpecifier["PackedArray"]'),
    ('TypeSpecifier["Integer64"]', 'TypeSpecifier["Integer64"]'),
    ('TypeSpecifier["integer"]', Raises(ValueError)),
    ('TypeHint[1, "UnsignedInteger8"]', '1'),
    ('FunctionCompile[Function[{}, TypeHint[1, "UnsignedInteger8"]]][]', '1'),
    ('TypeHint[1, "NoSuchType"]', Raises(ValueError)),
    ('Function[{}, IfCompiled["compiled", "uncompiled"]][]', '"uncompiled"'),
    ('FunctionCompile[Function[{}, IfCompiled["compiled", "uncompiled"]]][]', '"compiled"'),
    ('Block[{w30cnt = 0}, IfCompiled[w30cnt += 10, w30cnt += 1]; w30cnt]', '1'),
    ('TypeOf[Function[{Typed[x, "Real64"], Typed[y, "Integer64"]}, x + y]]', 'TypeSpecifier[{"Real64", "Integer64"} -> "Real64"]'),
    ('TypeOf[5040]', 'TypeSpecifier["Integer64"]'),
    ('TypeOf[StringLength[str]^2 + 3.6, Typed[str, "String"]]', 'TypeSpecifier["Real64"]'),
    ('TypeOf[Table[a + b, {a, 10}, {b, 5}] . Range[5]]', 'TypeSpecifier["PackedArray"]["Integer64", 1]'),
    ('TypeOf[Function[Typed[arg, "Integer64"], w30Sub[arg]], FunctionDeclaration[w30Sub, Typed[{"Integer64"} -> "Integer64"]@Function[arg, arg + 1]]]', 'TypeSpecifier[{"Integer64"} -> "Integer64"]'),
    ('TypeOf[{1, 2.5}]', 'TypeSpecifier["PackedArray"]["Real64", 1]'),
    ('TypeOf[w30unknown[1]]', Raises(NotImplementedError)),
    ('TypeOf[x + 1, Typed[x, "Nope"]]', Raises(ValueError)),
    ('LiteralType[2] // Head', 'LiteralType'),
    ('FunctionCompile[Function[Typed[x, TypeSpecifier["PackedArray"]["MachineInteger", LiteralType[2]]], Total[x, 2]]][{{1, 2}, {3, 4}}]', '10'),
    ('LiteralType[1.5]', Raises(TypeError)),
    ('ForAllType[t, {t} -> t] // Head', 'ForAllType'),
    ('ForAllType[1, {t} -> t]', Raises(TypeError)),
    ('ForAllType[t]', Raises(TypeError)),
    ('SequenceType[t] // Head', 'SequenceType'),
    ('SequenceType["t", 2]', Raises(TypeError)),
    ('SequenceType[1]', Raises(TypeError)),
    ('TypeEvaluate[r + 1] // Head', 'TypeEvaluate'),
    ('TypeEvaluate[1, 2]', Raises(TypeError)),
    ('FunctionCompile[{FunctionDeclaration[w30te, Typed[ForAllType[r, {TypeSpecifier["PackedArray"]["MachineInteger", r]} -> TypeSpecifier["PackedArray"]["MachineInteger", TypeEvaluate[r + 1]]]]@Function[array, array]]}, Function[Typed[arg, TypeSpecifier["PackedArray"]["MachineInteger", 1]], w30te[arg]]][{1, 2}]', Raises(TypeError)),
    # options consumed elsewhere
    ('NIntegrate[x^2, {x, 0, 1}, Compiled -> True]', 1 / 3),
    ('NIntegrate[x^2, {x, 0, 1}, Compiled -> False] == NIntegrate[x^2, {x, 0, 1}, Compiled -> True]', 'True'),
    ('FindRoot[Cos[x] == x, {x, 1}, Compiled -> False]', Pred(lambda r: abs(float(r.args[0].args[1]) - 0.7390851332151607) < 1e-9, "x -> 0.739085")),
    ('Head[BayesianMinimization[(# - 2)^2 &, {0, 4}, AssumeDeterministic -> True, MaxIterations -> 4]]', 'BayesianMinimizationObject'),
    ('BayesianMinimization[(# - 2)^2 &, {0, 4}, AssumeDeterministic -> False, MaxIterations -> 4]["MinimumValue"] <= 4', 'True'),
    ('w30bm = BayesianMinimization[(# - 2)^2 &, {0, 4}, AssumeDeterministic -> True, MaxIterations -> 4]; w30bm["MinimumValue"] == Min[Values[w30bm["EvaluationHistory"]]]', 'True'),
    ('MatchQ[Compile[{x}, x + 1], _CompiledFunction]', 'True'),
    ('w30cf2 = Compile[{x}, x + 1]; CompiledFunction @@ w30cf2 === w30cf2', 'True'),
    ('MatchQ[FunctionCompile[Function[Typed[x, "Integer64"], x]], _CompiledCodeFunction]', 'True'),
    ('MatchQ[CreateCompilerEnvironment[], _CompilerEnvironmentObject]', 'True'),
    ('w30env4 = CreateCompilerEnvironment[]; CompilerEnvironmentObject @@ w30env4 === w30env4', 'True'),
]
FFI = [
    # ------------------------------------------------------------ ctypes against the C standard library only
    ('w30abs = ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"]; w30abs[-12]', '12'),
    ('ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"][x]', Raises(TypeError)),
    ('ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"][2^40]', Raises(ValueError)),
    ('ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"][1, 2]', Raises(ValueError)),
    ('ForeignFunctionLoad["c", "w30_no_such_symbol", {"CInt"} -> "CInt"]', Raises(ValueError)),
    ('ForeignFunctionLoad["w30-no-such-library", "abs", {"CInt"} -> "CInt"]', Raises(ValueError)),
    ('ForeignFunctionLoad["c", "abs", {"CBogus"} -> "CInt"]', Raises(ValueError)),
    ('ForeignFunctionLoad["c", "abs", {{"CInt", "CInt"}} -> "CInt"]', Raises(NotImplementedError)),
    ('ForeignFunctionLoad["c", "toupper", {"CInt"} -> "CInt"][97]', '65'),
    ('ForeignFunctionLoad["c", "labs", {"CLong"} -> "CLong"][-2^40]', '2^40'),
    ('ForeignFunctionLoad["c", "strlen", {"OpaqueRawPointer"} -> "CSizeT"][RawMemoryExport["hello"]]', '5'),
    ('ForeignFunctionLoad["c", "atoi", {"OpaqueRawPointer"} -> "CInt"][RawMemoryExport["-1234xyz"]]', '-1234'),
    ('ForeignFunctionLoad["c", "strtod", {"OpaqueRawPointer", "OpaqueRawPointer"} -> "CDouble"][RawMemoryExport["2.5e3"], OpaqueRawPointer[0]]', 2500.0),
    ('ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"][[3]]["Type"]', '{"CInt"} -> "CInt"'),
    ('ForeignFunction[1, 2, <|"FunctionPointer" -> OpaqueRawPointer[1], "Type" -> {"CInt"} -> "CInt"|>]', Raises(ValueError)),
    ('ForeignFunction[1, 2, 3]', Raises(TypeError)),
    ('ForeignFunctionLoad["c", "strlen", {"OpaqueRawPointer"} -> "CSizeT"]["notapointer"]', Raises(TypeError)),
    ('w30ptr = ForeignPointerLookup["c", "abs"]; ForeignFunctionLoad[w30ptr, {"CInt"} -> "CInt"][-41]', '41'),
    ('w30ptr = ForeignPointerLookup["c", "abs"]; w30ptr["Address"] == ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"][[3]]["FunctionPointer"]["Address"]', 'True'),
    ('Head[ForeignPointerLookup["c", "strlen"]]', 'OpaqueRawPointer'),
    ('ForeignPointerLookup["c", "w30_no_such_symbol"]', Raises(ValueError)),
    ('ForeignFunctionLoad[OpaqueRawPointer[16], {"CInt"} -> "CInt"]', Raises(ValueError)),
    ('NullRawPointerQ[OpaqueRawPointer[0]]', 'True'),
    ('NullRawPointerQ[RawPointer[0, "UnsignedInteger8"]]', 'True'),
    ('NullRawPointerQ[RawMemoryExport["A string"]["Value"]]', 'False'),
    ('NullRawPointerQ[ForeignPointerLookup["c", "abs"]]', 'False'),
    ('NullRawPointerQ[5]', 'False'),
    ('OpaqueRawPointer[-1]', Raises(TypeError)),
    ('OpaqueRawPointer[1234]["Address"]', '1234'),
    ('OpaqueRawPointer[RawPointer[77, "CInt"]]', 'OpaqueRawPointer[77]'),
    ('OpaqueRawPointer[5]["Nope"]', Raises(ValueError)),
    ('RawPointer[77, "CInt"][{"Address"}]', Raises(ValueError)),
    ('RawPointer[77, "CInt"]["Type"]', '"CInt"'),
    ('RawPointer[OpaqueRawPointer[77], "CDouble"]', 'RawPointer[77, "CDouble"]'),
    ('RawPointer[-3, "CInt"]', Raises(TypeError)),
    ('RawPointer[3, "CBogus"]', Raises(ValueError)),
    ('w30p = RawMemoryAllocate["CInt"]; RawMemoryWrite[w30p, -1]; RawMemoryRead[w30p]', '-1'),
    ('w30p = RawMemoryAllocate["CInt"]; {Head[w30p], w30p["Value"][[2]]}', '{ManagedObject, "CInt"}'),
    ('w30p = RawMemoryAllocate["CFloat", 10]; Do[RawMemoryWrite[w30p, i/4, i], {i, 0, 9}]; RawMemoryImport[w30p, {"List", 10}]', '{0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25}'),
    ('RawMemoryAllocate["CInt", 0]', Raises(ValueError)),
    ('RawMemoryAllocate["CNope"]', Raises(ValueError)),
    ('w30p = RawMemoryAllocate["UnsignedInteger8"]; RawMemoryRead[w30p]', '0'),
    ('w30p = UnmanageObject[RawMemoryAllocate["UnsignedInteger8"]]; RawMemoryWrite[w30p, 42]; RawMemoryFree[w30p]', 'Null'),
    ('w30p = UnmanageObject[RawMemoryAllocate["UnsignedInteger8"]]; RawMemoryFree[w30p]; RawMemoryRead[w30p]', Raises(ValueError)),
    ('w30p = RawMemoryAllocate["UnsignedInteger8"]; RawMemoryFree[w30p]; RawMemoryFree[w30p]', Raises(ValueError)),
    ('RawMemoryFree[RawPointer[123, "CInt"]]', Raises(ValueError)),
    ('w30p = RawMemoryExport["A sample text"]; Table[RawMemoryRead[w30p, i], {i, 0, 3}]', '{65, 32, 115, 97}'),
    ('RawMemoryRead[RawPointer[12345, "CInt"]]', Raises(ValueError)),
    ('w30p = RawMemoryAllocate["CFloat", 4]; RawMemoryRead[w30p, 4]', Raises(ValueError)),
    ('RawMemoryRead[ForeignPointerLookup["c", "abs"]]', Raises(TypeError)),
    ('RawMemoryRead[RawMemoryExport[{1.5, 2.5}, "CDouble"], 1]', '2.5'),
    ('w30p = RawMemoryExport["a string"]; w30q = UnmanageObject[w30p]; RawMemoryWrite[w30q, 65]; RawMemoryImport[w30q, {"String", 8}]', '"A string"'),
    ('w30p = RawMemoryAllocate["CInt"]; RawMemoryWrite[w30p, -1] === w30p["Value"]', 'True'),
    ('RawMemoryWrite[RawMemoryAllocate["UnsignedInteger8"], 256]', Raises(ValueError)),
    ('RawMemoryWrite[RawMemoryAllocate["CInt"], "x"]', Raises(TypeError)),
    ('RawMemoryWrite[RawPointer[999, "CInt"], 1]', Raises(ValueError)),
    ('RawMemoryRead[RawMemoryExport[-1, "CInt"]]', '-1'),
    ('RawMemoryImport[RawMemoryExport["A string."], "String"]', '"A string."'),
    ('RawMemoryExport[-1, "CInt"]["Value"]["Type"]', '"CInt"'),
    ('RawMemoryImport[RawMemoryExport[ByteArray[{1, 2, 3}]], {"List", 3}]', '{1, 2, 3}'),
    ('RawMemoryImport[RawMemoryExport["café", CharacterEncoding -> "ISO8859-1"], {"ByteArray", 4}]', 'ByteArray[{99, 97, 102, 233}]'),
    ('RawMemoryExport[{1, 2}]', Raises(TypeError)),
    ('RawMemoryExport[3.5, "CInt"]', Raises(TypeError)),
    ('RawMemoryExport["x", Bogus -> 1]', Raises(ValueError)),
    ('RawMemoryImport[RawMemoryExport["A string"], {"List", 8}]', '{65, 32, 115, 116, 114, 105, 110, 103}'),
    ('RawMemoryImport[RawMemoryExport["A test."], "String"]', '"A test."'),
    ('RawMemoryImport[RawMemoryExport["héllo"], {"String", 6}]', '"héllo"'),
    ('RawMemoryImport[RawMemoryExport["abc"], {"List", 10}]', Raises(ValueError)),
    ('RawMemoryImport[RawMemoryExport["abc"], {"NumericArray", 3}]', Raises(NotImplementedError)),
    ('RawMemoryImport[RawMemoryExport["abc"], "Nope"]', Raises(ValueError)),
    ('RawMemoryImport[RawMemoryExport[ByteArray[{97, 98}]], "String"]', Raises(ValueError)),
    ('w30strchr = ForeignFunctionLoad["c", "strchr", {"OpaqueRawPointer", "CInt"} -> TypeSpecifier["RawPointer"]["UnsignedInteger8"]]; RawMemoryImport[w30strchr[RawMemoryExport["key=value"], 61], "String"]', '"=value"'),
    ('NullRawPointerQ[ForeignFunctionLoad["c", "strchr", {"OpaqueRawPointer", "CInt"} -> "OpaqueRawPointer"][RawMemoryExport["abc"], 122]]', 'True'),
    # callbacks: libc qsort with a comparison function
    ('w30arr = RawMemoryExport[{5, 3, 9, 1}, "CInt"]; w30cmp = CreateForeignCallback[Function[{a, b}, RawMemoryRead[RawPointer[a, "CInt"]] - RawMemoryRead[RawPointer[b, "CInt"]]], {"OpaqueRawPointer", "OpaqueRawPointer"} -> "CInt"]; ForeignFunctionLoad["c", "qsort", {"OpaqueRawPointer", "CSizeT", "CSizeT", "OpaqueRawPointer"} -> "Void"][w30arr, 4, 4, w30cmp]; RawMemoryImport[w30arr, {"List", 4}]', '{1, 3, 5, 9}'),
    ('w30arr = RawMemoryExport[{5, 3, 9, 1}, "CInt"]; w30cmp = CreateForeignCallback[Function[{a, b}, RawMemoryRead[RawPointer[b, "CInt"]] - RawMemoryRead[RawPointer[a, "CInt"]]], {"OpaqueRawPointer", "OpaqueRawPointer"} -> "CInt"]; ForeignFunctionLoad["c", "qsort", {"OpaqueRawPointer", "CSizeT", "CSizeT", "OpaqueRawPointer"} -> "Void"][w30arr, 4, 4, w30cmp["Value"]]; RawMemoryImport[w30arr, {"List", 4}]', '{9, 5, 3, 1}'),
    ('w30arr = RawMemoryExport[{2, 1}, "CInt"]; w30bad = CreateForeignCallback[Function[{a, b}, "x"], {"OpaqueRawPointer", "OpaqueRawPointer"} -> "CInt"]; ForeignFunctionLoad["c", "qsort", {"OpaqueRawPointer", "CSizeT", "CSizeT", "OpaqueRawPointer"} -> "Void"][w30arr, 2, 4, w30bad]', Raises(TypeError)),
    ('CreateForeignCallback[Identity, "CInt"]', Raises(TypeError)),
    ('Head[CreateForeignCallback[#^2 &, {"CLong"} -> "CLong"]["Value"]]', 'ForeignCallback'),
    ('CreateForeignCallback[#^2 &, {"CLong"} -> "CLong"]["Value"]["Function"]', '#^2 &'),
    ('CreateForeignCallback[#^2 &, {"CLong"} -> "CLong"]["Value"]["Type"]', '{"CLong"} -> "CLong"'),
    ('ForeignCallback[f, {"CLong"} -> "CLong", <|"FunctionPointer" -> OpaqueRawPointer[1]|>]', Raises(ValueError)),
    ('CreateForeignCallback[#^2 &, {"CLong"} -> "CLong"]["Value"]["Nope"]', Raises(ValueError)),
    # managed objects (Wolfram reference examples)
    ('CreateManagedObject[w30foo, Echo]["Value"]', 'w30foo'),
    ('w30man = CreateManagedObject[w30foo, Echo]; {w30man["Active"], UnmanageObject[w30man], w30man["Active"]}', '{True, w30foo, False}'),
    ('w30man = CreateManagedObject[w30foo, Echo]; CreateManagedObject[w30bar, w30man][{"BaseValue"}]', Raises(ValueError)),
    ('w30man = CreateManagedObject[w30foo, Echo]; w30m2 = CreateManagedObject[w30bar, w30man]; {w30m2["Value"], w30m2["BaseValue"], w30m2["FreeingFunction"]}', '{w30bar, w30foo, Echo}'),
    ('w30man = CreateManagedObject[w30foo, Echo]; w30man["FreeingFunction"]', 'Echo'),
    ('ManagedObject[1, 2, 3, <|"ID" -> -5|>]', Raises(ValueError)),
    ('ManagedObject[1, 2]', Raises(TypeError)),
    ('w30man = CreateManagedObject[w30foo, Echo]; UnmanageObject[w30man]; UnmanageObject[w30man]', Raises(ValueError)),
    ('UnmanageObject[5]', Raises(TypeError)),
    ('UnmanageObject[RawMemoryAllocate["CInt"]] // Head', 'RawPointer'),
    ('UnmanageObject[w30sym]', 'UnmanageObject[w30sym]'),
    ('MatchQ[CreateManagedObject[w30foo, Echo], _ManagedObject]', 'True'),
    ('MatchQ[ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"], _ForeignFunction]', 'True'),
    ('w30ff = ForeignFunctionLoad["c", "abs", {"CInt"} -> "CInt"]; (ForeignFunction @@ w30ff)[-3]', '3'),
    ('MatchQ[CreateForeignCallback[# + 1 &, {"CInt"} -> "CInt"]["Value"], _ForeignCallback]', 'True'),
    ('w30cb = CreateForeignCallback[# + 1 &, {"CInt"} -> "CInt"]["Value"]; ForeignCallback @@ w30cb === w30cb', 'True'),
]

CASES = NET + COMP + FFI


@pytest.fixture(scope="module")
def w30_servers():
    http, echo = HTTPServer(), EchoServer()
    subs = {"{P}": str(http.port), "{E}": str(echo.port), "{F}": str(free_port()),
            "{T}": tempfile.gettempdir().replace("\\", "/")}
    yield subs
    from mikoshilang.library import wl_external as W
    for lst in list(W.LISTENERS.values()):
        try:
            lst.delete()
        except Exception:
            pass
    for sock in list(W.SOCKETS.values()):
        try:
            sock.close()
        except Exception:
            pass
    http.close()
    echo.close()


def _sub(x, subs):
    if not isinstance(x, str):
        return x
    for k, v in subs.items():
        x = x.replace(k, v)
    return x


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w30(case, w30_servers):
    check(tuple(_sub(c, w30_servers) for c in case))
