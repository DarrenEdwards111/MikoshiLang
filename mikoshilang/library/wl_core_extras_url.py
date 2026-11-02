"""Wolfram batch X01 (part 4): URLSubmit and URLDownloadSubmit (asynchronous HTTP tasks).

The request is sent by the URLRead machinery of wl_external in a background thread; the task is a TaskObject of
the W29 task system (TaskWait, TaskRemove, Tasks, ... work on it) with TaskEnvironment "External".
"""
from __future__ import annotations

import os

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, L, is_list, Unevaluated
from .wl_common import ev, is_sym

AREA = "core extras (X01)"
MISSING = Expr("Missing", "NotAvailable")
_CHUNK = 8192

_SUBMIT_EVENTS = ("BodyReceived", "BodyChunkReceived", "ConnectionFailed", "CookiesReceived", "HeadersReceived",
                  "TaskFinished", "TaskStatusChanged")
_DOWNLOAD_EVENTS = ("BodyReceived", "ConnectionFailed", "CookiesReceived", "HeadersReceived", "TaskFinished",
                    "TaskProgress", "TaskStatusChanged")
_SUBMIT_KEYS = ("Body", "BodyByteArray", "BodyBytes", "BodyChunk", "BodyChunkByteArray", "BodyChunkBytes",
                "Headers", "Cookies", "ContentType", "StatusCode", "StatusCodeDescription", "ByteCountDownloaded",
                "ByteCountTotal", "FractionComplete", "Task", "TaskStatus", "TaskType", "TaskUUID", "EventName",
                "Failure")
_DOWNLOAD_KEYS = ("Body", "BodyByteArray", "BodyBytes", "Headers", "Cookies", "ContentType", "StatusCode",
                  "StatusCodeDescription", "File", "ByteCountDownloaded", "ByteCountTotal", "FractionComplete",
                  "Task", "TaskStatus", "TaskType", "TaskUUID", "EventName", "Failure")
_AUTO_KEY = {"TaskFinished": "TaskStatus", "TaskStatusChanged": "TaskStatus", "HeadersReceived": "Headers",
             "CookiesReceived": "Cookies", "BodyReceived": "Body", "BodyChunkReceived": "BodyChunk",
             "ConnectionFailed": "Failure", "TaskProgress": "ByteCountDownloaded"}


def _S():
    from . import wl_system as S
    return S


def _X():
    from . import wl_external as X
    return X


def _make_task_class():
    S = _S()

    class _URLTask(S._Task):
        """An External asynchronous task performing one HTTP request."""

        def __init__(self, fname, req, opts, kw, handlers, keys, download=None):
            super().__init__(Symbol("None"), "External", None, Symbol("None"), handlers, keys, False)
            self.fname = fname
            self.req = req
            self.opts = opts
            self.kw = kw
            self.download = download
            self.http = {}

        def prop(self, k):
            if k in self.http:
                return self.http[k]
            if k in _SUBMIT_KEYS or k in _DOWNLOAD_KEYS:
                if k in ("Task", "TaskStatus", "TaskType", "TaskUUID"):
                    return super().prop(k)
                return MISSING
            return super().prop(k)

        def _fire(self, event):
            self.fire(event, **{k: v for k, v in self.http.items() if k in self.keys})

        def run(self):
            X = _X()
            try:
                t, fail = X._read_one(self.req, self.opts, self.fname, **self.kw)
                if fail is not None:
                    self.http["Failure"] = fail
                    self.result = fail
                    self._fire("ConnectionFailed")
                    return
                resp = X._response_expr(t)
                props = X._resp_props(resp, self.fname)
                total = MISSING
                for hk, hv in t.headers:
                    if hk.lower() == "content-length" and hv.strip().isdigit():
                        total = int(hv.strip())
                self.http.update({"Headers": props["Headers"], "ContentType": props["ContentType"],
                                  "StatusCode": props["StatusCode"],
                                  "StatusCodeDescription": props["StatusCodeDescription"],
                                  "ByteCountTotal": total, "ByteCountDownloaded": 0})
                self._fire("HeadersReceived")
                self.http["Cookies"] = props["Cookies"]
                self._fire("CookiesReceived")
                body = t.body
                if self.download is not None:
                    with open(self.download, "wb") as fh:
                        for i in range(0, max(len(body), 1), _CHUNK):
                            part = body[i:i + _CHUNK]
                            fh.write(part)
                            self._progress(i + len(part), total)
                            self._fire("TaskProgress")
                    self.http["File"] = Expr("File", self.download)
                else:
                    from .wl_importexport import byte_array
                    for i in range(0, len(body), _CHUNK):
                        part = body[i:i + _CHUNK]
                        self.http.update({"BodyChunkByteArray": byte_array(part), "BodyChunkBytes": L(*part),
                                          "BodyChunk": _decode(part)})
                        self._progress(i + len(part), total)
                        self._fire("BodyChunkReceived")
                self._progress(len(body), total)
                self.http.update({"Body": props["Body"], "BodyByteArray": props["BodyByteArray"],
                                  "BodyBytes": props["BodyBytes"]})
                self.result = resp if self.download is None else self.http["File"]
                self._fire("BodyReceived")
            except Exception as e:  # noqa: BLE001 - the error is the task's failure
                self.http["Failure"] = Expr("Failure", "TaskError", Expr("Association", Expr(
                    "Rule", "MessageTemplate", f"{self.fname}: {e}")))
                self.result = self.http["Failure"]
                self._fire("ConnectionFailed")
            finally:
                if not self.removed:
                    self.set_status("Finished")
                    self._fire("TaskFinished")
                self.done.set()

        def _progress(self, n, total):
            self.http["ByteCountDownloaded"] = n
            self.http["FractionComplete"] = (n / total if total else 1.0) if isinstance(total, int) else MISSING

    return _URLTask


def _decode(b):
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1")


_TASK_CLS = []


def _cls():
    if not _TASK_CLS:
        _TASK_CLS.append(_make_task_class())
    return _TASK_CLS[0]


def _handler_opts(o, fname, events, allowed_keys):
    hf = o.pop("HandlerFunctions", Expr("Association"))
    if is_list(hf):
        hf = Expr("Association", *hf.args)
    if not (isinstance(hf, Expr) and hf.head == "Association"):
        raise TypeError(f"{fname}: HandlerFunctions must be an association event -> function")
    handlers = {}
    for r in hf.args:
        k = r.args[0]
        if k not in events:
            raise ValueError(f"{fname}: {k!r} is not an event of {fname} (known: {', '.join(events)})")
        handlers[k] = r.args[1]
    keys = o.pop("HandlerFunctionsKeys", Symbol("Automatic"))
    if is_sym(keys, "Automatic"):
        keys = []
        for e in handlers:
            if _AUTO_KEY[e] not in keys:
                keys.append(_AUTO_KEY[e])
        keys = keys or ["TaskStatus"]
    elif isinstance(keys, str):
        keys = [keys]
    elif is_list(keys) and all(isinstance(x, str) for x in keys.args):
        keys = list(keys.args)
    else:
        raise TypeError(f"{fname}: HandlerFunctionsKeys must be Automatic or a list of strings")
    for k in keys:
        if k not in allowed_keys:
            raise ValueError(f"{fname}: {k!r} is not a handler key (known: {', '.join(allowed_keys)})")
    return handlers, keys


def _param_kw(params, fname):
    X = _X()
    if params is None or (is_list(params) and not params.args):
        return {}
    pairs = X._rules(params, fname, "the parameters")
    if X._simple_params(pairs):
        return {"extra_query": X._flat_pairs(pairs, fname)}
    return {"body_override": params, "method_override": "POST"}


def _check_req(req, fname):
    X = _X()
    if isinstance(req, Symbol):
        raise Unevaluated()
    if isinstance(req, Expr) and req.head == "CloudObject":
        raise NotImplementedError(f"{fname}: cloud objects are not supported (there is no Wolfram Cloud)")
    if not X._is_req(req):
        raise TypeError(f"{fname}: expected a URL, an association or an HTTPRequest, not {req!r}")


_COMMON_NOTE = ("The request is sent with the URLRead machinery (same request forms, parameters as for URLExecute: "
                "simple values in the query string with GET, otherwise a POST form) in a background thread and the "
                "result is a TaskObject (TaskEnvironment \"External\", TaskType \"Asynchronous\") usable with "
                "TaskWait, TaskRemove and Tasks. HandlerFunctionsKeys -> Automatic gives the natural key of each "
                "handled event (TaskFinished/TaskStatusChanged: TaskStatus, HeadersReceived: Headers, "
                "CookiesReceived: Cookies, BodyReceived: Body, ...). Values not received yet are "
                "Missing[\"NotAvailable\"]; ByteCountTotal and FractionComplete need a Content-Length header. A "
                "connection failure fires ConnectionFailed with the Failure[\"ConnectionFailure\", ...] of URLRead "
                "under the key \"Failure\". Unknown events, keys or options raise ValueError; cloud objects "
                "NotImplementedError.")


@fn("URLSubmit", area=AREA, status="added",
    notes="URLSubmit[url], URLSubmit[url, {param -> val, ...}] and URLSubmit[HTTPRequest[...] or assoc] submit the "
          "request asynchronously. Events HeadersReceived, CookiesReceived, BodyChunkReceived (the body is read "
          "completely and then delivered in chunks of 8192 bytes, so chunks never arrive before the headers), "
          "BodyReceived, ConnectionFailed, TaskStatusChanged, TaskFinished; keys Body, BodyByteArray, BodyBytes, "
          "BodyChunk, BodyChunkByteArray, BodyChunkBytes, Headers, Cookies, ContentType, StatusCode, "
          "StatusCodeDescription, ByteCountDownloaded, ByteCountTotal, FractionComplete, Task, TaskStatus, "
          "TaskType, TaskUUID, EventName. The task's EvaluationResult is the HTTPResponse. " + _COMMON_NOTE +
          " Check: exact (round trips against a local http.server: handler data, status, body, chunks) in tests.")
def _urlsubmit(req: Expr, *rest: Expr | Option) -> Expr:
    X = _X()
    fname = "URLSubmit"
    _check_req(req, fname)
    args, o = X.split_opts(rest, X._READ_OPTS | {"HandlerFunctions", "HandlerFunctionsKeys"}, fname)
    if len(args) > 1:
        raise TypeError("URLSubmit: expected URLSubmit[url] or URLSubmit[url, params]")
    handlers, keys = _handler_opts(o, fname, _SUBMIT_EVENTS, _SUBMIT_KEYS)
    kw = _param_kw(args[0] if args else None, fname)
    t = _cls()(fname, req, o, kw, handlers, keys)
    t.start()
    return t.expr_obj()


@fn("URLDownloadSubmit", area=AREA, status="added",
    notes="URLDownloadSubmit[url, file] and URLDownloadSubmit[url, file, {param -> val, ...}] download "
          "asynchronously to file (a string or File[...]); URLDownloadSubmit[url] uses a new temporary file. "
          "Events HeadersReceived, CookiesReceived, TaskProgress (after each 8192-byte block written), "
          "BodyReceived, ConnectionFailed, TaskStatusChanged, TaskFinished; keys as for URLSubmit without the "
          "chunk keys, plus \"File\" (File[path]). The task's EvaluationResult is File[path]. " + _COMMON_NOTE +
          " Check: exact (a local http.server download compared byte for byte) in tests.")
def _urldownloadsubmit(req: Expr, *rest: Expr | Option) -> Expr:
    X = _X()
    fname = "URLDownloadSubmit"
    _check_req(req, fname)
    args, o = X.split_opts(rest, X._READ_OPTS | {"HandlerFunctions", "HandlerFunctionsKeys"}, fname)
    if len(args) > 2:
        raise TypeError("URLDownloadSubmit: expected URLDownloadSubmit[url, file] or URLDownloadSubmit[url, file, "
                        "params]")
    if args:
        f = args[0]
        if isinstance(f, Expr) and f.head == "File" and len(f.args) == 1:
            f = f.args[0]
        if not isinstance(f, str):
            raise TypeError(f"URLDownloadSubmit: the file must be a file name or File[...], not {f!r}")
        path = os.path.abspath(os.path.expanduser(f))
    else:
        import tempfile
        fd, path = tempfile.mkstemp(prefix="mikoshilang-download-")
        os.close(fd)
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        raise ValueError(f"URLDownloadSubmit: the directory {d} does not exist")
    handlers, keys = _handler_opts(o, fname, _DOWNLOAD_EVENTS, _DOWNLOAD_KEYS)
    kw = _param_kw(args[1] if len(args) > 1 else None, fname)
    t = _cls()(fname, req, o, kw, handlers, keys, download=path)
    t.start()
    return t.expr_obj()
