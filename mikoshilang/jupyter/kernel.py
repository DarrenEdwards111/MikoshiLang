"""MikoshiLang Jupyter kernel implementation."""

from __future__ import annotations
import io
import base64
import traceback

from ipykernel.kernelbase import Kernel

from ..parser import parse, ParseError
from ..evaluate import evaluate
from ..latex import to_latex
from ..expr import Expr, Symbol


class MikoshiLangKernel(Kernel):
    implementation = "MikoshiLang"
    implementation_version = "0.2.0"
    language = "mikoshilang"
    language_version = "0.2.0"
    language_info = {
        "name": "mikoshilang",
        "mimetype": "text/x-mikoshilang",
        "file_extension": ".ml",
    }
    banner = "MikoshiLang v0.2.0 — Symbolic Computation Language\nBuilt by Mikoshi Ltd"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env = {}
        self._counter = 0

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        self._counter += 1
        code = code.strip()
        if not code:
            return {"status": "ok", "execution_count": self._counter, "payload": [], "user_expressions": {}}

        try:
            expr = parse(code)
            result = evaluate(expr)
        except Exception as e:
            if not silent:
                self.send_response(self.iopub_socket, "stream", {
                    "name": "stderr",
                    "text": f"{type(e).__name__}: {e}\n",
                })
            return {"status": "error", "execution_count": self._counter,
                    "ename": type(e).__name__, "evalue": str(e), "traceback": traceback.format_exc().splitlines()}

        if not silent and result is not None:
            # Check if result is a matplotlib figure
            try:
                import matplotlib.figure
                if isinstance(result, matplotlib.figure.Figure):
                    buf = io.BytesIO()
                    result.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    img_data = base64.b64encode(buf.read()).decode("ascii")
                    self.send_response(self.iopub_socket, "display_data", {
                        "data": {"image/png": img_data},
                        "metadata": {},
                    })
                    import matplotlib.pyplot as plt
                    plt.close(result)
                    return {"status": "ok", "execution_count": self._counter, "payload": [], "user_expressions": {}}
            except ImportError:
                pass

            # LaTeX + text display
            latex_str = None
            if isinstance(result, (Expr, Symbol)):
                try:
                    latex_str = "$" + to_latex(result) + "$"
                except Exception:
                    latex_str = None

            data = {"text/plain": str(result)}
            if latex_str:
                data["text/latex"] = latex_str

            self.send_response(self.iopub_socket, "display_data", {
                "data": data,
                "metadata": {},
            })

        return {"status": "ok", "execution_count": self._counter, "payload": [], "user_expressions": {}}

    def do_is_complete(self, code):
        # Simple heuristic
        try:
            parse(code)
            return {"status": "complete"}
        except Exception:
            return {"status": "incomplete", "indent": "  "}

    def do_complete(self, code, cursor_pos):
        return {"matches": [], "cursor_start": cursor_pos, "cursor_end": cursor_pos, "status": "ok"}
