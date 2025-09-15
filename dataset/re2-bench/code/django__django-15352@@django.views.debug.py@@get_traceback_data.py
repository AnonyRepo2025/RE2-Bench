import functools
import re
import sys
import types
import warnings
from pathlib import Path
from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import get_docs_version
from django import get_version

DEBUG_ENGINE = Engine(
    debug=True,
    libraries={"i18n": "django.templatetags.i18n"},
)

class ExceptionReporter:
    def _get_raw_insecure_uri(self):
        """
        Return an absolute URI from variables available in this request. Skip
        allowed hosts protection, so may return insecure URI.
        """
        return "{scheme}://{host}{path}".format(
            scheme=self.request.scheme,
            host=self.request._get_raw_host(),
            path=self.request.get_full_path(),
        )
    def get_traceback_data(self):
        """Return a dictionary containing traceback information."""
        if self.exc_type and issubclass(self.exc_type, TemplateDoesNotExist):
            self.template_does_not_exist = True
            self.postmortem = self.exc_value.chain or [self.exc_value]

        frames = self.get_traceback_frames()
        for i, frame in enumerate(frames):
            if "vars" in frame:
                frame_vars = []
                for k, v in frame["vars"]:
                    v = pprint(v)
                    # Trim large blobs of data
                    if len(v) > 4096:
                        v = "%s… <trimmed %d bytes string>" % (v[0:4096], len(v))
                    frame_vars.append((k, v))
                frame["vars"] = frame_vars
            frames[i] = frame

        unicode_hint = ""
        if self.exc_type and issubclass(self.exc_type, UnicodeError):
            start = getattr(self.exc_value, "start", None)
            end = getattr(self.exc_value, "end", None)
            if start is not None and end is not None:
                unicode_str = self.exc_value.args[1]
                unicode_hint = force_str(
                    unicode_str[max(start - 5, 0) : min(end + 5, len(unicode_str))],
                    "ascii",
                    errors="replace",
                )
        from django import get_version

        if self.request is None:
            user_str = None
        else:
            try:
                user_str = str(self.request.user)
            except Exception:
                # request.user may raise OperationalError if the database is
                # unavailable, for example.
                user_str = "[unable to retrieve the current user]"

        c = {
            "is_email": self.is_email,
            "unicode_hint": unicode_hint,
            "frames": frames,
            "request": self.request,
            "request_meta": self.filter.get_safe_request_meta(self.request),
            "request_COOKIES_items": self.filter.get_safe_cookies(self.request).items(),
            "user_str": user_str,
            "filtered_POST_items": list(
                self.filter.get_post_parameters(self.request).items()
            ),
            "settings": self.filter.get_safe_settings(),
            "sys_executable": sys.executable,
            "sys_version_info": "%d.%d.%d" % sys.version_info[0:3],
            "server_time": timezone.now(),
            "django_version_info": get_version(),
            "sys_path": sys.path,
            "template_info": self.template_info,
            "template_does_not_exist": self.template_does_not_exist,
            "postmortem": self.postmortem,
        }
        if self.request is not None:
            c["request_GET_items"] = self.request.GET.items()
            c["request_FILES_items"] = self.request.FILES.items()
            c["request_insecure_uri"] = self._get_raw_insecure_uri()
            c["raising_view_name"] = get_caller(self.request)

        # Check whether exception info is available
        if self.exc_type:
            c["exception_type"] = self.exc_type.__name__
        if self.exc_value:
            c["exception_value"] = str(self.exc_value)
        if frames:
            c["lastframe"] = frames[-1]
        return c
    def _get_source(self, filename, loader, module_name):
        source = None
        if hasattr(loader, "get_source"):
            try:
                source = loader.get_source(module_name)
            except ImportError:
                pass
            if source is not None:
                source = source.splitlines()
        if source is None:
            try:
                with open(filename, "rb") as fp:
                    source = fp.read().splitlines()
            except OSError:
                pass
        return source
    def _get_lines_from_file(
        self, filename, lineno, context_lines, loader=None, module_name=None
    ):
        """
        Return context_lines before and after lineno from file.
        Return (pre_context_lineno, pre_context, context_line, post_context).
        """
        source = self._get_source(filename, loader, module_name)
        if source is None:
            return None, [], None, []

        # If we just read the source from a file, or if the loader did not
        # apply tokenize.detect_encoding to decode the source into a
        # string, then we should do that ourselves.
        if isinstance(source[0], bytes):
            encoding = "ascii"
            for line in source[:2]:
                # File coding may be specified. Match pattern from PEP-263
                # (https://www.python.org/dev/peps/pep-0263/)
                match = re.search(rb"coding[:=]\s*([-\w.]+)", line)
                if match:
                    encoding = match[1].decode("ascii")
                    break
            source = [str(sline, encoding, "replace") for sline in source]

        lower_bound = max(0, lineno - context_lines)
        upper_bound = lineno + context_lines

        try:
            pre_context = source[lower_bound:lineno]
            context_line = source[lineno]
            post_context = source[lineno + 1 : upper_bound]
        except IndexError:
            return None, [], None, []
        return lower_bound, pre_context, context_line, post_context
    def _get_explicit_or_implicit_cause(self, exc_value):
        explicit = getattr(exc_value, "__cause__", None)
        suppress_context = getattr(exc_value, "__suppress_context__", None)
        implicit = getattr(exc_value, "__context__", None)
        return explicit or (None if suppress_context else implicit)
    def get_traceback_frames(self):
        # Get the exception and all its causes
        exceptions = []
        exc_value = self.exc_value
        while exc_value:
            exceptions.append(exc_value)
            exc_value = self._get_explicit_or_implicit_cause(exc_value)
            if exc_value in exceptions:
                warnings.warn(
                    "Cycle in the exception chain detected: exception '%s' "
                    "encountered again." % exc_value,
                    ExceptionCycleWarning,
                )
                # Avoid infinite loop if there's a cyclic reference (#29393).
                break

        frames = []
        # No exceptions were supplied to ExceptionReporter
        if not exceptions:
            return frames

        # In case there's just one exception, take the traceback from self.tb
        exc_value = exceptions.pop()
        tb = self.tb if not exceptions else exc_value.__traceback__
        while True:
            frames.extend(self.get_exception_traceback_frames(exc_value, tb))
            try:
                exc_value = exceptions.pop()
            except IndexError:
                break
            tb = exc_value.__traceback__
        return frames
    def get_exception_traceback_frames(self, exc_value, tb):
        exc_cause = self._get_explicit_or_implicit_cause(exc_value)
        exc_cause_explicit = getattr(exc_value, "__cause__", True)
        if tb is None:
            yield {
                "exc_cause": exc_cause,
                "exc_cause_explicit": exc_cause_explicit,
                "tb": None,
                "type": "user",
            }
        while tb is not None:
            # Support for __traceback_hide__ which is used by a few libraries
            # to hide internal frames.
            if tb.tb_frame.f_locals.get("__traceback_hide__"):
                tb = tb.tb_next
                continue
            filename = tb.tb_frame.f_code.co_filename
            function = tb.tb_frame.f_code.co_name
            lineno = tb.tb_lineno - 1
            loader = tb.tb_frame.f_globals.get("__loader__")
            module_name = tb.tb_frame.f_globals.get("__name__") or ""
            (
                pre_context_lineno,
                pre_context,
                context_line,
                post_context,
            ) = self._get_lines_from_file(
                filename,
                lineno,
                7,
                loader,
                module_name,
            )
            if pre_context_lineno is None:
                pre_context_lineno = lineno
                pre_context = []
                context_line = "<source code not available>"
                post_context = []
            yield {
                "exc_cause": exc_cause,
                "exc_cause_explicit": exc_cause_explicit,
                "tb": tb,
                "type": "django" if module_name.startswith("django.") else "user",
                "filename": filename,
                "function": function,
                "lineno": lineno + 1,
                "vars": self.filter.get_traceback_frame_variables(
                    self.request, tb.tb_frame
                ),
                "id": id(tb),
                "pre_context": pre_context,
                "context_line": context_line,
                "post_context": post_context,
                "pre_context_lineno": pre_context_lineno + 1,
            }
            tb = tb.tb_next