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

class SafeExceptionReporterFilter:
    """
    Use annotations made by the sensitive_post_parameters and
    sensitive_variables decorators to filter out sensitive information.
    """

    cleansed_substitute = "********************"
    hidden_settings = _lazy_re_compile(
        "API|TOKEN|KEY|SECRET|PASS|SIGNATURE|HTTP_COOKIE", flags=re.I
    )

    def cleanse_setting(self, key, value):
        """
        Cleanse an individual setting key/value of sensitive content. If the
        value is a dictionary, recursively cleanse the keys in that dictionary.
        """
        if key == settings.SESSION_COOKIE_NAME:
            is_sensitive = True
        else:
            try:
                is_sensitive = self.hidden_settings.search(key)
            except TypeError:
                is_sensitive = False

        if is_sensitive:
            cleansed = self.cleansed_substitute
        elif isinstance(value, dict):
            cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            cleansed = [self.cleanse_setting("", v) for v in value]
        elif isinstance(value, tuple):
            cleansed = tuple([self.cleanse_setting("", v) for v in value])
        else:
            cleansed = value

        if callable(cleansed):
            cleansed = CallableSettingWrapper(cleansed)

        return cleansed

    def get_safe_settings(self):
        """
        Return a dictionary of the settings module with values of sensitive
        settings replaced with stars (*********).
        """
        settings_dict = {}
        for k in dir(settings):
            if k.isupper():
                settings_dict[k] = self.cleanse_setting(k, getattr(settings, k))
        return settings_dict

    def get_safe_request_meta(self, request):
        """
        Return a dictionary of request.META with sensitive values redacted.
        """
        if not hasattr(request, "META"):
            return {}
        return {k: self.cleanse_setting(k, v) for k, v in request.META.items()}

    def get_safe_cookies(self, request):
        """
        Return a dictionary of request.COOKIES with sensitive values redacted.
        """
        if not hasattr(request, "COOKIES"):
            return {}
        return {k: self.cleanse_setting(k, v) for k, v in request.COOKIES.items()}

    def is_active(self, request):
        """
        This filter is to add safety in production environments (i.e. DEBUG
        is False). If DEBUG is True then your site is not safe anyway.
        This hook is provided as a convenience to easily activate or
        deactivate the filter on a per request basis.
        """
        return settings.DEBUG is False

    def get_cleansed_multivaluedict(self, request, multivaluedict):
        """
        Replace the keys in a MultiValueDict marked as sensitive with stars.
        This mitigates leaking sensitive POST parameters if something like
        request.POST['nonexistent_key'] throws an exception (#21098).
        """
        sensitive_post_parameters = getattr(request, "sensitive_post_parameters", [])
        if self.is_active(request) and sensitive_post_parameters:
            multivaluedict = multivaluedict.copy()
            for param in sensitive_post_parameters:
                if param in multivaluedict:
                    multivaluedict[param] = self.cleansed_substitute
        return multivaluedict

    def get_post_parameters(self, request):
        """
        Replace the values of POST parameters marked as sensitive with
        stars (*********).
        """
        if request is None:
            return {}
        else:
            sensitive_post_parameters = getattr(
                request, "sensitive_post_parameters", []
            )
            if self.is_active(request) and sensitive_post_parameters:
                cleansed = request.POST.copy()
                if sensitive_post_parameters == "__ALL__":
                    # Cleanse all parameters.
                    for k in cleansed:
                        cleansed[k] = self.cleansed_substitute
                    return cleansed
                else:
                    # Cleanse only the specified parameters.
                    for param in sensitive_post_parameters:
                        if param in cleansed:
                            cleansed[param] = self.cleansed_substitute
                    return cleansed
            else:
                return request.POST

    def cleanse_special_types(self, request, value):
        try:
            # If value is lazy or a complex object of another kind, this check
            # might raise an exception. isinstance checks that lazy
            # MultiValueDicts will have a return value.
            is_multivalue_dict = isinstance(value, MultiValueDict)
        except Exception as e:
            return "{!r} while evaluating {!r}".format(e, value)

        if is_multivalue_dict:
            # Cleanse MultiValueDicts (request.POST is the one we usually care about)
            value = self.get_cleansed_multivaluedict(request, value)
        return value

    def get_traceback_frame_variables(self, request, tb_frame):
        """
        Replace the values of variables marked as sensitive with
        stars (*********).
        """
        # Loop through the frame's callers to see if the sensitive_variables
        # decorator was used.
        current_frame = tb_frame.f_back
        sensitive_variables = None
        while current_frame is not None:
            if (
                current_frame.f_code.co_name == "sensitive_variables_wrapper"
                and "sensitive_variables_wrapper" in current_frame.f_locals
            ):
                # The sensitive_variables decorator was used, so we take note
                # of the sensitive variables' names.
                wrapper = current_frame.f_locals["sensitive_variables_wrapper"]
                sensitive_variables = getattr(wrapper, "sensitive_variables", None)
                break
            current_frame = current_frame.f_back

        cleansed = {}
        if self.is_active(request) and sensitive_variables:
            if sensitive_variables == "__ALL__":
                # Cleanse all variables
                for name in tb_frame.f_locals:
                    cleansed[name] = self.cleansed_substitute
            else:
                # Cleanse specified variables
                for name, value in tb_frame.f_locals.items():
                    if name in sensitive_variables:
                        value = self.cleansed_substitute
                    else:
                        value = self.cleanse_special_types(request, value)
                    cleansed[name] = value
        else:
            # Potentially cleanse the request and any MultiValueDicts if they
            # are one of the frame variables.
            for name, value in tb_frame.f_locals.items():
                cleansed[name] = self.cleanse_special_types(request, value)

        if (
            tb_frame.f_code.co_name == "sensitive_variables_wrapper"
            and "sensitive_variables_wrapper" in tb_frame.f_locals
        ):
            # For good measure, obfuscate the decorated function's arguments in
            # the sensitive_variables decorator's frame, in case the variables
            # associated with those arguments were meant to be obfuscated from
            # the decorated function's frame.
            cleansed["func_args"] = self.cleansed_substitute
            cleansed["func_kwargs"] = self.cleansed_substitute

        return cleansed.items()
