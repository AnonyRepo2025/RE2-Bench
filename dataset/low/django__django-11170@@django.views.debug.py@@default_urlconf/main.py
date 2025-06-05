import functools
import re
import sys
import types
from pathlib import Path
from django.conf import settings
from django.http import HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import Resolver404, resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.version import get_docs_version
from django import get_version

DEBUG_ENGINE = Engine(
    debug=True,
    libraries={'i18n': 'django.templatetags.i18n'},
)
HIDDEN_SETTINGS = re.compile('API|TOKEN|KEY|SECRET|PASS|SIGNATURE', flags=re.IGNORECASE)
CLEANSED_SUBSTITUTE = '********************'
CURRENT_DIR = Path(__file__).parent

def default_urlconf(request):
    """Create an empty URLconf 404 error response."""
    with Path(CURRENT_DIR, 'templates', 'default_urlconf.html').open(encoding='utf-8') as fh:
        t = DEBUG_ENGINE.from_string(fh.read())
    c = Context({
        'version': get_docs_version(),
    })

    return HttpResponse(t.render(c), content_type='text/html')
