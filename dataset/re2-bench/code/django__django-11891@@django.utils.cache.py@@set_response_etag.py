import hashlib
import re
import time
from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseNotModified
from django.utils.encoding import iri_to_uri
from django.utils.http import (
    http_date, parse_etags, parse_http_date_safe, quote_etag,
)
from django.utils.log import log_response
from django.utils.timezone import get_current_timezone_name
from django.utils.translation import get_language

cc_delim_re = re.compile(r'\s*,\s*')

def set_response_etag(response):
    if not response.streaming and response.content:
        response['ETag'] = quote_etag(hashlib.md5(response.content).hexdigest())
    return response
