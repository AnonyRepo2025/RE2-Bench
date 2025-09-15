def get_origin_req_host(self):
    return self.get_host()

def get_host(self):
    return urlparse(self._r.url).netloc

def get_full_url(self):
    if not self._r.headers.get('Host'):
        return self._r.url
    host = self._r.headers['Host']
    parsed = urlparse(self._r.url)
    return urlunparse([parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment])

def __getitem__(self, key):
    return self._store[key.lower()][1]

def __init__(self, headers):
    self._headers = headers

def info(self):
    return self._headers

def set_cookie(self, cookie, *args, **kwargs):
    if hasattr(cookie.value, 'startswith') and cookie.value.startswith('"') and cookie.value.endswith('"'):
        cookie.value = cookie.value.replace('\\"', '')
    return super(RequestsCookieJar, self).set_cookie(cookie, *args, **kwargs)

def origin_req_host(self):
    return self.get_origin_req_host()

def __setitem__(self, key, value):
    self._store[key.lower()] = (key, value)

def __getitem__(self, key):
    val = _dict_getitem(self, key.lower())
    return ', '.join(val[1:])

def copy(self):
    p = PreparedRequest()
    p.method = self.method
    p.url = self.url
    p.headers = self.headers.copy() if self.headers is not None else None
    p._cookies = _copy_cookie_jar(self._cookies)
    p.body = self.body
    p.hooks = self.hooks
    return p

def __init__(self):
    self.method = None
    self.url = None
    self.headers = None
    self._cookies = None
    self.body = None
    self.hooks = default_hooks()

def unverifiable(self):
    return self.is_unverifiable()

def is_unverifiable(self):
    return True

def __init__(self, total=None, connect=_Default, read=_Default):
    self._connect = self._validate_timeout(connect, 'connect')
    self._read = self._validate_timeout(read, 'read')
    self.total = self._validate_timeout(total, 'total')
    self._start_connect = None



import os
import re
import time
import hashlib
import threading
from base64 import b64encode
from .compat import urlparse, str
from .cookies import extract_cookies_to_jar
from .utils import parse_dict_header, to_native_string
from .status_codes import codes
CONTENT_TYPE_FORM_URLENCODED = 'application/x-www-form-urlencoded'
CONTENT_TYPE_MULTI_PART = 'multipart/form-data'

class HTTPDigestAuth(AuthBase):

    def build_digest_header(self, method, url):
        realm = self._thread_local.chal['realm']
        nonce = self._thread_local.chal['nonce']
        qop = self._thread_local.chal.get('qop')
        algorithm = self._thread_local.chal.get('algorithm')
        opaque = self._thread_local.chal.get('opaque')
        if algorithm is None:
            _algorithm = 'MD5'
        else:
            _algorithm = algorithm.upper()
        if _algorithm == 'MD5' or _algorithm == 'MD5-SESS':

            def md5_utf8(x):
                if isinstance(x, str):
                    x = x.encode('utf-8')
                return hashlib.md5(x).hexdigest()
            hash_utf8 = md5_utf8
        elif _algorithm == 'SHA':

            def sha_utf8(x):
                if isinstance(x, str):
                    x = x.encode('utf-8')
                return hashlib.sha1(x).hexdigest()
            hash_utf8 = sha_utf8
        KD = lambda s, d: hash_utf8('%s:%s' % (s, d))
        if hash_utf8 is None:
            return None
        entdig = None
        p_parsed = urlparse(url)
        path = p_parsed.path or '/'
        if p_parsed.query:
            path += '?' + p_parsed.query
        A1 = '%s:%s:%s' % (self.username, realm, self.password)
        A2 = '%s:%s' % (method, path)
        HA1 = hash_utf8(A1)
        HA2 = hash_utf8(A2)
        if nonce == self._thread_local.last_nonce:
            self._thread_local.nonce_count += 1
        else:
            self._thread_local.nonce_count = 1
        ncvalue = '%08x' % self._thread_local.nonce_count
        s = str(self._thread_local.nonce_count).encode('utf-8')
        s += nonce.encode('utf-8')
        s += time.ctime().encode('utf-8')
        s += os.urandom(8)
        cnonce = hashlib.sha1(s).hexdigest()[:16]
        if _algorithm == 'MD5-SESS':
            HA1 = hash_utf8('%s:%s:%s' % (HA1, nonce, cnonce))
        if qop is None:
            respdig = KD(HA1, '%s:%s' % (nonce, HA2))
        elif qop == 'auth' or 'auth' in qop.split(','):
            noncebit = '%s:%s:%s:%s:%s' % (nonce, ncvalue, cnonce, 'auth', HA2)
            respdig = KD(HA1, noncebit)
        else:
            return None
        self._thread_local.last_nonce = nonce
        base = 'username="%s", realm="%s", nonce="%s", uri="%s", response="%s"' % (self.username, realm, nonce, path, respdig)
        if opaque:
            base += ', opaque="%s"' % opaque
        if algorithm:
            base += ', algorithm="%s"' % algorithm
        if entdig:
            base += ', digest="%s"' % entdig
        if qop:
            base += ', qop="auth", nc=%s, cnonce="%s"' % (ncvalue, cnonce)
        return 'Digest %s' % base

    def handle_redirect(self, r, **kwargs):
        if r.is_redirect:
            self._thread_local.num_401_calls = 1