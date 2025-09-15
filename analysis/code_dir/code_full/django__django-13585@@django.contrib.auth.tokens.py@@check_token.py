def base36_to_int(s):
    if len(s) > 13:
        raise ValueError('Base36 input too large')
    return int(s, 36)

def int_to_base36(i):
    char_set = '0123456789abcdefghijklmnopqrstuvwxyz'
    if i < 0:
        raise ValueError('Negative base36 conversion input.')
    if i < 36:
        return char_set[i]
    b36 = ''
    while i != 0:
        i, n = divmod(i, 36)
        b36 = char_set[n] + b36
    return b36

def get_email_field_name(cls):
    try:
        return cls.EMAIL_FIELD
    except AttributeError:
        return 'email'

def _get_pk_val(self, meta=None):
    meta = meta or self._meta
    return getattr(self, meta.pk.attname)

def salted_hmac(key_salt, value, secret=None, *, algorithm='sha1'):
    if secret is None:
        secret = settings.SECRET_KEY
    key_salt = force_bytes(key_salt)
    secret = force_bytes(secret)
    try:
        hasher = getattr(hashlib, algorithm)
    except AttributeError as e:
        raise InvalidAlgorithm('%r is not an algorithm accepted by the hashlib module.' % algorithm) from e
    key = hasher(key_salt + secret).digest()
    return hmac.new(key, msg=force_bytes(value), digestmod=hasher)

def force_bytes(s, encoding='utf-8', strings_only=False, errors='strict'):
    if isinstance(s, bytes):
        if encoding == 'utf-8':
            return s
        else:
            return s.decode('utf-8', errors).encode(encoding, errors)
    if strings_only and is_protected_type(s):
        return s
    if isinstance(s, memoryview):
        return bytes(s)
    return str(s).encode(encoding, errors)

def constant_time_compare(val1, val2):
    return secrets.compare_digest(force_bytes(val1), force_bytes(val2))

def __getattr__(self, name):
    if self._wrapped is empty:
        self._setup(name)
    val = getattr(self._wrapped, name)
    if name in {'MEDIA_URL', 'STATIC_URL'} and val is not None:
        val = self._add_script_prefix(val)
    elif name == 'SECRET_KEY' and (not val):
        raise ImproperlyConfigured('The SECRET_KEY setting must not be empty.')
    self.__dict__[name] = val
    return val

def __getattr__(self, name):
    if not name.isupper() or name in self._deleted:
        raise AttributeError
    return getattr(self.default_settings, name)



from datetime import datetime, time
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.http import base36_to_int, int_to_base36
default_token_generator = PasswordResetTokenGenerator()

class PasswordResetTokenGenerator:
    key_salt = 'django.contrib.auth.tokens.PasswordResetTokenGenerator'
    algorithm = None
    secret = None

    def check_token(self, user, token):
        if not (user and token):
            return False
        try:
            ts_b36, _ = token.split('-')
            legacy_token = len(ts_b36) < 4
        except ValueError:
            return False
        try:
            ts = base36_to_int(ts_b36)
        except ValueError:
            return False
        if not constant_time_compare(self._make_token_with_timestamp(user, ts), token):
            if not constant_time_compare(self._make_token_with_timestamp(user, ts, legacy=True), token):
                return False
        now = self._now()
        if legacy_token:
            ts *= 24 * 60 * 60
            ts += int((now - datetime.combine(now.date(), time.min)).total_seconds())
        if self._num_seconds(now) - ts > settings.PASSWORD_RESET_TIMEOUT:
            return False
        return True

    def _make_token_with_timestamp(self, user, timestamp, legacy=False):
        ts_b36 = int_to_base36(timestamp)
        hash_string = salted_hmac(self.key_salt, self._make_hash_value(user, timestamp), secret=self.secret, algorithm='sha1' if legacy else self.algorithm).hexdigest()[::2]
        return '%s-%s' % (ts_b36, hash_string)

    def _make_hash_value(self, user, timestamp):
        login_timestamp = '' if user.last_login is None else user.last_login.replace(microsecond=0, tzinfo=None)
        email_field = user.get_email_field_name()
        email = getattr(user, email_field, '') or ''
        return f'{user.pk}{user.password}{login_timestamp}{timestamp}{email}'

    def _num_seconds(self, dt):
        return int((dt - datetime(2001, 1, 1)).total_seconds())

    def _now(self):
        return datetime.now()