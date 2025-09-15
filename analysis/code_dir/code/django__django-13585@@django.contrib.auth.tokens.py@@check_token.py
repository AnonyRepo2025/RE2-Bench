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