import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.safestring import SafeData, mark_safe



class CookieStorage(BaseStorage):
    cookie_name = 'messages'
    max_cookie_size = 2048
    not_finished = '__messagesnotfinished__'
    key_salt = 'django.contrib.messages'
    def _encode(self, messages, encode_empty=False):
        """
        Return an encoded version of the messages list which can be stored as
        plain text.

        Since the data will be retrieved from the client-side, the encoded data
        also contains a hash to ensure that the data was not tampered with.
        """
        if messages or encode_empty:
            return self.signer.sign_object(messages, serializer=MessageSerializer, compress=True)