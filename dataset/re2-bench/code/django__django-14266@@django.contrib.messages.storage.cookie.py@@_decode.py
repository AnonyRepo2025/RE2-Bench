import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe



class CookieStorage(BaseStorage):
    cookie_name = 'messages'
    max_cookie_size = 2048
    not_finished = '__messagesnotfinished__'
    key_salt = 'django.contrib.messages'
    def _decode(self, data):
        """
        Safely decode an encoded text stream back into a list of messages.

        If the encoded text stream contained an invalid hash or was in an
        invalid format, return None.
        """
        if not data:
            return None
        try:
            return self.signer.unsign_object(data, serializer=MessageSerializer)
        # RemovedInDjango41Warning: when the deprecation ends, replace with:
        #
        # except (signing.BadSignature, json.JSONDecodeError):
        #     pass
        except signing.BadSignature:
            decoded = None
        except (binascii.Error, json.JSONDecodeError):
            decoded = self.signer.unsign(data)

        if decoded:
            # RemovedInDjango41Warning.
            try:
                return json.loads(decoded, cls=MessageDecoder)
            except json.JSONDecodeError:
                pass
        # Mark the data as used (so it gets removed) since something was wrong
        # with the data.
        self.used = True
        return None