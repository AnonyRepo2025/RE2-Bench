import socket
from django.utils.encoding import punycode

DNS_NAME = CachedDnsName()

class CachedDnsName:
    def get_fqdn(self):
        if not hasattr(self, '_fqdn'):
            self._fqdn = punycode(socket.getfqdn())
        return self._fqdn