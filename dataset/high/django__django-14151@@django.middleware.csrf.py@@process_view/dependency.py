## django.http.request.HttpRequest.get_host
def get_host(self):
    host = self._get_raw_host()

    # Allow variants of localhost if ALLOWED_HOSTS is empty and DEBUG=True.
    allowed_hosts = settings.ALLOWED_HOSTS
    if settings.DEBUG and not allowed_hosts:
        allowed_hosts = ['.localhost', '127.0.0.1', '[::1]']

    domain, port = split_domain_port(host)
    if domain and validate_host(domain, allowed_hosts):
        return host
    else:
        msg = "Invalid HTTP_HOST header: %r." % host
        if domain:
            msg += " You may need to add %r to ALLOWED_HOSTS." % domain
        else:
            msg += " The domain name provided is not valid according to RFC 1034/1035."
        raise DisallowedHost(msg)

## django.http.request.HttpRequest._get_raw_host
def _get_raw_host(self):
    if settings.USE_X_FORWARDED_HOST and (
            'HTTP_X_FORWARDED_HOST' in self.META):
        host = self.META['HTTP_X_FORWARDED_HOST']
    elif 'HTTP_HOST' in self.META:
        host = self.META['HTTP_HOST']
    else:
        # Reconstruct the host using the algorithm from PEP 333.
        host = self.META['SERVER_NAME']
        server_port = self.get_port()
        if server_port != ('443' if self.is_secure() else '80'):
            host = '%s:%s' % (host, server_port)
    return host

## django.http.request.split_domain_port
def split_domain_port(host):
    host = host.lower()

    if not host_validation_re.match(host):
        return '', ''

    if host[-1] == ']':
        # It's an IPv6 address without a port.
        return host, ''
    bits = host.rsplit(':', 1)
    domain, port = bits if len(bits) == 2 else (bits[0], '')
    # Remove a trailing dot (if present) from the domain.
    domain = domain[:-1] if domain.endswith('.') else domain
    return domain, port