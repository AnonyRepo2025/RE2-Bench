import logging
import socket
import socketserver
import sys
from wsgiref import simple_server
from django.core.exceptions import ImproperlyConfigured
from django.core.handlers.wsgi import LimitedStream
from django.core.wsgi import get_wsgi_application
from django.utils.module_loading import import_string
from django.conf import settings

__all__ = ('WSGIServer', 'WSGIRequestHandler')
logger = logging.getLogger('django.server')

def is_broken_pipe_error():
    exc_type, _, _ = sys.exc_info()
    return issubclass(exc_type, (
        BrokenPipeError,
        ConnectionAbortedError,
        ConnectionResetError,
    ))
