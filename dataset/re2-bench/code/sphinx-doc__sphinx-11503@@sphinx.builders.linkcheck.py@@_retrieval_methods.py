from __future__ import annotations
import json
import re
import socket
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from os import path
from queue import PriorityQueue, Queue
from threading import Thread
from typing import TYPE_CHECKING, NamedTuple, cast
from urllib.parse import unquote, urlparse, urlsplit, urlunparse
from docutils import nodes
from requests.exceptions import ConnectionError, HTTPError, SSLError, TooManyRedirects
from sphinx.builders.dummy import DummyBuilder
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import encode_uri, logging, requests
from sphinx.util.console import darkgray, darkgreen, purple, red, turquoise  # type: ignore
from sphinx.util.nodes import get_node_line
from typing import Any, Callable, Generator, Iterator
from requests import Response
from sphinx.application import Sphinx
from sphinx.config import Config

logger = logging.getLogger(__name__)
uri_re = re.compile('([a-z]+:)?//')  # matches to foo:// and // (a protocol relative URL)
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
}
CHECK_IMMEDIATELY = 0
QUEUE_POLL_SECS = 1
DEFAULT_DELAY = 60.0

class HyperlinkAvailabilityCheckWorker(Thread):
    def _retrieval_methods(self,
                           check_anchors: bool,
                           anchor: str) -> Iterator[tuple[Callable, dict]]:
        if not check_anchors or not anchor:
            yield self._session.head, {'allow_redirects': True}
        yield self._session.get, {'stream': True}