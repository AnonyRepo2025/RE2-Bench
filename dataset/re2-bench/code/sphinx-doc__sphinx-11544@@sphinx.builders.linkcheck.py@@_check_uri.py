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
    def _check(self, docname: str, uri: str, hyperlink: Hyperlink) -> tuple[str, str, int]:
        # check for various conditions without bothering the network

        for doc_matcher in self.documents_exclude:
            if doc_matcher.match(docname):
                info = (
                    f'{docname} matched {doc_matcher.pattern} from '
                    'linkcheck_exclude_documents'
                )
                return 'ignored', info, 0

        if len(uri) == 0 or uri.startswith(('#', 'mailto:', 'tel:')):
            return 'unchecked', '', 0
        if not uri.startswith(('http:', 'https:')):
            if uri_re.match(uri):
                # Non-supported URI schemes (ex. ftp)
                return 'unchecked', '', 0

            src_dir = path.dirname(hyperlink.docpath)
            if path.exists(path.join(src_dir, uri)):
                return 'working', '', 0
            return 'broken', '', 0

        # need to actually check the URI
        status, info, code = '', '', 0
        for _ in range(self.retries):
            status, info, code = self._check_uri(uri, hyperlink)
            if status != 'broken':
                break

        return status, info, code
    def _retrieval_methods(self,
                           check_anchors: bool,
                           anchor: str) -> Iterator[tuple[Callable, dict]]:
        if not check_anchors or not anchor:
            yield self._session.head, {'allow_redirects': True}
        yield self._session.get, {'stream': True}
    def _check_uri(self, uri: str, hyperlink: Hyperlink) -> tuple[str, str, int]:
        req_url, delimiter, anchor = uri.partition('#')
        if delimiter and anchor:
            for rex in self.anchors_ignore:
                if rex.match(anchor):
                    anchor = ''
                    break
            else:
                for rex in self.anchors_ignore_for_url:
                    if rex.match(req_url):
                        anchor = ''
                        break

        # handle non-ASCII URIs
        try:
            req_url.encode('ascii')
        except UnicodeError:
            req_url = encode_uri(req_url)

        # Get auth info, if any
        for pattern, auth_info in self.auth:  # noqa: B007 (false positive)
            if pattern.match(uri):
                break
        else:
            auth_info = None

        # update request headers for the URL
        headers = _get_request_headers(uri, self.request_headers)

        # Linkcheck HTTP request logic:
        #
        # - Attempt HTTP HEAD before HTTP GET unless page content is required.
        # - Follow server-issued HTTP redirects.
        # - Respect server-issued HTTP 429 back-offs.
        error_message = ''
        status_code = -1
        response_url = retry_after = ''
        for retrieval_method, kwargs in self._retrieval_methods(self.check_anchors, anchor):
            try:
                with retrieval_method(
                    url=req_url, auth=auth_info,
                    headers=headers,
                    timeout=self.timeout,
                    **kwargs,
                    _user_agent=self.user_agent,
                    _tls_info=(self.tls_verify, self.tls_cacerts),
                ) as response:
                    if (self.check_anchors and response.ok and anchor
                            and not contains_anchor(response, anchor)):
                        raise Exception(__(f'Anchor {anchor!r} not found'))

                # Copy data we need from the (closed) response
                status_code = response.status_code
                redirect_status_code = response.history[-1].status_code if response.history else None  # NoQA: E501
                retry_after = response.headers.get('Retry-After')
                response_url = f'{response.url}'
                response.raise_for_status()
                del response
                break

            except SSLError as err:
                # SSL failure; report that the link is broken.
                return 'broken', str(err), 0

            except (ConnectionError, TooManyRedirects) as err:
                # Servers drop the connection on HEAD requests, causing
                # ConnectionError.
                error_message = str(err)
                continue

            except HTTPError as err:
                error_message = str(err)

                # Unauthorised: the reference probably exists
                if status_code == 401:
                    return 'working', 'unauthorized', 0

                # Rate limiting; back-off if allowed, or report failure otherwise
                if status_code == 429:
                    if next_check := self.limit_rate(response_url, retry_after):
                        self.wqueue.put(CheckRequest(next_check, hyperlink), False)
                        return 'rate-limited', '', 0
                    return 'broken', error_message, 0

                # Don't claim success/failure during server-side outages
                if status_code == 503:
                    return 'ignored', 'service unavailable', 0

                # For most HTTP failures, continue attempting alternate retrieval methods
                continue

            except Exception as err:
                # Unhandled exception (intermittent or permanent); report that
                # the link is broken.
                return 'broken', str(err), 0

        else:
            # All available retrieval methods have been exhausted; report
            # that the link is broken.
            return 'broken', error_message, 0

        # Success; clear rate limits for the origin
        netloc = urlsplit(req_url).netloc
        self.rate_limits.pop(netloc, None)

        if ((response_url.rstrip('/') == req_url.rstrip('/'))
                or _allowed_redirect(req_url, response_url,
                                     self.allowed_redirects)):
            return 'working', '', 0
        elif redirect_status_code is not None:
            return 'redirected', response_url, redirect_status_code
        else:
            return 'redirected', response_url, 0
    def limit_rate(self, response_url: str, retry_after: str) -> float | None:
        delay = DEFAULT_DELAY
        next_check = None
        if retry_after:
            try:
                # Integer: time to wait before next attempt.
                delay = float(retry_after)
            except ValueError:
                try:
                    # An HTTP-date: time of next attempt.
                    until = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    # TypeError: Invalid date format.
                    # ValueError: Invalid date, e.g. Oct 52th.
                    pass
                else:
                    next_check = datetime.timestamp(until)
                    delay = (until - datetime.now(timezone.utc)).total_seconds()
            else:
                next_check = time.time() + delay
        netloc = urlsplit(response_url).netloc
        if next_check is None:
            max_delay = self.rate_limit_timeout
            try:
                rate_limit = self.rate_limits[netloc]
            except KeyError:
                delay = DEFAULT_DELAY
            else:
                last_wait_time = rate_limit.delay
                delay = 2.0 * last_wait_time
                if delay > max_delay > last_wait_time:
                    delay = max_delay
            if delay > max_delay:
                return None
            next_check = time.time() + delay
        self.rate_limits[netloc] = RateLimit(delay, next_check)
        return next_check