from codecs import open
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, tzinfo
from os import getenv, path, walk
from time import time
from typing import Any, DefaultDict, Dict, Generator, Iterable, List, Set, Tuple, Union
from uuid import uuid4
from docutils import nodes
from docutils.nodes import Element
from sphinx import addnodes, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.domains.python import pairindextypes
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging, split_index_msg, status_iterator
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogInfo, docname_to_domain
from sphinx.util.nodes import extract_messages, traverse_translatable_index
from sphinx.util.osutil import canon_path, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.template import SphinxRenderer

logger = logging.getLogger(__name__)
timestamp = time()
tzdelta = datetime.fromtimestamp(timestamp) - \
    datetime.utcfromtimestamp(timestamp)
source_date_epoch = getenv('SOURCE_DATE_EPOCH')
ltz = LocalTimeZone()

class Catalog:
    def __iter__(self) -> Generator[Message, None, None]:
        for message in self.messages:
            positions = sorted(set((source, line) for source, line, uuid
                                   in self.metadata[message]))
            uuids = [uuid for source, line, uuid in self.metadata[message]]
            yield Message(message, positions, uuids)