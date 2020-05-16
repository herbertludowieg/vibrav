from .base import resource, list_resource

from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_version__ = versions['full-revisionid']
del get_versions, versions
