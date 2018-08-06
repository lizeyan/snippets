import sys

import mock
try:  # for pip >= 10
    # noinspection PyProtectedMember
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
