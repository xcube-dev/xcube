# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import fnmatch
from typing import Optional, Set

from xcube.server.api import ApiError

READ_ALL_DATASETS_SCOPE = 'read:dataset:*'
READ_ALL_VARIABLES_SCOPE = 'read:variable:*'


def assert_scopes(required_scopes: Set[str],
                  granted_scopes: Optional[Set[str]],
                  is_substitute: bool = False):
    """
    Assert scopes.
    Raise ServiceAuthError if one of *required_scopes* is
    not in *granted_scopes*.

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes.
        If user is not authenticated, its value is None.
    :param is_substitute: True, if the resource to be checked
        is a substitute.
    """
    missing_scope = _get_missing_scope(required_scopes,
                                       granted_scopes,
                                       is_substitute=is_substitute)
    if missing_scope is not None:
        raise ApiError.Unauthorized(
            f'Missing permission "{missing_scope}"'
        )


def check_scopes(required_scopes: Set[str],
                 granted_scopes: Optional[Set[str]],
                 is_substitute: bool = False) -> bool:
    """
    Check scopes.

    This function is used to filter out a resource's sub-resources for
    which a given client has no permission.

    If one of *required_scopes* is not in *granted_scopes*, fail.
    If *granted_scopes* exists and *is_substitute*, fail too.
    Else succeed.

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes.
        If user is not authenticated, its value is None.
    :param is_substitute: True, if the resource to be checked
        is a substitute.
    :return: True, if scopes are ok.
    """
    return _get_missing_scope(required_scopes,
                              granted_scopes,
                              is_substitute=is_substitute) is None


def _get_missing_scope(required_scopes: Set[str],
                       granted_scopes: Optional[Set[str]],
                       is_substitute: bool = False) -> Optional[str]:
    """
    Return the first required scope that is
    fulfilled by any granted scope

    :param required_scopes: The list of required scopes
    :param granted_scopes: The set of granted scopes.
        If user is not authenticated, its value is None.
    :param is_substitute: True, if the resource to be checked
        is a substitute.
    :return: The missing scope.
    """
    is_authenticated = granted_scopes is not None
    if is_authenticated:
        for required_scope in required_scopes:
            required_permission_given = False
            for granted_scope in granted_scopes:
                if required_scope == granted_scope \
                        or fnmatch.fnmatch(required_scope, granted_scope):
                    # If any granted scope matches, we can stop
                    required_permission_given = True
                    break
            if not required_permission_given:
                # The required scope is not a granted scope --> fail
                return required_scope

        # If we end here, required_scopes are either empty or satisfied
        if is_substitute:
            # All required scopes are satisfied, now fail for
            # substitute resources (e.g. demo resources) as there
            # is usually a better (non-demo) resource that replaces it.
            # Return missing scope (dummy, not used) --> fail
            return READ_ALL_DATASETS_SCOPE

    elif required_scopes:
        # We require scopes but have no granted scopes
        if not is_substitute:
            # ...and resource is not a substitute --> fail
            return next(iter(required_scopes))

    # All ok.
    return None
