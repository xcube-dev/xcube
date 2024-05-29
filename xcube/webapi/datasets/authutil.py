# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import fnmatch
from typing import Optional, Set

from xcube.server.api import ApiError

READ_ALL_DATASETS_SCOPE = "read:dataset:*"
READ_ALL_VARIABLES_SCOPE = "read:variable:*"


def assert_scopes(
    required_scopes: set[str],
    granted_scopes: Optional[set[str]],
    is_substitute: bool = False,
):
    """Assert scopes.
    Raise ServiceAuthError if one of *required_scopes* is
    not in *granted_scopes*.

    Args:
        required_scopes: The list of required scopes
        granted_scopes: The set of granted scopes. If user is not
            authenticated, its value is None.
        is_substitute: True, if the resource to be checked is a
            substitute.
    """
    missing_scope = _get_missing_scope(
        required_scopes, granted_scopes, is_substitute=is_substitute
    )
    if missing_scope is not None:
        raise ApiError.Unauthorized(f'Missing permission "{missing_scope}"')


def check_scopes(
    required_scopes: set[str],
    granted_scopes: Optional[set[str]],
    is_substitute: bool = False,
) -> bool:
    """Check scopes.

    This function is used to filter out a resource's sub-resources for
    which a given client has no permission.

    If one of *required_scopes* is not in *granted_scopes*, fail.
    If *granted_scopes* exists and *is_substitute*, fail too.
    Else succeed.

    Args:
        required_scopes: The list of required scopes
        granted_scopes: The set of granted scopes. If user is not
            authenticated, its value is None.
        is_substitute: True, if the resource to be checked is a
            substitute.

    Returns:
        True, if scopes are ok.
    """
    return (
        _get_missing_scope(required_scopes, granted_scopes, is_substitute=is_substitute)
        is None
    )


def _get_missing_scope(
    required_scopes: set[str],
    granted_scopes: Optional[set[str]],
    is_substitute: bool = False,
) -> Optional[str]:
    """Return the first required scope that is
    fulfilled by any granted scope

    Args:
        required_scopes: The list of required scopes
        granted_scopes: The set of granted scopes. If user is not
            authenticated, its value is None.
        is_substitute: True, if the resource to be checked is a
            substitute.

    Returns:
        The missing scope.
    """
    is_authenticated = granted_scopes is not None
    if is_authenticated:
        for required_scope in required_scopes:
            required_permission_given = False
            for granted_scope in granted_scopes:
                if required_scope == granted_scope or fnmatch.fnmatch(
                    required_scope, granted_scope
                ):
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
