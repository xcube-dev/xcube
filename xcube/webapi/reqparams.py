# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import abstractmethod, ABCMeta
from typing import Optional

import numpy as np

from .errors import ServiceBadRequestError
from .undefined import UNDEFINED


class RequestParams(metaclass=ABCMeta):

    @classmethod
    def to_int(cls, name: str, value: str) -> int:
        """
        Convert str value to int.
        :param name: Name of the value
        :param value: The string value
        :return: The int value
        :raise: ServiceBadRequestError
        """
        if value is None:
            raise ServiceBadRequestError(f'Parameter "{name}" must be an integer, but none was given')
        try:
            return int(value)
        except ValueError as e:
            raise ServiceBadRequestError(f'Parameter "{name}" must be an integer, but was {value!r}') from e

    @classmethod
    def to_float(cls, name: str, value: str) -> float:
        """
        Convert str value to float.
        :param name: Name of the value
        :param value: The string value
        :return: The float value
        :raise: ServiceBadRequestError
        """
        if value is None:
            raise ServiceBadRequestError(f'Parameter "{name}" must be a number, but none was given')
        try:
            return float(value)
        except ValueError as e:
            raise ServiceBadRequestError(f'Parameter "{name}" must be a number, but was {value!r}') from e

    @classmethod
    def to_datetime(cls, name: str, value: str) -> np.datetime64:
        """
        Convert str value to date/time value.
        :param name: Name of the value
        :param value: The string value
        :return: The date/time value
        :raise: ServiceBadRequestError
        """
        if value is None:
            raise ServiceBadRequestError(f'Parameter "{name}" must be a date/time, but none was given')
        try:
            return np.datetime64(value)
        except ValueError as e:
            raise ServiceBadRequestError(f'Parameter "{name}" must be a date/time, but was {value!r}') from e

    @abstractmethod
    def get_query_argument(self,
                           name: str,
                           default: Optional[str] = UNDEFINED) -> Optional[str]:
        """
        Get query argument.
        :param name: Query argument name
        :param default: Default value.
        :return: the value or none
        :raise: ServiceBadRequestError
        """

    def get_query_argument_int(self,
                               name: str,
                               default: Optional[int] = UNDEFINED) -> Optional[int]:
        """
        Get query argument of type int.
        :param name: Query argument name
        :param default: Default value.
        :return: int value
        :raise: ServiceBadRequestError
        """
        value = self.get_query_argument(name, default=default)
        return self.to_int(name, value) if value is not None else default

    def get_query_argument_float(self,
                                 name: str,
                                 default: Optional[float] = UNDEFINED) -> Optional[float]:
        """
        Get query argument of type float.
        :param name: Query argument name
        :param default: Default value.
        :return: float value
        :raise: ServiceBadRequestError
        """
        value = self.get_query_argument(name, default=default)
        return self.to_float(name, value) if value is not None else default

    def get_query_argument_datetime(self,
                                    name: str,
                                    default: Optional[np.datetime64] = UNDEFINED) -> Optional[np.datetime64]:
        """
        Get query argument of type float.
        :param name: Query argument name
        :param default: Default value.
        :return: float value
        :raise: ServiceBadRequestError
        """
        value = self.get_query_argument(name, default=default)

        return self.to_datetime(name, value) if value is not None else default
