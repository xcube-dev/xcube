import unittest

from tornado.web import HTTPError

from xcube.webapi.errors import ServiceError, ServiceConfigError, ServiceBadRequestError, ServiceResourceNotFoundError


class ErrorsTest(unittest.TestCase):
    def test_same_base_type(self):

        self.assertIsInstance(ServiceError(''), HTTPError)
        self.assertEqual(500, ServiceError('').status_code)
        self.assertEqual(503, ServiceError('', status_code=503).status_code)

        self.assertIsInstance(ServiceConfigError(''), ServiceError)
        self.assertEqual(500, ServiceConfigError('').status_code)

        self.assertIsInstance(ServiceBadRequestError(''), ServiceError)
        self.assertEqual(400, ServiceBadRequestError('').status_code)

        self.assertIsInstance(ServiceResourceNotFoundError(''), ServiceError)
        self.assertEqual(404, ServiceResourceNotFoundError('').status_code)
