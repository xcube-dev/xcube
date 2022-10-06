from xcube.server.testing import ServerTestCase


class ServerTestCaseTest(ServerTestCase):

    def test_demonstrate_usage(self):
        url = f'http://localhost:{self.port}/I_do_not_exist'
        response = self.http.request('GET', url)
        self.assertEqual(404, response.status)
