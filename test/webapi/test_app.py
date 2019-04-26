import unittest


class AppSmokeTest(unittest.TestCase):

    def test_start_stop_service(self):
        pass
        # TODO: The following test code will cause timeouts in test/test_handlers.py - why?
        # service = new_service(args=['--port', '20001', '--update', '0'])
        # self.assertIsInstance(service, Service)
        # service.stop()
        # IOLoop.current().call_later(0.1, service.stop)
        # service.start()
