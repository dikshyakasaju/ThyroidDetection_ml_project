import unittest

from main import app
import os


class TestToPerform(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        print(self.app)

    def tearDown(self):
        pass

    def test_page(self):
        response = self.app.get('/', follow_redirects=True)
        print(response)
        # Checking if response is equal to 200
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
