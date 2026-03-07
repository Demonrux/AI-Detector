import unittest
from pathlib import Path
from datetime import datetime


class TestRunner(unittest.TextTestRunner):
    """Custom test runner that logs test results to a file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = Path(__file__).parent.parent / "logs" / "test.log"
        self.log_file.parent.mkdir(exist_ok=True)

    def run(self, test):
        """Run the test suite and log results to file."""
        result = super().run(test)

        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write(f"\n{'=' * 60}\n")
            file.write(f"Run tests: {datetime.now()}\n")
            file.write(f"{'=' * 60}\n")
            file.write(f"Tests run: {result.testsRun}\n")
            file.write(f"Failures: {len(result.failures)}\n")
            file.write(f"Errors: {len(result.errors)}\n")

        return result


if __name__ == "__main__":
    tests_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir))

    runner = TestRunner(verbosity=2, )
    test_result = runner.run(suite)

    print(f"\nTests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")

    exit(len(test_result.failures) + len(test_result.errors))
