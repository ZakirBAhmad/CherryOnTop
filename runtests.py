import unittest
import sys
from datetime import datetime
import os

def run_tests():
    # Create test directory if it doesn't exist
    if not os.path.exists('test_reports'):
        os.makedirs('test_reports')
    
    # Get current timestamp for report filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'test_reports/test_report_{timestamp}.txt'
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests and capture output
    with open(report_file, 'w') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        result = runner.run(suite)
        
        # Write summary
        f.write('\n\n=== Test Summary ===\n')
        f.write(f'Tests Run: {result.testsRun}\n')
        f.write(f'Failures: {len(result.failures)}\n')
        f.write(f'Errors: {len(result.errors)}\n')
        f.write(f'Skipped: {len(result.skipped)}\n')
        
        # Write detailed failure information
        if result.failures:
            f.write('\n=== Failures ===\n')
            for failure in result.failures:
                f.write(f'\n{failure[0]}\n')
                f.write(f'{failure[1]}\n')
        
        # Write detailed error information
        if result.errors:
            f.write('\n=== Errors ===\n')
            for error in result.errors:
                f.write(f'\n{error[0]}\n')
                f.write(f'{error[1]}\n')
    
    # Print summary to console
    print(f'\nTest report written to: {report_file}')
    print(f'Tests Run: {result.testsRun}')
    print(f'Failures: {len(result.failures)}')
    print(f'Errors: {len(result.errors)}')
    print(f'Skipped: {len(result.skipped)}')
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())
