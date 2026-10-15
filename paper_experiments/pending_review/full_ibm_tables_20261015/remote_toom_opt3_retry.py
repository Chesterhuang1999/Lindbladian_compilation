"""Separate user-authorized Toom-L6 retry, preserving completed TFIM retries."""
import remote_opt3_retry as retry

if __name__ == '__main__':
    retry.CASES = ('Toom-L6',)
    retry.coordinate('remote_toom_opt3_retry_status.json')
