import errno
import os
import os.path as osp
import sys


# class Logger(object):
#     def __init__(self, log_dir):
#         # clean previous logged data under the same directory name
#         self._remove(log_dir)
#
#         # configure the project
#         configure(log_dir)
#
#         self.global_step = 0
#
#     def log_value(self, name, value):
#         log_value(name, value, self.global_step)
#         return self
#
#     def step(self):
#         self.global_step += 1
#
#     @staticmethod
#     def _remove(path):
#         """ param <path> could either be relative or absolute. """
#         if os.path.isfile(path):
#             os.remove(path)  # remove the file
#         elif os.path.isdir(path):
#             import shutil
#             shutil.rmtree(path)  # remove dir and all contains


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class NewLogger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None, mode='a'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
