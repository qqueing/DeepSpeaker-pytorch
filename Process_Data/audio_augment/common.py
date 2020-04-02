#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: common.py
@Time: 2020/4/2 11:31 AM
@Overview:
"""
import subprocess


def RunCommand(command):
    """ Runs commands frequently seen in scripts. These are usually a
        sequence of commands connected by pipes, so we use shell=True """
    # logger.info("Running the command\n{0}".format(command))
    if command.endswith('|'):
        command = command.rstrip('|')

    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    # p.wait()
    [stdout, stderr] = p.communicate()

    return p.pid, stdout, stderr
