'''
=====================================================
=====================================================
Copyright (c) 2018,LADYHR
All rights reserved

FileName: .py
Abstract:
This file is used to transform global variables.
=====================================================
=====================================================
'''

# !user/bin/env python
# -*- coding: utf-8 -*-

def _init():
    global _global_dict
    _global_dict = {}


def set_value(name, value):
    _global_dict[name] = value


def get_value(name, defValue=None):
    try:
        return _global_dict[name]
    except KeyError:
        return defValue