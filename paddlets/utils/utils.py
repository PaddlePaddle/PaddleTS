#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import uuid
import hashlib

def get_uuid(prefix: str = "", suffix: str = ""):
    """
    get short UUID

    Args:
        prefix: data to be predicted

    Returns:
        prefix & 16 random letters
    """
    digits = "01234abcdefghijklmnopqrstuvwxyz56789"
    new_uuid = uuid.uuid1()
    md = hashlib.md5()
    md.update(str(new_uuid).encode())
    for i in md.digest():
        x = (i + 128) % 34
        prefix = prefix + digits[x]
    res = prefix + suffix if suffix is not None else prefix
    return res