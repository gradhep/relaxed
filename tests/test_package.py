from __future__ import annotations

import relaxed as m
import relaxed._compat.typing as typing_backports


def test_version():
    assert m.__version__


def test_has_typing():
    assert hasattr(typing_backports, "TypeAlias")
    assert hasattr(typing_backports, "Self")
    assert hasattr(typing_backports, "assert_never")
