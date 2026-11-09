import pytest

import crane


def test_lazy_load_exports_writers():
    from crane.json import JsonDatasetWriter

    assert crane.JsonDatasetWriter is JsonDatasetWriter


def test_lazy_load_unknown_name_is_attribute_error():
    with pytest.raises(AttributeError):
        crane.no_such_name  # noqa: B018
    # what `getattr` with a default and `hasattr` rely on
    assert getattr(crane, "no_such_name", None) is None
    assert not hasattr(crane, "no_such_name")
