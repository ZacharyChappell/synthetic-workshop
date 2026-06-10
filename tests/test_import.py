import synthworkshop


def test_package_imports() -> None:
    assert isinstance(synthworkshop.version, str)
    assert synthworkshop.GridSpec is not None
    assert synthworkshop.RenderedScene is not None
