def assert_positive_integer(**kwargs):
    for key, value in kwargs.items():
        ok = False
        try:
            ok = int(value) > 0
        except ValueError:
            pass
        finally:
            if not ok:
                raise ValueError(f"{key} should be positive integer, but it is {value}")
        return True
