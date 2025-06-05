def get_lock_path(path: _AnyPurePath) -> _AnyPurePath:
    return path.joinpath(".lock")