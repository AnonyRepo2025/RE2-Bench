def __getattribute__(self, name):
    if name == "_wrapped":
        return super().__getattribute__(name)
    value = super().__getattribute__(name)
    if not getattr(value, "_mask_wrapped", True):
        raise AttributeError
    return value

def __getattr__(self, name):
    """Return the value of a setting and cache it in self.__dict__."""
    if (_wrapped := self._wrapped) is empty:
        self._setup(name)
        _wrapped = self._wrapped
    val = getattr(_wrapped, name)

    # Special case some settings which require further modification.
    # This is done here for performance reasons so the modified value is cached.
    if name in {"MEDIA_URL", "STATIC_URL"} and val is not None:
        val = self._add_script_prefix(val)
    elif name == "SECRET_KEY" and not val:
        raise ImproperlyConfigured("The SECRET_KEY setting must not be empty.")

    self.__dict__[name] = val
    return val