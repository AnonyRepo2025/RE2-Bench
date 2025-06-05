@print_function(ReprPrinter)
def srepr(expr, **settings):
    """return expr in repr form"""
    return ReprPrinter(settings).doprint(expr)

class Printer(object):
    _global_settings = {}  # type: Dict[str, Any]
    _default_settings = {}  # type: Dict[str, Any]
    printmethod = None  # type: str

    @classmethod
    def _get_initial_settings(cls):
        settings = cls._default_settings.copy()
        for key, val in cls._global_settings.items():
            if key in cls._default_settings:
                settings[key] = val
        return settings

    def __init__(self, settings=None):
        self._str = str

        self._settings = self._get_initial_settings()
        self._context = dict()  # mutable during printing

        if settings is not None:
            self._settings.update(settings)

            if len(self._settings) > len(self._default_settings):
                for key in self._settings:
                    if key not in self._default_settings:
                        raise TypeError("Unknown setting '%s'." % key)
        self._print_level = 0
        
    def doprint(self, expr):
        """Returns printer's representation for expr (as a string)"""
        return self._str(self._print(expr))
    
    def _print(self, expr, **kwargs):
        self._print_level += 1
        try:
            if (self.printmethod and hasattr(expr, self.printmethod)
                    and not isinstance(expr, BasicMeta)):
                return getattr(expr, self.printmethod)(self, **kwargs)
            classes = type(expr).__mro__
            if AppliedUndef in classes:
                classes = classes[classes.index(AppliedUndef):]
            if UndefinedFunction in classes:
                classes = classes[classes.index(UndefinedFunction):]
            if Function in classes:
                i = classes.index(Function)
                classes = tuple(c for c in classes[:i] if \
                    c.__name__ == classes[0].__name__ or \
                    c.__name__.endswith("Base")) + classes[i:]
            for cls in classes:
                printmethod = '_print_' + cls.__name__
                if hasattr(self, printmethod):
                    return getattr(self, printmethod)(expr, **kwargs)
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1