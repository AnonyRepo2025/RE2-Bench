from collections.abc import Iterable

class NDimArray(Printable):

    _diff_wrt = True
    is_scalar = False

    @classmethod
    def _scan_iterable_shape(cls, iterable):
        def f(pointer):
            if not isinstance(pointer, Iterable):
                return [pointer], ()

            if len(pointer) == 0:
                return [], (0,)

            result = []
            elems, shapes = zip(*[f(i) for i in pointer])
            if len(set(shapes)) != 1:
                raise ValueError("could not determine shape unambiguously")
            for i in elems:
                result.extend(i)
            return result, (len(shapes),)+shapes[0]

        return f(iterable)
