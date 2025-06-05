class Variable(
    common.AbstractArray, arithmetic.SupportsArithmetic, utils.NdimSizeLenMixin
):
    __slots__ = ("_dims", "_data", "_attrs", "_encoding")

    def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
        self._data = as_compatible_data(data, fastpath=fastpath)
        self._dims = self._parse_dimensions(dims)
        self._attrs = None
        self._encoding = None
        if attrs is not None:
            self.attrs = attrs
        if encoding is not None:
            self.encoding = encoding
    @property
    def _in_memory(self):
        return isinstance(self._data, (np.ndarray, np.number, PandasIndexAdapter)) or (
            isinstance(self._data, indexing.MemoryCachedArray)
            and isinstance(self._data.array, indexing.NumpyIndexingAdapter)
        )
