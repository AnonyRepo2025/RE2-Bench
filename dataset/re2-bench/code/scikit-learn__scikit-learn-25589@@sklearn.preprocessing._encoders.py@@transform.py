import numbers
from numbers import Integral, Real
import warnings
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from ..utils import check_array, is_scalar_nan, _safe_indexing
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_feature_names_in
from ..utils._param_validation import Interval, StrOptions, Hidden
from ..utils._mask import _get_mask
from ..utils._encode import _encode, _check_unknown, _unique, _get_counts

__all__ = ["OneHotEncoder", "OrdinalEncoder"]

class OneHotEncoder(_BaseEncoder):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse_output``
    parameter)

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

        .. versionadded:: 0.20

    drop : {'first', 'if_binary'} or an array-like of shape (n_features,), \
            default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into an unregularized linear regression model.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - 'if_binary' : drop the first category in each feature with two
          categories. Features with 1 or more than 2 categories are
          left intact.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

        When `max_categories` or `min_frequency` is configured to group
        infrequent categories, the dropping behavior is handled after the
        grouping.

        .. versionadded:: 0.21
           The parameter `drop` was added in 0.21.

        .. versionchanged:: 0.23
           The option `drop='if_binary'` was added in 0.23.

        .. versionchanged:: 1.1
            Support for dropping infrequent categories.

    sparse : bool, default=True
        Will return sparse matrix if set True else will return an array.

        .. deprecated:: 1.2
           `sparse` is deprecated in 1.2 and will be removed in 1.4. Use
           `sparse_output` instead.

    sparse_output : bool, default=True
        Will return sparse matrix if set True else will return an array.

        .. versionadded:: 1.2
           `sparse` was renamed to `sparse_output`

    dtype : number type, default=float
        Desired dtype of output.

    handle_unknown : {'error', 'ignore', 'infrequent_if_exist'}, \
                     default='error'
        Specifies the way unknown categories are handled during :meth:`transform`.

        - 'error' : Raise an error if an unknown category is present during transform.
        - 'ignore' : When an unknown category is encountered during
          transform, the resulting one-hot encoded columns for this feature
          will be all zeros. In the inverse transform, an unknown category
          will be denoted as None.
        - 'infrequent_if_exist' : When an unknown category is encountered
          during transform, the resulting one-hot encoded columns for this
          feature will map to the infrequent category if it exists. The
          infrequent category will be mapped to the last position in the
          encoding. During inverse transform, an unknown category will be
          mapped to the category denoted `'infrequent'` if it exists. If the
          `'infrequent'` category does not exist, then :meth:`transform` and
          :meth:`inverse_transform` will handle an unknown category as with
          `handle_unknown='ignore'`. Infrequent categories exist based on
          `min_frequency` and `max_categories`. Read more in the
          :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

        .. versionchanged:: 1.1
            `'infrequent_if_exist'` was added to automatically handle unknown
            categories and infrequent categories.

    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.

        - If `int`, categories with a smaller cardinality will be considered
          infrequent.

        - If `float`, categories with a smaller cardinality than
          `min_frequency * n_samples`  will be considered infrequent.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

    max_categories : int, default=None
        Specifies an upper limit to the number of output features for each input
        feature when considering infrequent categories. If there are infrequent
        categories, `max_categories` includes the category representing the
        infrequent categories along with the frequent categories. If `None`,
        there is no limit to the number of output features.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

    feature_name_combiner : "concat" or callable, default="concat"
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        `"concat"` concatenates encoded feature name and category with
        `feature + "_" + str(category)`.E.g. feature X with values 1, 6, 7 create
        feature names `X_1, X_6, X_7`.

        .. versionadded:: 1.3

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for each feature.
        - ``drop_idx_[i] = None`` if no category is to be dropped from the
          feature with index ``i``, e.g. when `drop='if_binary'` and the
          feature isn't binary.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

        If infrequent categories are enabled by setting `min_frequency` or
        `max_categories` to a non-default value and `drop_idx[i]` corresponds
        to a infrequent category, then the entire infrequent category is
        dropped.

        .. versionchanged:: 0.23
           Added the possibility to contain `None` values.

    infrequent_categories_ : list of ndarray
        Defined only if infrequent categories are enabled by setting
        `min_frequency` or `max_categories` to a non-default value.
        `infrequent_categories_[i]` are the infrequent categories for feature
        `i`. If the feature `i` has no infrequent categories
        `infrequent_categories_[i]` is None.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    feature_name_combiner : callable or None
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        .. versionadded:: 1.3

    See Also
    --------
    OrdinalEncoder : Performs an ordinal (integer)
      encoding of the categorical features.
    sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
      encoding of dictionary items or strings.
    LabelBinarizer : Binarizes labels in a one-vs-all
      fashion.
    MultiLabelBinarizer : Transforms between iterable of
      iterables and a multilabel format, e.g. a (samples x classes) binary
      matrix indicating the presence of a class label.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder

    One can discard categories not seen during `fit`:

    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OneHotEncoder(handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)
    >>> enc.get_feature_names_out(['gender', 'group'])
    array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)

    One can always drop the first column for each feature:

    >>> drop_enc = OneHotEncoder(drop='first').fit(X)
    >>> drop_enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 0., 0.],
           [1., 1., 0.]])

    Or drop a column for feature only having 2 categories:

    >>> drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
    >>> drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 1., 0., 0.],
           [1., 0., 1., 0.]])

    One can change the way feature names are created.

    >>> def custom_combiner(feature, category):
    ...     return str(feature) + "_" + type(category).__name__ + "_" + str(category)
    >>> custom_fnames_enc = OneHotEncoder(feature_name_combiner=custom_combiner).fit(X)
    >>> custom_fnames_enc.get_feature_names_out()
    array(['x0_str_Female', 'x0_str_Male', 'x1_int_1', 'x1_int_2', 'x1_int_3'],
          dtype=object)

    Infrequent categories are enabled by setting `max_categories` or `min_frequency`.

    >>> import numpy as np
    >>> X = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
    >>> ohe = OneHotEncoder(max_categories=3, sparse_output=False).fit(X)
    >>> ohe.infrequent_categories_
    [array(['a', 'd'], dtype=object)]
    >>> ohe.transform([["a"], ["b"]])
    array([[0., 0., 1.],
           [1., 0., 0.]])
    """

    _parameter_constraints: dict = {
        "categories": [StrOptions({"auto"}), list],
        "drop": [StrOptions({"first", "if_binary"}), "array-like", None],
        "dtype": "no_validation",  # validation delegated to numpy
        "handle_unknown": [StrOptions({"error", "ignore", "infrequent_if_exist"})],
        "max_categories": [Interval(Integral, 1, None, closed="left"), None],
        "min_frequency": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="neither"),
            None,
        ],
        "sparse": [Hidden(StrOptions({"deprecated"})), "boolean"],  # deprecated
        "sparse_output": ["boolean"],
        "feature_name_combiner": [StrOptions({"concat"}), callable],
    }

    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse="deprecated",
        sparse_output=True,
        dtype=np.float64,
        handle_unknown="error",
        min_frequency=None,
        max_categories=None,
        feature_name_combiner="concat",
    ):
        self.categories = categories
        # TODO(1.4): Remove self.sparse
        self.sparse = sparse
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    @property
    def infrequent_categories_(self):
        """Infrequent categories for each feature."""
        # raises an AttributeError if `_infrequent_indices` is not defined
        infrequent_indices = self._infrequent_indices
        return [
            None if indices is None else category[indices]
            for category, indices in zip(self.categories_, infrequent_indices)
        ]

    def _check_infrequent_enabled(self):
        """
        This functions checks whether _infrequent_enabled is True or False.
        This has to be called after parameter validation in the fit function.
        """
        self._infrequent_enabled = (
            self.max_categories is not None and self.max_categories >= 1
        ) or self.min_frequency is not None

    def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
        """Convert `drop_idx` into the index for infrequent categories.

        If there are no infrequent categories, then `drop_idx` is
        returned. This method is called in `_set_drop_idx` when the `drop`
        parameter is an array-like.
        """
        if not self._infrequent_enabled:
            return drop_idx

        default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
        if default_to_infrequent is None:
            return drop_idx

        # Raise error when explicitly dropping a category that is infrequent
        infrequent_indices = self._infrequent_indices[feature_idx]
        if infrequent_indices is not None and drop_idx in infrequent_indices:
            categories = self.categories_[feature_idx]
            raise ValueError(
                f"Unable to drop category {categories[drop_idx]!r} from feature"
                f" {feature_idx} because it is infrequent"
            )
        return default_to_infrequent[drop_idx]

    def _set_drop_idx(self):
        """Compute the drop indices associated with `self.categories_`.

        If `self.drop` is:
        - `None`, No categories have been dropped.
        - `'first'`, All zeros to drop the first category.
        - `'if_binary'`, All zeros if the category is binary and `None`
          otherwise.
        - array-like, The indices of the categories that match the
          categories in `self.drop`. If the dropped category is an infrequent
          category, then the index for the infrequent category is used. This
          means that the entire infrequent category is dropped.

        This methods defines a public `drop_idx_` and a private
        `_drop_idx_after_grouping`.

        - `drop_idx_`: Public facing API that references the drop category in
          `self.categories_`.
        - `_drop_idx_after_grouping`: Used internally to drop categories *after* the
          infrequent categories are grouped together.

        If there are no infrequent categories or drop is `None`, then
        `drop_idx_=_drop_idx_after_grouping`.
        """
        if self.drop is None:
            drop_idx_after_grouping = None
        elif isinstance(self.drop, str):
            if self.drop == "first":
                drop_idx_after_grouping = np.zeros(len(self.categories_), dtype=object)
            elif self.drop == "if_binary":
                n_features_out_no_drop = [len(cat) for cat in self.categories_]
                if self._infrequent_enabled:
                    for i, infreq_idx in enumerate(self._infrequent_indices):
                        if infreq_idx is None:
                            continue
                        n_features_out_no_drop[i] -= infreq_idx.size - 1

                drop_idx_after_grouping = np.array(
                    [
                        0 if n_features_out == 2 else None
                        for n_features_out in n_features_out_no_drop
                    ],
                    dtype=object,
                )

        else:
            drop_array = np.asarray(self.drop, dtype=object)
            droplen = len(drop_array)

            if droplen != len(self.categories_):
                msg = (
                    "`drop` should have length equal to the number "
                    "of features ({}), got {}"
                )
                raise ValueError(msg.format(len(self.categories_), droplen))
            missing_drops = []
            drop_indices = []
            for feature_idx, (drop_val, cat_list) in enumerate(
                zip(drop_array, self.categories_)
            ):
                if not is_scalar_nan(drop_val):
                    drop_idx = np.where(cat_list == drop_val)[0]
                    if drop_idx.size:  # found drop idx
                        drop_indices.append(
                            self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0])
                        )
                    else:
                        missing_drops.append((feature_idx, drop_val))
                    continue

                # drop_val is nan, find nan in categories manually
                for cat_idx, cat in enumerate(cat_list):
                    if is_scalar_nan(cat):
                        drop_indices.append(
                            self._map_drop_idx_to_infrequent(feature_idx, cat_idx)
                        )
                        break
                else:  # loop did not break thus drop is missing
                    missing_drops.append((feature_idx, drop_val))

            if any(missing_drops):
                msg = (
                    "The following categories were supposed to be "
                    "dropped, but were not found in the training "
                    "data.\n{}".format(
                        "\n".join(
                            [
                                "Category: {}, Feature: {}".format(c, v)
                                for c, v in missing_drops
                            ]
                        )
                    )
                )
                raise ValueError(msg)
            drop_idx_after_grouping = np.array(drop_indices, dtype=object)

        # `_drop_idx_after_grouping` are the categories to drop *after* the infrequent
        # categories are grouped together. If needed, we remap `drop_idx` back
        # to the categories seen in `self.categories_`.
        self._drop_idx_after_grouping = drop_idx_after_grouping

        if not self._infrequent_enabled or drop_idx_after_grouping is None:
            self.drop_idx_ = self._drop_idx_after_grouping
        else:
            drop_idx_ = []
            for feature_idx, drop_idx in enumerate(drop_idx_after_grouping):
                default_to_infrequent = self._default_to_infrequent_mappings[
                    feature_idx
                ]
                if drop_idx is None or default_to_infrequent is None:
                    orig_drop_idx = drop_idx
                else:
                    orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]

                drop_idx_.append(orig_drop_idx)

            self.drop_idx_ = np.asarray(drop_idx_, dtype=object)

    def _identify_infrequent(self, category_count, n_samples, col_idx):
        """Compute the infrequent indices.

        Parameters
        ----------
        category_count : ndarray of shape (n_cardinality,)
            Category counts.

        n_samples : int
            Number of samples.

        col_idx : int
            Index of the current category. Only used for the error message.

        Returns
        -------
        output : ndarray of shape (n_infrequent_categories,) or None
            If there are infrequent categories, indices of infrequent
            categories. Otherwise None.
        """
        if isinstance(self.min_frequency, numbers.Integral):
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)

        n_current_features = category_count.size - infrequent_mask.sum() + 1
        if self.max_categories is not None and self.max_categories < n_current_features:
            # stable sort to preserve original count order
            smallest_levels = np.argsort(category_count, kind="mergesort")[
                : -self.max_categories + 1
            ]
            infrequent_mask[smallest_levels] = True

        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None

    def _fit_infrequent_category_mapping(self, n_samples, category_counts):
        """Fit infrequent categories.

        Defines the private attribute: `_default_to_infrequent_mappings`. For
        feature `i`, `_default_to_infrequent_mappings[i]` defines the mapping
        from the integer encoding returned by `super().transform()` into
        infrequent categories. If `_default_to_infrequent_mappings[i]` is None,
        there were no infrequent categories in the training set.

        For example if categories 0, 2 and 4 were frequent, while categories
        1, 3, 5 were infrequent for feature 7, then these categories are mapped
        to a single output:
        `_default_to_infrequent_mappings[7] = array([0, 3, 1, 3, 2, 3])`

        Defines private attribute: `_infrequent_indices`. `_infrequent_indices[i]`
        is an array of indices such that
        `categories_[i][_infrequent_indices[i]]` are all the infrequent category
        labels. If the feature `i` has no infrequent categories
        `_infrequent_indices[i]` is None.

        .. versionadded:: 1.1

        Parameters
        ----------
        n_samples : int
            Number of samples in training set.
        category_counts: list of ndarray
            `category_counts[i]` is the category counts corresponding to
            `self.categories_[i]`.
        """
        self._infrequent_indices = [
            self._identify_infrequent(category_count, n_samples, col_idx)
            for col_idx, category_count in enumerate(category_counts)
        ]

        # compute mapping from default mapping to infrequent mapping
        self._default_to_infrequent_mappings = []

        for cats, infreq_idx in zip(self.categories_, self._infrequent_indices):
            # no infrequent categories
            if infreq_idx is None:
                self._default_to_infrequent_mappings.append(None)
                continue

            n_cats = len(cats)
            # infrequent indices exist
            mapping = np.empty(n_cats, dtype=np.int64)
            n_infrequent_cats = infreq_idx.size

            # infrequent categories are mapped to the last element.
            n_frequent_cats = n_cats - n_infrequent_cats
            mapping[infreq_idx] = n_frequent_cats

            frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
            mapping[frequent_indices] = np.arange(n_frequent_cats)

            self._default_to_infrequent_mappings.append(mapping)

    def _map_infrequent_categories(self, X_int, X_mask):
        """Map infrequent categories to integer representing the infrequent category.

        This modifies X_int in-place. Values that were invalid based on `X_mask`
        are mapped to the infrequent category if there was an infrequent
        category for that feature.

        Parameters
        ----------
        X_int: ndarray of shape (n_samples, n_features)
            Integer encoded categories.

        X_mask: ndarray of shape (n_samples, n_features)
            Bool mask for valid values in `X_int`.
        """
        if not self._infrequent_enabled:
            return

        for col_idx in range(X_int.shape[1]):
            infrequent_idx = self._infrequent_indices[col_idx]
            if infrequent_idx is None:
                continue

            X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]
            if self.handle_unknown == "infrequent_if_exist":
                # All the unknown values are now mapped to the
                # infrequent_idx[0], which makes the unknown values valid
                # This is needed in `transform` when the encoding is formed
                # using `X_mask`.
                X_mask[:, col_idx] = True

        # Remaps encoding in `X_int` where the infrequent categories are
        # grouped together.
        for i, mapping in enumerate(self._default_to_infrequent_mappings):
            if mapping is None:
                continue
            X_int[:, i] = np.take(mapping, X_int[:, i])

    def _compute_transformed_categories(self, i, remove_dropped=True):
        """Compute the transformed categories used for column `i`.

        1. If there are infrequent categories, the category is named
        'infrequent_sklearn'.
        2. Dropped columns are removed when remove_dropped=True.
        """
        cats = self.categories_[i]

        if self._infrequent_enabled:
            infreq_map = self._default_to_infrequent_mappings[i]
            if infreq_map is not None:
                frequent_mask = infreq_map < infreq_map.max()
                infrequent_cat = "infrequent_sklearn"
                # infrequent category is always at the end
                cats = np.concatenate(
                    (cats[frequent_mask], np.array([infrequent_cat], dtype=object))
                )

        if remove_dropped:
            cats = self._remove_dropped_categories(cats, i)
        return cats

    def _remove_dropped_categories(self, categories, i):
        """Remove dropped categories."""
        if (
            self._drop_idx_after_grouping is not None
            and self._drop_idx_after_grouping[i] is not None
        ):
            return np.delete(categories, self._drop_idx_after_grouping[i])
        return categories

    def _compute_n_features_outs(self):
        """Compute the n_features_out for each input feature."""
        output = [len(cats) for cats in self.categories_]

        if self._drop_idx_after_grouping is not None:
            for i, drop_idx in enumerate(self._drop_idx_after_grouping):
                if drop_idx is not None:
                    output[i] -= 1

        if not self._infrequent_enabled:
            return output

        # infrequent is enabled, the number of features out are reduced
        # because the infrequent categories are grouped together
        for i, infreq_idx in enumerate(self._infrequent_indices):
            if infreq_idx is None:
                continue
            output[i] -= infreq_idx.size - 1

        return output

    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
            Fitted encoder.
        """
        self._validate_params()

        if self.sparse != "deprecated":
            warnings.warn(
                "`sparse` was renamed to `sparse_output` in version 1.2 and "
                "will be removed in 1.4. `sparse_output` is ignored unless you "
                "leave `sparse` to its default value.",
                FutureWarning,
            )
            self.sparse_output = self.sparse

        self._check_infrequent_enabled()

        fit_results = self._fit(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",
            return_counts=self._infrequent_enabled,
        )
        if self._infrequent_enabled:
            self._fit_infrequent_category_mapping(
                fit_results["n_samples"], fit_results["category_counts"]
            )
        self._set_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        return self

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        If there are infrequent categories for a feature, the infrequent
        categories will be grouped into a single category.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse_output=True`, a sparse matrix will be
            returned.
        """
        check_is_fitted(self)
        # validation of X happens in _check_X called by _transform
        warn_on_unknown = self.drop is not None and self.handle_unknown in {
            "ignore",
            "infrequent_if_exist",
        }
        X_int, X_mask = self._transform(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",
            warn_on_unknown=warn_on_unknown,
        )
        self._map_infrequent_categories(X_int, X_mask)

        n_samples, n_features = X_int.shape

        if self._drop_idx_after_grouping is not None:
            to_drop = self._drop_idx_after_grouping.copy()
            # We remove all the dropped categories from mask, and decrement all
            # categories that occur after them to avoid an empty column.
            keep_cells = X_int != to_drop
            for i, cats in enumerate(self.categories_):
                # drop='if_binary' but feature isn't binary
                if to_drop[i] is None:
                    # set to cardinality to not drop from X_int
                    to_drop[i] = len(cats)

            to_drop = to_drop.reshape(1, -1)
            X_int[X_int > to_drop] -= 1
            X_mask &= keep_cells

        mask = X_mask.ravel()
        feature_indices = np.cumsum([0] + self._n_features_outs)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]

        indptr = np.empty(n_samples + 1, dtype=int)
        indptr[0] = 0
        np.sum(X_mask, axis=1, out=indptr[1:], dtype=indptr.dtype)
        np.cumsum(indptr[1:], out=indptr[1:])
        data = np.ones(indptr[-1])

        out = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_samples, feature_indices[-1]),
            dtype=self.dtype,
        )
        if not self.sparse_output:
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        When unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category. If the
        feature with the unknown category has a dropped category, the dropped
        category will be its inverse.

        For a given input feature, if there is an infrequent category,
        'infrequent_sklearn' will be used to represent the infrequent category.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        n_features_out = np.sum(self._n_features_outs)

        # validate shape of passed X
        msg = (
            "Shape of the passed X data is not correct. Expected {0} columns, got {1}."
        )
        if X.shape[1] != n_features_out:
            raise ValueError(msg.format(n_features_out, X.shape[1]))

        transformed_features = [
            self._compute_transformed_categories(i, remove_dropped=False)
            for i, _ in enumerate(self.categories_)
        ]

        # create resulting array of appropriate dtype
        dt = np.result_type(*[cat.dtype for cat in transformed_features])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        j = 0
        found_unknown = {}

        if self._infrequent_enabled:
            infrequent_indices = self._infrequent_indices
        else:
            infrequent_indices = [None] * n_features

        for i in range(n_features):
            cats_wo_dropped = self._remove_dropped_categories(
                transformed_features[i], i
            )
            n_categories = cats_wo_dropped.shape[0]

            # Only happens if there was a column with a unique
            # category. In this case we just fill the column with this
            # unique category value.
            if n_categories == 0:
                X_tr[:, i] = self.categories_[i][self._drop_idx_after_grouping[i]]
                j += n_categories
                continue
            sub = X[:, j : j + n_categories]
            # for sparse X argmax returns 2D matrix, ensure 1D array
            labels = np.asarray(sub.argmax(axis=1)).flatten()
            X_tr[:, i] = cats_wo_dropped[labels]

            if self.handle_unknown == "ignore" or (
                self.handle_unknown == "infrequent_if_exist"
                and infrequent_indices[i] is None
            ):
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                # ignored unknown categories: we have a row of all zero
                if unknown.any():
                    # if categories were dropped then unknown categories will
                    # be mapped to the dropped category
                    if (
                        self._drop_idx_after_grouping is None
                        or self._drop_idx_after_grouping[i] is None
                    ):
                        found_unknown[i] = unknown
                    else:
                        X_tr[unknown, i] = self.categories_[i][
                            self._drop_idx_after_grouping[i]
                        ]
            else:
                dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                if dropped.any():
                    if self._drop_idx_after_grouping is None:
                        all_zero_samples = np.flatnonzero(dropped)
                        raise ValueError(
                            f"Samples {all_zero_samples} can not be inverted "
                            "when drop=None and handle_unknown='error' "
                            "because they contain all zeros"
                        )
                    # we can safely assume that all of the nulls in each column
                    # are the dropped value
                    drop_idx = self._drop_idx_after_grouping[i]
                    X_tr[dropped, i] = transformed_features[i][drop_idx]

            j += n_categories

        # if ignored are found: potentially need to upcast result to
        # insert None values
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)

            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None

        return X_tr

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        cats = [
            self._compute_transformed_categories(i)
            for i, _ in enumerate(self.categories_)
        ]

        name_combiner = self._check_get_feature_name_combiner()
        feature_names = []
        for i in range(len(cats)):
            names = [name_combiner(input_features[i], t) for t in cats[i]]
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)

    def _check_get_feature_name_combiner(self):
        if self.feature_name_combiner == "concat":
            return lambda feature, category: feature + "_" + str(category)
        else:  # callable
            dry_run_combiner = self.feature_name_combiner("feature", "category")
            if not isinstance(dry_run_combiner, str):
                raise TypeError(
                    "When `feature_name_combiner` is a callable, it should return a "
                    f"Python string. Got {type(dry_run_combiner)} instead."
                )
            return self.feature_name_combiner
