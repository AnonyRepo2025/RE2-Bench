def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})
