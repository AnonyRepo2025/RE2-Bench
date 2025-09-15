def construct_instance(file_path: str, class_name: str, self_dict: dict):
    import os
    import importlib
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    try:
        instance = cls.__new__(cls)
        try:
            cls.__init__(instance, *[], **{})
        except TypeError:
            pass
    except Exception as e:
        raise RuntimeError(f"Could not construct {class_name}: {e}")
    for k, v in self_dict.items():
        setattr(instance, k, v)
    return instance
