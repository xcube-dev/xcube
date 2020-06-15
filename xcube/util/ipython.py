from typing import Type


def register_json_formatter(cls: Type, to_dict_method_name: str = 'to_dict'):
    """
    TODO
    :param cls:
    :param to_dict_method_name:
    :return:
    """
    if not hasattr(cls, to_dict_method_name) or not callable(getattr(cls, to_dict_method_name)):
        raise ValueError(f'{cls} must define a {to_dict_method_name}() method')

    try:
        import IPython
        import IPython.display

        if IPython.get_ipython() is not None:
            def obj_to_dict(obj):
                return getattr(obj, to_dict_method_name)()

            ipy_formatter = IPython.get_ipython().display_formatter.formatters['application/json']
            ipy_formatter.for_type(cls, obj_to_dict)

    except ImportError:
        pass
