from typing import Dict, Any, Optional, Union, Sequence, Type, Tuple

import click


def new_cli_ctx_obj():
    return {
        "traceback": False,
        "scheduler": None,
    }


def cli_option_traceback(func):
    """Decorator for adding a pre-defined, reusable CLI option `--traceback`."""

    # noinspection PyUnusedLocal
    def _callback(ctx: click.Context, param: click.Option, value: bool):
        ctx_obj = ctx.ensure_object(dict)
        if ctx_obj is not None:
            ctx_obj["traceback"] = value
        return value

    return click.option(
        '--traceback',
        is_flag=True,
        help="Enable tracing back errors by dumping the Python call stack. "
             "Pass as very first option to also trace back error during command-line validation.",
        callback=_callback)(func)


def cli_option_scheduler(func):
    """Decorator for adding a pre-defined, reusable CLI option `--scheduler`."""

    # noinspection PyUnusedLocal
    def _callback(ctx: click.Context, param: click.Option, value: Optional[str]):
        if not value:
            return

        address_and_kwargs = value.split("?", 2)
        if len(address_and_kwargs) == 2:
            address, kwargs_string = address_and_kwargs
            kwargs = parse_cli_kwargs(kwargs_string, metavar="SCHEDULER")
        else:
            address, = address_and_kwargs
            kwargs = dict()

        try:
            # The Dask Client registers itself as the default Dask scheduler, and so runs dask.array used by xarray
            import distributed
            scheduler_client = distributed.Client(address, **kwargs)
            ctx_obj = ctx.ensure_object(dict)
            if ctx_obj is not None:
                ctx_obj["scheduler"] = scheduler_client
            return scheduler_client
        except ValueError as e:
            raise click.BadParameter(f'Failed to create Dask scheduler client: {e}') from e

    return click.option(
        '--scheduler',
        metavar='SCHEDULER',
        help="Enable distributed computing using the Dask scheduler identified by SCHEDULER. "
             "SCHEDULER can have the form <address>?<keyword>=<value>,... where <address> "
             "is <host> or <host>:<port> and specifies the scheduler's address in your network. "
             "For more information on distributed computing "
             "using Dask, refer to http://distributed.dask.org/. "
             "Pairs of <keyword>=<value> are passed to the Dask client. "
             "Refer to http://distributed.dask.org/en/latest/api.html#distributed.Client",
        callback=_callback)(func)


def parse_cli_kwargs(value: str, metavar: str = None) -> Dict[str, Any]:
    """
    Parse a string value of the form [<kw>=<arg>{,<kw>=<arg>}] into a dictionary.
    <kw> must be a valid Python identifier, <arg> must be a Python literal.

    :param value: A string value.
    :param metavar: Name of a meta-variable used in CLI.
    :return: a dictionary of keyword-arguments
    """
    if value:
        try:
            return eval(f"dict({value})", {}, {})
        except Exception:
            if metavar:
                message = f"Invalid value for {metavar}: {value!r}"
            else:
                message = f"Invalid value: {value!r}"
            raise click.ClickException(message)
    else:
        return dict()


def parse_cli_sequence(seq_value: Union[None, str, Sequence[Any]],
                       metavar: str = 'parameter',
                       item_parser=None,
                       item_validator=None,
                       allow_none: bool = True,
                       allow_empty_items: bool = False,
                       strip_items: bool = True,
                       item_plural_name: str = 'items',
                       num_items: int = None,
                       num_items_min: int = None,
                       num_items_max: int = None,
                       separator: str = ',',
                       error_type: Type[Exception] = click.ClickException) -> Optional[Tuple[Any, ...]]:
    """
    Parse a CLI argument that is supposed to be a sequence.

    :param seq_value: A string, a string sequence, or None.
    :param metavar: CLI metavar name
    :param item_parser: an optional function that takes a string and parses it
    :param item_validator: a optional function that takes a parsed value and parses it
    :param allow_none: whether it is ok for *seq_value* to be None
    :param allow_empty_items: whether it is ok to have empty values in *seq_value*
    :param strip_items: whether to strip values in *seq_value*
    :param item_plural_name: a name for multiple items
    :param num_items: expected number of items
    :param num_items_min: expected minimum number of items
    :param num_items_max: expected maximum number of items
    :param separator: expected separator if *seq_value* is a string, default is ','.
    :param error_type: value error to be raised in case, defaults to ``click.ClickException``
    :return: parsed and validated *seq_value* as a tuple of values
    """
    if seq_value is None:
        if allow_none:
            return None
        raise error_type(f'{metavar} must be given')
    if isinstance(seq_value, str):
        if ',' in seq_value:
            items = seq_value.split(',')
        elif num_items is not None:
            items = num_items * (seq_value,)
        else:
            items = (seq_value,)
    else:
        items = seq_value
    item_count = len(items)
    if num_items is not None and item_count != num_items:
        raise error_type(f'{metavar} must have {num_items} {item_plural_name} separated by {separator!r}')
    if num_items_min is not None and item_count < num_items_min:
        raise error_type(f'{metavar} must have at least {num_items_min} {item_plural_name} separated by {separator!r}')
    if num_items_max is not None and item_count > num_items_max:
        raise error_type(f'{metavar} must have no more than {num_items_max} {item_plural_name} separated by {separator!r}')
    if strip_items:
        items = tuple(item.strip() for item in items)
    if not allow_empty_items:
        for item in items:
            if not item:
                raise ValueError(f'{item_plural_name} in {metavar} must not be empty')
    if item_parser:
        try:
            items = tuple(map(item_parser, items))
        except ValueError as e:
            raise error_type(f'Invalid {item_plural_name} in {metavar} found: {e}')
    if item_validator:
        try:
            for item in items:
                item_validator(item)
        except ValueError as e:
            raise error_type(f'Invalid {item_plural_name} in {metavar} found: {e}')
    return tuple(items)


def assert_positive_int_item(item: int):
    """
    A validator for positive integer number sequences.
    Frequently used to validate counts or image and tile sizes passes as args to the CLI.
    """
    if item <= 0:
        raise ValueError('all items must be positive integer numbers')



def handle_cli_exception(e: BaseException, exit_code: int = None, traceback_mode: bool = False) -> int:
    import sys
    if isinstance(e, click.Abort):
        print(f'Aborted!')
        exit_code = exit_code or 1
    elif isinstance(e, click.ClickException):
        e.show(file=sys.stderr)
        exit_code = exit_code or e.exit_code
    elif isinstance(e, OSError):
        print(f'OS error: {e}', file=sys.stderr)
        exit_code = exit_code or 2
    else:
        print(f'Internal error: {e}', file=sys.stderr)
        exit_code = exit_code or 3
    if traceback_mode:
        import traceback
        traceback.print_exc(file=sys.stderr)
    return exit_code
