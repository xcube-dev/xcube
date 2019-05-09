from typing import Dict, Any, Optional

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
            kwargs = parse_cli_kwargs(kwargs_string, metavar="<scheduler>")
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
        metavar='<scheduler>',
        help="Enable distributed computing using the Dask scheduler identified by <scheduler>. "
             "<scheduler> can have the form <address>?<keyword>=<value>,... where <address> "
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
