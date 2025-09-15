import functools
import os
import pkgutil
import sys
from argparse import _AppendConstAction, _CountAction, _StoreConstAction, _SubParsersAction
from collections import defaultdict
from difflib import get_close_matches
from importlib import import_module
import django
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError, CommandParser, handle_default_options
from django.core.management.color import color_style
from django.utils import autoreload

def call_command(command_name, *args, **options):
    if isinstance(command_name, BaseCommand):
        command = command_name
        command_name = command.__class__.__module__.split('.')[-1]
    else:
        try:
            app_name = get_commands()[command_name]
        except KeyError:
            raise CommandError('Unknown command: %r' % command_name)
        if isinstance(app_name, BaseCommand):
            command = app_name
        else:
            command = load_command_class(app_name, command_name)
    parser = command.create_parser('', command_name)
    opt_mapping = {min(s_opt.option_strings).lstrip('-').replace('-', '_'): s_opt.dest for s_opt in parser._actions if s_opt.option_strings}
    arg_options = {opt_mapping.get(key, key): value for key, value in options.items()}
    parse_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            parse_args += map(str, arg)
        else:
            parse_args.append(str(arg))

    def get_actions(parser):
        for opt in parser._actions:
            if isinstance(opt, _SubParsersAction):
                for sub_opt in opt.choices.values():
                    yield from get_actions(sub_opt)
            else:
                yield opt
    parser_actions = list(get_actions(parser))
    mutually_exclusive_required_options = {opt for group in parser._mutually_exclusive_groups for opt in group._group_actions if group.required}
    for opt in parser_actions:
        if opt.dest in options and (opt.required or opt in mutually_exclusive_required_options):
            opt_dest_count = sum((v == opt.dest for v in opt_mapping.values()))
            if opt_dest_count > 1:
                raise TypeError(f'Cannot pass the dest {opt.dest!r} that matches multiple arguments via **options.')
            parse_args.append(min(opt.option_strings))
            if isinstance(opt, (_AppendConstAction, _CountAction, _StoreConstAction)):
                continue
            value = arg_options[opt.dest]
            if isinstance(value, (list, tuple)):
                parse_args += map(str, value)
            else:
                parse_args.append(str(value))
    defaults = parser.parse_args(args=parse_args)
    defaults = dict(defaults._get_kwargs(), **arg_options)
    stealth_options = set(command.base_stealth_options + command.stealth_options)
    dest_parameters = {action.dest for action in parser_actions}
    valid_options = (dest_parameters | stealth_options).union(opt_mapping)
    unknown_options = set(options) - valid_options
    if unknown_options:
        raise TypeError('Unknown option(s) for %s command: %s. Valid options are: %s.' % (command_name, ', '.join(sorted(unknown_options)), ', '.join(sorted(valid_options))))
    args = defaults.pop('args', ())
    if 'skip_checks' not in options:
        defaults['skip_checks'] = True
    return command.execute(*args, **defaults)