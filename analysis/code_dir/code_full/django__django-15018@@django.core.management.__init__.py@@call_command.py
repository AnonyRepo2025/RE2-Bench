def get_commands():
    commands = {name: 'django.core' for name in find_commands(__path__[0])}
    if not settings.configured:
        return commands
    for app_config in reversed(list(apps.get_app_configs())):
        path = os.path.join(app_config.path, 'management')
        commands.update({name: app_config.name for name in find_commands(path)})
    return commands

def find_commands(management_dir):
    command_dir = os.path.join(management_dir, 'commands')
    return [name for _, name, is_pkg in pkgutil.iter_modules([command_dir]) if not is_pkg and (not name.startswith('_'))]

def configured(self):
    return self._wrapped is not empty

def get_app_configs(self):
    self.check_apps_ready()
    return self.app_configs.values()

def check_apps_ready(self):
    if not self.apps_ready:
        from django.conf import settings
        settings.INSTALLED_APPS
        raise AppRegistryNotReady("Apps aren't loaded yet.")

def load_command_class(app_name, name):
    module = import_module('%s.management.commands.%s' % (app_name, name))
    return module.Command()

def __init__(self, stdout=None, stderr=None, no_color=False, force_color=False):
    self.stdout = OutputWrapper(stdout or sys.stdout)
    self.stderr = OutputWrapper(stderr or sys.stderr)
    if no_color and force_color:
        raise CommandError("'no_color' and 'force_color' can't be used together.")
    if no_color:
        self.style = no_style()
    else:
        self.style = color_style(force_color)
        self.stderr.style_func = self.style.ERROR
    if not isinstance(self.requires_system_checks, (list, tuple)) and self.requires_system_checks != ALL_CHECKS:
        raise TypeError('requires_system_checks must be a list or tuple.')

def __init__(self, out, ending='\n'):
    self._out = out
    self.style_func = None
    self.ending = ending

def style_func(self):
    return self._style_func

def color_style(force_color=False):
    if not force_color and (not supports_color()):
        return no_style()
    return make_style(os.environ.get('DJANGO_COLORS', ''))

def supports_color():

    def vt_codes_enabled_in_windows_registry():
        try:
            import winreg
        except ImportError:
            return False
        else:
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Console')
            try:
                reg_key_value, _ = winreg.QueryValueEx(reg_key, 'VirtualTerminalLevel')
            except FileNotFoundError:
                return False
            else:
                return reg_key_value == 1
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return is_a_tty and (sys.platform != 'win32' or HAS_COLORAMA or 'ANSICON' in os.environ or ('WT_SESSION' in os.environ) or (os.environ.get('TERM_PROGRAM') == 'vscode') or vt_codes_enabled_in_windows_registry())

def isatty(self):
    return hasattr(self._out, 'isatty') and self._out.isatty()

def create_parser(self, prog_name, subcommand, **kwargs):
    parser = CommandParser(prog='%s %s' % (os.path.basename(prog_name), subcommand), description=self.help or None, formatter_class=DjangoHelpFormatter, missing_args_message=getattr(self, 'missing_args_message', None), called_from_command_line=getattr(self, '_called_from_command_line', None), **kwargs)
    self.add_base_argument(parser, '--version', action='version', version=self.get_version(), help="Show program's version number and exit.")
    self.add_base_argument(parser, '-v', '--verbosity', default=1, type=int, choices=[0, 1, 2, 3], help='Verbosity level; 0=minimal output, 1=normal output, 2=verbose output, 3=very verbose output')
    self.add_base_argument(parser, '--settings', help='The Python path to a settings module, e.g. "myproject.settings.main". If this isn\'t provided, the DJANGO_SETTINGS_MODULE environment variable will be used.')
    self.add_base_argument(parser, '--pythonpath', help='A directory to add to the Python path, e.g. "/home/djangoprojects/myproject".')
    self.add_base_argument(parser, '--traceback', action='store_true', help='Raise on CommandError exceptions.')
    self.add_base_argument(parser, '--no-color', action='store_true', help="Don't colorize the command output.")
    self.add_base_argument(parser, '--force-color', action='store_true', help='Force colorization of the command output.')
    if self.requires_system_checks:
        parser.add_argument('--skip-checks', action='store_true', help='Skip system checks.')
    self.add_arguments(parser)
    return parser

def __init__(self, *, missing_args_message=None, called_from_command_line=None, **kwargs):
    self.missing_args_message = missing_args_message
    self.called_from_command_line = called_from_command_line
    super().__init__(**kwargs)

def get_version(self):
    return django.get_version()



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