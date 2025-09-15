def __init__(self, operation, indentation=2):
    self.operation = operation
    self.buff = []
    self.indentation = indentation

def serialize(self):

    def _write(_arg_name, _arg_value):
        if _arg_name in self.operation.serialization_expand_args and isinstance(_arg_value, (list, tuple, dict)):
            if isinstance(_arg_value, dict):
                self.feed('%s={' % _arg_name)
                self.indent()
                for key, value in _arg_value.items():
                    key_string, key_imports = MigrationWriter.serialize(key)
                    arg_string, arg_imports = MigrationWriter.serialize(value)
                    args = arg_string.splitlines()
                    if len(args) > 1:
                        self.feed('%s: %s' % (key_string, args[0]))
                        for arg in args[1:-1]:
                            self.feed(arg)
                        self.feed('%s,' % args[-1])
                    else:
                        self.feed('%s: %s,' % (key_string, arg_string))
                    imports.update(key_imports)
                    imports.update(arg_imports)
                self.unindent()
                self.feed('},')
            else:
                self.feed('%s=[' % _arg_name)
                self.indent()
                for item in _arg_value:
                    arg_string, arg_imports = MigrationWriter.serialize(item)
                    args = arg_string.splitlines()
                    if len(args) > 1:
                        for arg in args[:-1]:
                            self.feed(arg)
                        self.feed('%s,' % args[-1])
                    else:
                        self.feed('%s,' % arg_string)
                    imports.update(arg_imports)
                self.unindent()
                self.feed('],')
        else:
            arg_string, arg_imports = MigrationWriter.serialize(_arg_value)
            args = arg_string.splitlines()
            if len(args) > 1:
                self.feed('%s=%s' % (_arg_name, args[0]))
                for arg in args[1:-1]:
                    self.feed(arg)
                self.feed('%s,' % args[-1])
            else:
                self.feed('%s=%s,' % (_arg_name, arg_string))
            imports.update(arg_imports)
    imports = set()
    name, args, kwargs = self.operation.deconstruct()
    operation_args = get_func_args(self.operation.__init__)
    if getattr(migrations, name, None) == self.operation.__class__:
        self.feed('migrations.%s(' % name)
    else:
        imports.add('import %s' % self.operation.__class__.__module__)
        self.feed('%s.%s(' % (self.operation.__class__.__module__, name))
    self.indent()
    for i, arg in enumerate(args):
        arg_value = arg
        arg_name = operation_args[i]
        _write(arg_name, arg_value)
    i = len(args)
    for arg_name in operation_args[i:]:
        if arg_name in kwargs:
            arg_value = kwargs[arg_name]
            _write(arg_name, arg_value)
    self.unindent()
    self.feed('),')
    return (self.render(), imports)

def get_func_args(func):
    params = _get_callable_parameters(func)
    return [param.name for param in params if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]

def _get_callable_parameters(meth_or_func):
    is_method = inspect.ismethod(meth_or_func)
    func = meth_or_func.__func__ if is_method else meth_or_func
    return _get_func_parameters(func, remove_first=is_method)

def feed(self, line):
    self.buff.append(' ' * (self.indentation * 4) + line)

def indent(self):
    self.indentation += 1

def unindent(self):
    self.indentation -= 1

def render(self):
    return '\n'.join(self.buff)

def deconstruct(self):
    kwargs = {'name': self.name, 'fields': self.fields}
    if self.options:
        kwargs['options'] = self.options
    if self.bases and self.bases != (models.Model,):
        kwargs['bases'] = self.bases
    if self.managers and self.managers != [('objects', models.Manager())]:
        kwargs['managers'] = self.managers
    return (self.__class__.__qualname__, [], kwargs)

def _get_func_parameters(func, remove_first):
    parameters = tuple(inspect.signature(func).parameters.values())
    if remove_first:
        parameters = parameters[1:]
    return parameters

def _write(_arg_name, _arg_value):
    if _arg_name in self.operation.serialization_expand_args and isinstance(_arg_value, (list, tuple, dict)):
        if isinstance(_arg_value, dict):
            self.feed('%s={' % _arg_name)
            self.indent()
            for key, value in _arg_value.items():
                key_string, key_imports = MigrationWriter.serialize(key)
                arg_string, arg_imports = MigrationWriter.serialize(value)
                args = arg_string.splitlines()
                if len(args) > 1:
                    self.feed('%s: %s' % (key_string, args[0]))
                    for arg in args[1:-1]:
                        self.feed(arg)
                    self.feed('%s,' % args[-1])
                else:
                    self.feed('%s: %s,' % (key_string, arg_string))
                imports.update(key_imports)
                imports.update(arg_imports)
            self.unindent()
            self.feed('},')
        else:
            self.feed('%s=[' % _arg_name)
            self.indent()
            for item in _arg_value:
                arg_string, arg_imports = MigrationWriter.serialize(item)
                args = arg_string.splitlines()
                if len(args) > 1:
                    for arg in args[:-1]:
                        self.feed(arg)
                    self.feed('%s,' % args[-1])
                else:
                    self.feed('%s,' % arg_string)
                imports.update(arg_imports)
            self.unindent()
            self.feed('],')
    else:
        arg_string, arg_imports = MigrationWriter.serialize(_arg_value)
        args = arg_string.splitlines()
        if len(args) > 1:
            self.feed('%s=%s' % (_arg_name, args[0]))
            for arg in args[1:-1]:
                self.feed(arg)
            self.feed('%s,' % args[-1])
        else:
            self.feed('%s=%s,' % (_arg_name, arg_string))
        imports.update(arg_imports)

def serializer_factory(value):
    if isinstance(value, Promise):
        value = str(value)
    elif isinstance(value, LazyObject):
        value = value.__reduce__()[1][0]
    if isinstance(value, models.Field):
        return ModelFieldSerializer(value)
    if isinstance(value, models.manager.BaseManager):
        return ModelManagerSerializer(value)
    if isinstance(value, Operation):
        return OperationSerializer(value)
    if isinstance(value, type):
        return TypeSerializer(value)
    if hasattr(value, 'deconstruct'):
        return DeconstructableSerializer(value)
    for type_, serializer_cls in Serializer._registry.items():
        if isinstance(value, type_):
            return serializer_cls(value)
    raise ValueError('Cannot serialize: %r\nThere are some values Django cannot serialize into migration files.\nFor more, see https://docs.djangoproject.com/en/%s/topics/migrations/#migration-serializing' % (value, get_docs_version()))

def __init__(self, value):
    self.value = value

def serialize(self):
    return (repr(self.value), set())

def get_version(version=None):
    version = get_complete_version(version)
    main = get_main_version(version)
    sub = ''
    if version[3] == 'alpha' and version[4] == 0:
        git_changeset = get_git_changeset()
        if git_changeset:
            sub = '.dev%s' % git_changeset
    elif version[3] != 'final':
        mapping = {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}
        sub = mapping[version[3]] + str(version[4])
    return main + sub



import os
import re
from importlib import import_module
from django import get_version
from django.apps import apps
from django.conf import SettingsReference
from django.db import migrations
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.serializer import Serializer, serializer_factory
from django.utils.inspect import get_func_args
from django.utils.module_loading import module_dir
from django.utils.timezone import now
MIGRATION_HEADER_TEMPLATE = '# Generated by Django %(version)s on %(timestamp)s\n\n'
MIGRATION_TEMPLATE = '%(migration_header)s%(imports)s\n\nclass Migration(migrations.Migration):\n%(replaces_str)s%(initial_str)s\n    dependencies = [\n%(dependencies)s    ]\n\n    operations = [\n%(operations)s    ]\n'

class MigrationWriter:

    def as_string(self):
        items = {'replaces_str': '', 'initial_str': ''}
        imports = set()
        operations = []
        for operation in self.migration.operations:
            operation_string, operation_imports = OperationWriter(operation).serialize()
            imports.update(operation_imports)
            operations.append(operation_string)
        items['operations'] = '\n'.join(operations) + '\n' if operations else ''
        dependencies = []
        for dependency in self.migration.dependencies:
            if dependency[0] == '__setting__':
                dependencies.append('        migrations.swappable_dependency(settings.%s),' % dependency[1])
                imports.add('from django.conf import settings')
            else:
                dependencies.append('        %s,' % self.serialize(dependency)[0])
        items['dependencies'] = '\n'.join(dependencies) + '\n' if dependencies else ''
        migration_imports = set()
        for line in list(imports):
            if re.match('^import (.*)\\.\\d+[^\\s]*$', line):
                migration_imports.add(line.split('import')[1].strip())
                imports.remove(line)
                self.needs_manual_porting = True
        if 'from django.db import models' in imports:
            imports.discard('from django.db import models')
            imports.add('from django.db import migrations, models')
        else:
            imports.add('from django.db import migrations')
        sorted_imports = sorted(imports, key=lambda i: (i.split()[0] == 'from', i.split()[1]))
        items['imports'] = '\n'.join(sorted_imports) + '\n' if imports else ''
        if migration_imports:
            items['imports'] += '\n\n# Functions from the following migrations need manual copying.\n# Move them and any dependencies into this file, then update the\n# RunPython operations to refer to the local versions:\n# %s' % '\n# '.join(sorted(migration_imports))
        if self.migration.replaces:
            items['replaces_str'] = '\n    replaces = %s\n' % self.serialize(self.migration.replaces)[0]
        if self.include_header:
            items['migration_header'] = MIGRATION_HEADER_TEMPLATE % {'version': get_version(), 'timestamp': now().strftime('%Y-%m-%d %H:%M')}
        else:
            items['migration_header'] = ''
        if self.migration.initial:
            items['initial_str'] = '\n    initial = True\n'
        return MIGRATION_TEMPLATE % items

    @classmethod
    def serialize(cls, value):
        return serializer_factory(value).serialize()