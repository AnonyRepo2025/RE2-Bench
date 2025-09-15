import functools
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import Context, Template
from .context import _builtin_context_processors
from .exceptions import TemplateDoesNotExist
from .library import import_library
from django.template import engines
from django.template.backends.django import DjangoTemplates



class Engine:
    default_builtins = [
        'django.template.defaulttags',
        'django.template.defaultfilters',
        'django.template.loader_tags',
    ]
    @cached_property
    def template_loaders(self):
        return self.get_template_loaders(self.loaders)
    def get_template_loaders(self, template_loaders):
        loaders = []
        for template_loader in template_loaders:
            loader = self.find_template_loader(template_loader)
            if loader is not None:
                loaders.append(loader)
        return loaders
    def find_template_loader(self, loader):
        if isinstance(loader, (tuple, list)):
            loader, *args = loader
        else:
            args = []

        if isinstance(loader, str):
            loader_class = import_string(loader)
            return loader_class(self, *args)
        else:
            raise ImproperlyConfigured(
                "Invalid value in template loaders configuration: %r" % loader)
    def find_template(self, name, dirs=None, skip=None):
        tried = []
        for loader in self.template_loaders:
            try:
                template = loader.get_template(name, skip=skip)
                return template, template.origin
            except TemplateDoesNotExist as e:
                tried.extend(e.tried)
        raise TemplateDoesNotExist(name, tried=tried)
    def get_template(self, template_name):
        """
        Return a compiled Template object for the given template name,
        handling template inheritance recursively.
        """
        template, origin = self.find_template(template_name)
        if not hasattr(template, 'render'):
            # template needs to be compiled
            template = Template(template, origin, template_name, engine=self)
        return template
    def render_to_string(self, template_name, context=None):
        """
        Render the template specified by template_name with the given context.
        For use in Django's test suite.
        """
        if isinstance(template_name, (list, tuple)):
            t = self.select_template(template_name)
        else:
            t = self.get_template(template_name)
        # Django < 1.8 accepted a Context in `context` even though that's
        # unintended. Preserve this ability but don't rewrap `context`.
        if isinstance(context, Context):
            return t.render(context)
        else:
            return t.render(Context(context, autoescape=self.autoescape))