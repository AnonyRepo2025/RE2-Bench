from pathlib import Path
from django.dispatch import receiver
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils._os import to_path
from django.utils.autoreload import (
    autoreload_started, file_changed, is_django_path,
)



def template_changed(sender, file_path, **kwargs):
    if file_path.suffix == '.py':
        return
    for template_dir in get_template_directories():
        if template_dir in file_path.parents:
            reset_loaders()
            return True
