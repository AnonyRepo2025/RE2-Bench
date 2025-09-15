from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _



class ContentType(Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(_("python model class name"), max_length=100)
    objects = ContentTypeManager()
    @property
    def app_labeled_name(self):
        model = self.model_class()
        if not model:
            return self.model
        return "%s | %s" % (
            model._meta.app_config.verbose_name,
            model._meta.verbose_name,
        )
    def model_class(self):
        """Return the model class for this type of content."""
        try:
            return apps.get_model(self.app_label, self.model)
        except LookupError:
            return None