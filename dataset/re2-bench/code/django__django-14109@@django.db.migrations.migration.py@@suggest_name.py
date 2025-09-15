from django.db.migrations.utils import get_migration_name_timestamp
from django.db.transaction import atomic
from .exceptions import IrreversibleError



class Migration:
    operations = []
    dependencies = []
    run_before = []
    replaces = []
    initial = None
    atomic = True
    def suggest_name(self):
        """
        Suggest a name for the operations this migration might represent. Names
        are not guaranteed to be unique, but put some effort into the fallback
        name to avoid VCS conflicts if possible.
        """
        if self.initial:
            return 'initial'

        raw_fragments = [op.migration_name_fragment for op in self.operations]
        fragments = [name for name in raw_fragments if name]

        if not fragments or len(fragments) != len(self.operations):
            return 'auto_%s' % get_migration_name_timestamp()

        name = fragments[0]
        for fragment in fragments[1:]:
            new_name = f'{name}_{fragment}'
            if len(new_name) > 52:
                name = f'{name}_and_more'
                break
            name = new_name
        return name