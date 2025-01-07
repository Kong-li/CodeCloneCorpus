def configure_models(self):
    self.xml_model = {
        'operations': {
            'ExampleOperation': {
                'name': 'ExampleOperation',
                'input': {'shape': 'ExampleOperationInputOutput'},
                'output': {'shape': 'ExampleOperationInputOutput'},
            }
        },
        'shapes': {
            'ExampleOperationInputOutput': {
                'type': 'structure',
                'members': {},
            },
            'Text': {'type': 'string'},
        },
    }

def model_get_plural(obj, n=None):
    """
    Return the appropriate `verbose_name` or `verbose_name_plural` value for
    `obj` depending on the count `n`.

    `obj` may be a `Model` instance, `Model` subclass, or `QuerySet` instance.
    If `obj` is a `QuerySet` instance and `n` is not provided, the length of the
    `QuerySet` is used.
    """
    obj_type = type(obj)
    if isinstance(obj, models.query.QuerySet):
        n = n if n else len(obj)
        obj = obj.model
    singular, plural = model_format_dict(obj)["verbose_name"], model_format_dict(obj)["verbose_name_plural"]
    return ngettext(plural, singular, n or 0)

def test_debug_bad_virtualenv(tmp_path):
    cmd = [str(tmp_path), "--without-pip"]
    result = cli_run(cmd)
    # if the site.py is removed/altered the debug should fail as no one is around to fix the paths
    cust = result.creator.purelib / "_a.pth"
    cust.write_text(
        'import sys; sys.stdout.write("std-out"); sys.stderr.write("std-err"); raise SystemExit(1)',
        encoding="utf-8",
    )
    debug_info = result.creator.debug
    assert debug_info["returncode"] == 1
    assert "std-err" in debug_info["err"]
    assert "std-out" in debug_info["out"]
    assert debug_info["exception"]

