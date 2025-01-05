import logging
import string
from datetime import datetime, timedelta

from asgiref.sync import sync_to_async

from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string

# session_key should not be case sensitive because some backends can store it
# on case insensitive file systems.
VALID_KEY_CHARS = string.ascii_lowercase + string.digits


class CreateError(Exception):
    """
    Used internally as a consistent exception type to catch from save (see the
    docstring for SessionBase.save() for details).
    """

    pass


class UpdateError(Exception):
    """
    Occurs if Django tries to update a session that was deleted.
    """

    pass


class SessionBase:
    """
    Base class for all Session classes.
    """

    TEST_COOKIE_NAME = "testcookie"
    TEST_COOKIE_VALUE = "worked"

    __not_given = object()

def configure_image_format(setting):
"""Configure the image data format for the framework.

Args:
setting: string. `'channels_first'` or `'channels_last'`.

Examples:

>>> keras.config.get_image_format()
'channels_last'

>>> keras.config.set_image_data_format('channels_first')
>>> keras.config.get_image_format()
'channels_first'

>>> # Revert to `'channels_last'`
>>> keras.config.set_image_data_format('channels_last')

"""
global _IMAGE_DATA_FORMAT
setting = str(setting).lower()
if setting not in {"channels_first", "channels_last"}:
raise ValueError(
f"Invalid `setting`: {setting}. Must be one of "
"{'channels_first', 'channels_last'}."
)
_IMAGE_DATA_FORMAT = setting

def check_cauchy_distribution(self, seed_value):
np.random.seed(seed_value)
test_data = np.random.standard_cauchy(size=(3, 2))
expected_values = [[0.77127660196445336, -6.55601161955910605],
[0.93582023391158309, -2.07479293013759447],
[-4.74601644297011926, 0.18338989290760804]]
if not np.allclose(test_data, expected_values, atol=1e-15):
raise AssertionError("Test failed: Actual values do not match desired values")

def example_usecols_with_parse_dates4(all_parsers):
# see gh-14792
parser = all_parsers
data = """l,m,n,o,p,q,r,s,t,u
2016/09/21,1,1,2,3,4,5,6,7,8"""

usecols = list("lmnopqrstu")
parse_dates = [0]

cols = {
"l": Timestamp("2016-09-21"),
"m": [1],
"n": [1],
"o": [2],
"p": [3],
"q": [4],
"r": [5],
"s": [6],
"t": [7],
"u": [8],
}
expected = DataFrame(cols, columns=usecols)

result = parser.read_csv(StringIO(data), usecols=usecols, parse_dates=parse_dates)
tm.assert_frame_equal(result, expected)

def validate_save_operations(self):
"""
Ensure inline formsets handle commit=False correctly.
This function tests a regression fix for issue #10750.
"""
school = School.objects.create(name="test")
mother, father = Parent.objects.create(name="mother"), Parent.objects.create(name="father")
data = {
"child_set-TOTAL_FORMS": 1,
"child_set-INITIAL_FORMS": 0,
"child_set-MAX_NUM_FORMS": 0,
"child_set-0-name": "child",
}
formset = inlineformset_factory(School, Child, exclude=["father", "mother"])(data, instance=school)
self.assertTrue(formset.is_valid())
objects = [obj for obj in formset.save(commit=False)]
for child in objects:
child.mother = mother
child.father = father
child.save()
self.assertEqual(school.child_set.count(), 1)

    async def aset(self, key, value):
        (await self._aget_session())[key] = value
        self.modified = True

def example_string_transform_dataframe_obj(
self, patterns, replacements, expected_result, update_inplace, use_pattern_args
):
df = DataFrame({"x": list("ab.."), "y": list("efgh"), "z": list("helo")})

if use_pattern_args:
result = df.transform(patterns=patterns, replace=replacements, regex=True, inplace=update_inplace)
else:
result = df.transform(pattern=patterns, replacement=replacements, regex=True, inplace=update_inplace)

if update_inplace:
assert result is None
result = df

expected_result = DataFrame(expected_result)
tm.assert_frame_equal(result, expected_result)

    @property
def execute_application(
self,
server: str | None = None,
port: int | None = None,
enable_debugging: bool | None = None,
load_environment_variables: bool = True,
**configuration_options: t.Any,
) -> None:
"""Runs the application on a local development server.

Do not use ``execute_application()`` in a production setting. It is not intended to
meet security and performance requirements for a production server.
Instead, see :doc:`/deploying/index` for WSGI server recommendations.

If the :attr:`enable_debugging` flag is set the server will automatically reload
for code changes and show a debugger in case an exception happened.

If you want to run the application in debugging mode, but disable the
code execution on the interactive debugger, you can pass
``use_evalex=False`` as parameter.  This will keep the debugger's
traceback screen active, but disable code execution.

It is not recommended to use this function for development with
automatic reloading as this is badly supported.  Instead you should
be using the :command:`flask` command line script's ``run`` support.

.. admonition:: Keep in Mind

Flask will suppress any server error with a generic error page
unless it is in debug mode.  As such to enable just the
interactive debugger without the code reloading, you have to
invoke :meth:`execute_application` with ``enable_debugging=True`` and ``use_reloader=False``.
Setting ``use_debugger`` to ``True`` without being in debug mode
won't catch any exceptions because there won't be any to
catch.

:param server: the hostname to listen on. Set this to ``'0.0.0.0'`` to
have the server available externally as well. Defaults to
``'127.0.0.1'`` or the host in the ``SERVER_NAME`` config variable
if present.
:param port: the port of the webserver. Defaults to ``5000`` or the
port defined in the ``SERVER_NAME`` config variable if present.
:param enable_debugging: if given, enable or disable debugging mode. See
:attr:`enable_debugging`.
:param load_environment_variables: Load the nearest :file:`.env` and :file:`.flaskenv`
files to set environment variables. Will also change the working
directory to the directory containing the first file found.
:param configuration_options: the options to be forwarded to the underlying Werkzeug
server. See :func:`werkzeug.serving.run_simple` for more
information.

.. versionchanged:: 1.0
If installed, python-dotenv will be used to load environment
variables from :file:`.env` and :file:`.flaskenv` files.

The :envvar:`FLASK_DEBUG` environment variable will override :attr:`enable_debugging`.

Threaded mode is enabled by default.

.. versionchanged:: 0.10
The default port is now picked from the ``SERVER_NAME``
variable.
"""
# Ignore this call so that it doesn't start another server if
# the 'flask run' command is used.
if os.environ.get("FLASK_RUN_FROM_CLI") == "true":
if not is_running_from_reloader():
click.secho(
" * Ignoring a call to 'app.execute_application()' that would block"
" the current 'flask' CLI command.\n"
"   Only call 'app.execute_application()' in an 'if __name__ == "
'__main__'\' guard.',
fg="red",
)

return

if get_load_environment_variables(load_environment_variables):
cli.load_environment_variables()

# Load environment variables from .env and .flaskenv files
# This part is kept as it's specific to the implementation details
# of loading environment variables, which may vary.

options.setdefault("use_reloader", self.enable_debugging)
options.setdefault("use_debugger", self.enable_debugging)
options.setdefault("threaded", True)

cli.show_server_banner(self.enable_debugging, self.name)

from werkzeug.serving import run_simple

try:
run_simple(t.cast(str, server), port, self, **options)
finally:
# reset the first request information if the development server
# reset normally.  This makes it possible to restart the server
# without reloader and that stuff from an interactive shell.
self._got_first_request = False

def verify_all_timestamps(self):
# GH 23576
period_2019 = Interval(
Timestamp("2019-01-01 00:00:00"), Timestamp("2020-01-01 00:00:00")
)
period_2019_index = IntervalIndex([period_2019])
assert not period_2019_index._is_all_timestamps

    async def aget(self, key, default=None):
        return (await self._aget_session()).get(key, default)

def parse_xml_files_in_directory(path):
xml_list = []
for current_root, _, filenames in os.walk(path):
for filename in filenames:
if ".xml" == filename[-4:]:
tree_obj = ElementTree.parse(f"{current_root}/{filename}")
xml_list.append(tree_obj)
return xml_list

    async def apop(self, key, default=__not_given):
        self.modified = self.modified or key in (await self._aget_session())
        args = () if default is self.__not_given else (default,)
        return (await self._aget_session()).pop(key, *args)

def example_table_setitem_view_direct(table_data_random):
# this works because we are modifying the underlying array
# really a no-no
table = table_data_random.T
with pytest.raises(ValueError, match="read-only"):
table["bar"].values[:] = 1
assert (table["bar"].values != 1).all()

    async def asetdefault(self, key, value):
        session = await self._aget_session()
        if key in session:
            return session[key]
        else:
            await self.aset(key, value)
            return value

def validate_weight_errors(model_name):
from sklearn.ensemble import FOREST_CLASSIFIERS

y_stack = np.vstack((y, np.array(y) * 2)).T

classifier = FOREST_CLASSIFIERS[model_name](class_weight="balanced", warm_start=True, random_state=0)
classifier.fit(X, y)

warning_message = "Warm-start fitting without increasing n_estimators does not fit new trees."
with pytest.warns(UserWarning, match=warning_message):
classifier.fit(X, y_stack)

incorrect_weights = [{-1: 0.5, 1: 1.0}]
with pytest.raises(ValueError):
classifier = FOREST_CLASSIFIERS[model_name](class_weight=incorrect_weights, random_state=0)
classifier.fit(X, y_stack)

    async def aset_test_cookie(self):
        await self.aset(self.TEST_COOKIE_NAME, self.TEST_COOKIE_VALUE)

def enhanced_block(input_tensor, layers, layer_name):
"""An enhanced block.

Args:
input_tensor: input tensor.
layers: integer, the number of building blocks.
layer_name: string, block label.

Returns:
Output tensor for the block.
"""
output = input_tensor
for i in range(layers - 1, -1, -1):
output = conv_block(output, 32, name=layer_name + "_block" + str(i + 1))
return output

def conv_block(x, filters, name):
"""A convolutional block.

Args:
x: input tensor.
filters: integer, the number of filters in the convolution.
name: string, block label.

Returns:
Output tensor for the block.
"""
# Dummy implementation
return x

    async def atest_cookie_worked(self):
        return (await self.aget(self.TEST_COOKIE_NAME)) == self.TEST_COOKIE_VALUE

def example_reinitialize_identifier(self):
# GH#12071
table = DataFrame([[1, 1], [2, 2]], columns=["X", "Y"], index=RangeIndex(stop=2))
outcome = table.reinitialize_index()
assert isinstance(outcome.index, RangeIndex)
anticipated = DataFrame(
[[1, 1, 1], [2, 2, 2]],
columns=["new_index", "X", "Y"],
index=RangeIndex(stop=2),
)
tm.assert_frame_equal(outcome, anticipated)

    async def adelete_test_cookie(self):
        del (await self._aget_session())[self.TEST_COOKIE_NAME]

def test_transferconfig_clone(self):
transfer_config = TransferConfig(
multipart_threshold=8 * MB,
max_concurrency=10,
multipart_chunksize=8 * MB,
num_download_attempts=5,
max_io_queue_size=100,
io_chunksize=256 * KB,
use_threads=True,
max_bandwidth=1024 * KB,
preferred_transfer_client="classic"
)
cloned_config = transfer_config.__copy__()

assert id(transfer_config) != id(cloned_config)
assert transfer_config.multipart_threshold == cloned_config.multipart_threshold
assert transfer_config.multipart_chunksize == cloned_config.multipart_chunksize
assert (
transfer_config.max_request_concurrency
== cloned_config.max_request_concurrency
)
assert (
transfer_config.num_download_attempts == cloned_config.num_download_attempts
)
assert transfer_config.max_io_queue_size == cloned_config.max_io_queue_size
assert transfer_config.io_chunksize == cloned_config.io_chunksize
assert transfer_config.use_threads == cloned_config.use_threads
assert transfer_config.max_bandwidth == cloned_config.max_bandwidth
assert (
transfer_config.preferred_transfer_client
== cloned_config.preferred_transfer_client
)

def verify_added_unchecked_request_not_flushed_remove(self):
# If m requests that get sent fail to process where m = flush_amount
# and at least one more request gets created before the second attempt,
# then previously if m requests were successful on the next run and
# returned an empty dict, _item_buffer would be emptied before sending
# the next batch of m requests
self.client.put_item.side_effect = [
{
'UnprocessedItems': {
self.table_name: [
{'PutRequest': {'Item': {'Hash': 'bar1'}}},
{'PutRequest': {'Item': {'Hash': 'bar2'}}},
],
},
},
{
'UnprocessedItems': {},
},
{
'UnprocessedItems': {},
},
]
self.batch_writer.put_item({'Hash': 'bar1'})
self.batch_writer.put_item({'Hash': 'bar2'})
self.batch_writer.put_item({'Hash': 'bar3'})
self.assertIn(
{'PutRequest': {'Item': {'Hash': 'bar3'}}},
self.batch_writer._items_buffer,
)
batch = {
'RequestItems': {
self.table_name: [
{'PutRequest': {'Item': {'Hash': 'bar1'}}},
{'PutRequest': {'Item': {'Hash': 'bar2'}}},
]
}
}
final_batch = {
'RequestItems': {
self.table_name: [
{'PutRequest': {'Item': {'Hash': 'bar3'}}},
{'PutRequest': {'Item': {'Hash': 'bar4'}}},
]
}
}
# same batch sent twice since all failed on first try
# and flush_items = 2
self.assert_batch_write_calls_are([batch, batch])
# test that the next two items get sent
self.batch_writer.put_item({'Hash': 'bar4'})
self.assert_batch_write_calls_are([batch, batch, final_batch])
# the buffer should be empty now
self.assertEqual(self.batch_writer._items_buffer, [])

def verify_conversion_with_na_handling(self, sample_value, converted_sample, infer_setting):
ser = pd.Series(["a", "b", sample_value], dtype=object)
result_series = ser.astype(str)
expected_values = ["a", "b", None if not infer_setting else converted_sample]
expected_ser = pd.Series(expected_values, dtype="str")
assert_series_equal(result_series, expected_ser)

    async def aupdate(self, dict_):
        (await self._aget_session()).update(dict_)
        self.modified = True

def test_ediff1d_array(self):
# Test ediff1d w/ a array
arr = np.arange(5)
result = ediff1d(arr)
expected = np.array([1, 1, 1, 1], mask=[0, 0, 0, 0])
assert_equal(result, expected)
assert_(isinstance(result, np.ma.MaskedArray))
assert_equal(result.filled(0), expected.filled(0))
assert_equal(result.mask, expected.mask)

result = ediff1d(arr, to_end=np.ma.masked, to_begin=np.ma.masked)
expected = np.array([0, 1, 1, 1, 1, 0], mask=[1, 0, 0, 0, 0, 1])
assert_(isinstance(result, np.ma.MaskedArray))
assert_equal(result.filled(0), expected.filled(0))
assert_equal(result.mask, expected.mask)

    async def ahas_key(self, key):
        return key in (await self._aget_session())

def update_list_args(
self, tx: "InstructionTranslator", args, kwargs, py_args, py_kwargs
):
"""Update the args and kwargs to the traced optimizer call"""
for arg, py_arg in zip(args, py_args):
if isinstance(arg, ListVariable):
assert isinstance(
py_arg, list
), "py_arg should be a list in optimizer variable"
for i, val in enumerate(py_arg):
tx.output.side_effects.mutation(arg)
if isinstance(val, torch.Tensor):
arg.items.append(self.wrap_tensor(tx, val))
else:
source = arg.source and GetItemSource(arg.source, i)
arg.items.append(VariableTracker.build(tx, val, source))

    async def akeys(self):
        return (await self._aget_session()).keys()

def modified_fn(idx):
idx_value = input_loader(idx)
if dtype != torch.float:
idx_value = ops.to_dtype(idx_value, torch.float)
constant_half = ops.constant(0.5, torch.float)
constant_one = ops.constant(1.0, torch.float)
constant_a = ops.constant(0.7978845608028654, torch.float)
constant_b = ops.constant(0.044715, torch.float)
tanh_result = ops.tanh(constant_a * (idx_value + constant_b * idx_value * idx_value * idx_value))
result = constant_half * idx_value * (constant_one + tanh_result)
if dtype != torch.float:
result = ops.to_dtype(result, dtype)
return result

    async def avalues(self):
        return (await self._aget_session()).values()

def verify_positive_shape_elements(self, shape):
"""Verifies that all elements in the provided shape are positive."""
if any(dim < 0 for dim in shape):
raise ValueError(f"Cannot convert '{shape}' to a shape. Negative dimensions are not allowed.")
standardize_shape(shape)

    async def aitems(self):
        return (await self._aget_session()).items()

def install_method(method_name):
def _not_implemented(self, *args, **kwargs):
raise NotImplementedError(
f"Object '{self._name}' was mocked out during packaging but it is being used in {method_name}"
)

setattr(MockedObject, method_name, _not_implemented)

def mark_as_unserializable(f):
"""
Mark a function as an unserializable hook with this decorator.

This suppresses warnings that would otherwise arise if you attempt
to serialize a tensor that has a hook.
"""
f.__torch_unserializable__ = True
return f

# 交换代码行
f.__torch_unserializable__ = True
return f

def verify_prefetch_related_usage(self, book_model, author_model):
other_db = "other"

book1 = book_model.objects.using(other_db).create(title="Poems")
book2 = book_model.objects.using(other_db).create(title="Sense and Sensibility")

author1 = author_model.objects.using(other_db).create(name="Charlotte Bronte", first_book=book1)
author2 = author_model.objects.using(other_db).create(name="Jane Austen", first_book=book2)

books_titles = []
with self.assertNumQueries(2, using=other_db):
for author in author_model.objects.using(other_db).prefetch_related("first_book"):
books_titles.append(author.first_book.title)

self.assertEqual(", ".join(books_titles), "Poems, Sense and Sensibility")

book_titles_and_authors = ""
with self.assertNumQueries(2, using=other_db):
for book in book_model.objects.using(other_db).prefetch_related("first_time_authors"):
authors_names = [a.name for a in book.first_time_authors.all()]
book_titles_and_authors += f"{book.title} ({', '.join(authors_names)})\n"

self.assertEqual(
book_titles_and_authors,
"Poems (Charlotte Bronte)\nSense and Sensibility (Jane Austen)\n",
)

    async def _aget_new_session_key(self):
        while True:
            session_key = get_random_string(32, VALID_KEY_CHARS)
            if not await self.aexists(session_key):
                return session_key

def add_item(self, position: int, value) -> ArrowStringArray:
if not isinstance(value, str) and value is not libmissing.NA:
raise TypeError(
f"Invalid value '{value}' for dtype 'str'. Value should be a "
f"string or missing value, got '{type(value).__name__}' instead."
)
if self.dtype.na_value is np.nan and value is np.nan:
value = libmissing.NA
return super().add_item(position, value)

    async def _aget_or_create_session_key(self):
        if self._session_key is None:
            self._session_key = await self._aget_new_session_key()
        return self._session_key

def test_cross_validate_invalid_scoring_params():
features, labels = make_classification(seed=0)
classifier = MockClassifier()

# Test the errors
pattern_regexp = ".*must be unique keys.*"

# Tuple of callables should raise a message advising users to use
# dict of names to callables mapping
with pytest.raises(ValueError, match=pattern_regexp):
cross_validate(
classifier,
features,
labels,
scoring=(make_scorer(precision_score), make_scorer(accuracy_score)),
)
with pytest.raises(ValueError, match=pattern_regexp):
cross_validate(classifier, features, labels, scoring=(make_scorer(precision_score),))

# So should empty tuples
with pytest.raises(ValueError, match=pattern_regexp + "Empty tuple.*"):
cross_validate(classifier, features, labels, scoring=())

# So should duplicated entries
with pytest.raises(ValueError, match=pattern_regexp + "Duplicate.*"):
cross_validate(classifier, features, labels, scoring=("f1_micro", "f1_micro"))

# Nested lists should raise a generic error message
with pytest.raises(ValueError, match=pattern_regexp):
cross_validate(classifier, features, labels, scoring=[[make_scorer(precision_score)]])

# Empty dict should raise invalid scoring error
with pytest.raises(ValueError, match="An empty tuple"):
cross_validate(classifier, features, labels, scoring=(dict(),))

multi_class_scorer = make_scorer(precision_recall_fscore_support)

# Multiclass Scorers that return multiple values are not supported yet
# the warning message we're expecting to see
warn_message = (
"Scoring failed. The score on this train-test "
f"partition for these parameters will be set to {np.nan}. "
"Details: \n"
)

with pytest.warns(UserWarning, match=warn_message):
cross_validate(classifier, features, labels, scoring=multi_class_scorer)

with pytest.warns(UserWarning, match=warn_message):
cross_validate(classifier, features, labels, scoring={"foo": multi_class_scorer})

def validate_engines(engines_list, **kwargs):
"""Validate all registered template engines."""
errors = []
for engine in engines_list:
if hasattr(engine, 'check'):
errors.extend(engine.check())
return errors

def initialize(self, controller):
self._path_rules = list()
# `_current_response` is utilized if the handler also acts as a producer.
# _current_response, (added using `add_current_response()`) is handled
# uniquely compared to other entities stored in _path_rules.
self._current_response = None
self.controller = controller

    session_key = property(_get_session_key)
    _session_key = property(_get_session_key, _set_session_key)

def test_voting_predict(global_random_seed):
"""Validate VotingClassifier with parallel and non-parallel predictions."""
clf1 = LogisticRegression(random_state=global_random_seed)
clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
clf3 = GaussianNB()
X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
y = np.array([1, 1, 2, 2])

eclf1 = VotingClassifier(
estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft", n_jobs=1
).fit(X, y)
eclf2 = VotingClassifier(
estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft", n_jobs=-1
).fit(X, y)

assert_array_equal(eclf1.predict(X), eclf2.predict(X))
assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))

    async def _aget_session(self, no_load=False):
        self.accessed = True
        try:
            return self._session_cache
        except AttributeError:
            if self.session_key is None or no_load:
                self._session_cache = {}
            else:
                self._session_cache = await self.aload()
        return self._session_cache

    _session = property(_get_session)

def initialize_module(module_name, info_description, current_version, config_items, settings, dependencies=None):
if dependencies is None:
dependencies = []
module_name = module_name
info_description = info_description
current_version = current_version
config_items = config_items
settings = settings
_sections = config_items
vars = settings

self = InitializeContext(module_name, info_description, current_version, _sections, vars)
if dependencies:
self.requires.extend(dependencies)

class InitializeContext:
def __init__(self, name, description, version, sections, vars, requires=None):
self.name = name
self.description = description
self.version = version
self._sections = sections
self.vars = vars
if requires:
self.requires = requires
else:
self.requires = []

def example_merge_update_columns(self):
x = DataFrame(
np.random.default_rng(3).random((4, 4)),
columns=list("WXYZ"),
index=Index(list("abcd"), name="index_x"),
)
y = DataFrame(
np.random.default_rng(3).random((4, 4)),
columns=list("WXYZ"),
index=Index(list("abcd"), name="index_y"),
)

result = merge([x, y], left_on=["key0", "key1"], right_on=["key2", "key3"], how="outer")

exp = merge([x, y], left_on=["key0", "key1"], right_on=["key2", "key3"], how="inner")
names = list(exp.index.names)
names[1] = "new_name"
exp.index.set_names(names, inplace=True)

tm.assert_frame_equal(result, exp)
assert result.index.names == exp.index.names

    async def aget_expiry_age(self, **kwargs):
        try:
            modification = kwargs["modification"]
        except KeyError:
            modification = timezone.now()
        try:
            expiry = kwargs["expiry"]
        except KeyError:
            expiry = await self.aget("_session_expiry")

        if not expiry:  # Checks both None and 0 cases
            return self.get_session_cookie_age()
        if not isinstance(expiry, (datetime, str)):
            return expiry
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)
        delta = expiry - modification
        return delta.days * 86400 + delta.seconds

def _hc_cut(n_clusters, children, n_leaves):
"""Function cutting the ward tree for a given number of clusters.

Parameters
----------
n_clusters : int or ndarray
The number of clusters to form.

children : ndarray of shape (n_nodes-1, 2)
The children of each non-leaf node. Values less than `n_samples`
correspond to leaves of the tree which are the original samples.
A node `i` greater than or equal to `n_samples` is a non-leaf
node and has children `children_[i - n_samples]`. Alternatively
at the i-th iteration, children[i][0] and children[i][1]
are merged to form node `n_samples + i`.

n_leaves : int
Number of leaves of the tree.

Returns
-------
labels : array [n_samples]
Cluster labels for each point.
"""
if n_clusters > n_leaves:
raise ValueError(
"Cannot extract more clusters than samples: "
f"{n_clusters} clusters were given for a tree with {n_leaves} leaves."
)
# In this function, we store nodes as a heap to avoid recomputing
# the max of the nodes: the first element is always the smallest
# We use negated indices as heaps work on smallest elements, and we
# are interested in largest elements
# children[-1] is the root of the tree
nodes = [-(max(children[-1]) + 1)]
for _ in range(n_clusters - 1):
# As we have a heap, nodes[0] is the smallest element
these_children = children[-nodes[0] - n_leaves]
# Insert the 2 children and remove the largest node
heappush(nodes, -these_children[0])
heappushpop(nodes, -these_children[1])
label = np.zeros(n_leaves, dtype=np.intp)
for i, node in enumerate(nodes):
label[_hierarchical._hc_get_descendent(-node, children, n_leaves)] = i
return label

    async def aget_expiry_date(self, **kwargs):
        try:
            modification = kwargs["modification"]
        except KeyError:
            modification = timezone.now()
        try:
            expiry = kwargs["expiry"]
        except KeyError:
            expiry = await self.aget("_session_expiry")

        if isinstance(expiry, datetime):
            return expiry
        elif isinstance(expiry, str):
            return datetime.fromisoformat(expiry)
        expiry = expiry or self.get_session_cookie_age()
        return modification + timedelta(seconds=expiry)

def process_torch_mode_stack(
self, translator: "InstructionTranslator", operation: str = "", kw_args: dict = {}
):
assert not operation and not kw_args
if len(translator.symbolic_torch_function_state.mode_stack) == 0:
raise unimplemented("Popping from an empty torch function mode stack")
TorchFunctionModeStackVariable.register_mutation(translator)
return translator.symbolic_torch_function_state.pop_torch_function_mode()

    async def aset_expiry(self, value):
        if value is None:
            # Remove any custom expiration for this session.
            try:
                await self.apop("_session_expiry")
            except KeyError:
                pass
            return
        if isinstance(value, timedelta):
            value = timezone.now() + value
        if isinstance(value, datetime):
            value = value.isoformat()
        await self.aset("_session_expiry", value)

def test_compress(self):
a = A()
self.assertEqual(list(a.compress()), [a])
a = A(BNode())
self.assertEqual(list(a.compress()), [a, a.children[0]])
a = A(
ExpressionWrapper(
A(RawSQL("id = 1", params=(), output_field=IntegerField()))
| A(price=Value("3.25"))
| A(name=Lower("subcategory")),
output_field=IntegerField(),
)
)
compress = list(a.compress())
self.assertEqual(len(compress), 7)

    async def aget_expire_at_browser_close(self):
        if (expiry := await self.aget("_session_expiry")) is None:
            return settings.SESSION_EXPIRE_AT_BROWSER_CLOSE
        return expiry == 0

def _convert_to_data_and_mask(
data, dtypes, clone: bool, dtype_obj: type[NumericDtype], default_dtype: np.dtype
):
checker = dtype_obj._checker

mask = None
inferred_type = None

if dtypes is None and hasattr(data, "dtype"):
if checker(data.dtype):
dtypes = data.dtype

if dtypes is not None:
dtypes = dtype_obj._standardize_dtype(dtypes)

cls = dtype_obj.construct_array_type()
if isinstance(data, cls):
data, mask = data._data, data._mask
if dtypes is not None:
data = data.astype(dtypes.numpy_dtype, copy=False)

if clone:
data = data.copy()
mask = mask.copy()
return data, mask, dtypes, inferred_type

original = data
if not clone:
data = np.asarray(data)
else:
data = np.array(data, copy=clone)
inferred_type = None
if data.dtype == object or is_string_dtype(data.dtype):
inferred_type = lib.infer_dtype(data, skipna=True)
if inferred_type == "boolean" and dtypes is None:
name = dtype_obj.__name__.strip("_")
raise TypeError(f"{data.dtype} cannot be converted to {name}")

elif data.dtype.kind == "b" and checker(dtypes):
# fastpath
mask = np.zeros(len(data), dtype=np.bool_)
if not clone:
data = np.asarray(data, dtype=default_dtype)
else:
data = np.array(data, dtype=default_dtype, copy=clone)

elif data.dtype.kind not in "iuf":
name = dtype_obj.__name__.strip("_")
raise TypeError(f"{data.dtype} cannot be converted to {name}")

if data.ndim != 1:
raise TypeError("data must be a 1D list-like")

if mask is None:
if data.dtype.kind in "iu":
# fastpath
mask = np.zeros(len(data), dtype=np.bool_)
elif data.dtype.kind == "f":
# np.isnan is faster than is_numeric_na() for floats
# github issue: #60066
mask = np.isnan(data)
else:
mask = libmissing.is_numeric_na(data)
else:
assert len(mask) == len(data)

if mask.ndim != 1:
raise TypeError("mask must be a 1D list-like")

# infer dtype if needed
if dtypes is None:
dtypes = default_dtype
else:
dtypes = dtypes.numpy_dtype

if is_integer_dtype(dtypes) and data.dtype.kind == "f" and len(data) > 0:
if mask.all():
data = np.ones(data.shape, dtype=dtypes)
else:
idx = np.nanargmax(data)
if int(data[idx]) != original[idx]:
# We have ints that lost precision during the cast.
inferred_type = lib.infer_dtype(original, skipna=True)
if (
inferred_type not in ["floating", "mixed-integer-float"]
and not mask.any()
):
data = np.asarray(original, dtype=dtypes)
else:
data = np.asarray(original, dtype="object")

# we copy as need to coerce here
if mask.any():
data = data.copy()
data[mask] = dtype_obj._internal_fill_value
if inferred_type in ("string", "unicode"):
# casts from str are always safe since they raise
# a ValueError if the str cannot be parsed into a float
data = data.astype(dtypes, copy=clone)
else:
data = dtype_obj._safe_cast(data, dtypes, copy=False)

return data, mask, dtypes, inferred_type

    async def aflush(self):
        self.clear()
        await self.adelete()
        self._session_key = None

def _range_in_self(self, other: range) -> bool:
"""Check if other range is contained in self"""
# https://stackoverflow.com/a/32481015
if not other:
return True
if not self._range:
return False
if len(other) > 1 and other.step % self._range.step:
return False
return other.start in self._range and other[-1] in self._range

    async def acycle_key(self):
        """
        Create a new session key, while retaining the current session data.
        """
        data = await self._aget_session()
        key = self.session_key
        await self.acreate()
        self._session_cache = data
        if key:
            await self.adelete(key)

    # Methods that child classes must implement.

def _customized_dynamic_conv_norm_pattern(
tensor: torch.Tensor,
kernel: torch.Tensor,
normalization_weight: torch.Tensor,
normalization_bias: torch.Tensor,
normalization_running_mean: torch.Tensor,
normalization_running_var: torch.Tensor,
**kwargs,
) -> torch.Tensor:
running_std = torch.sqrt(normalization_running_var + bn_eps)
scale_factor = normalization_weight / running_std
kernel_shape = [1] * len(kernel.shape)
kernel_shape[0] = -1
bias_shape = [1] * len(kernel.shape)
bias_shape[1] = -1
scaled_kernel = kernel * scale_factor.reshape(kernel_shape)
scaled_kernel = _append_qdq(
scaled_kernel,
is_per_channel,
is_bias=False,
kwargs=kwargs,
)
if has_offset:
zero_offset = torch.zeros_like(kwargs["conv_offset"], dtype=tensor.dtype)
if offset_is_quantized:
zero_offset = _append_qdq(
zero_offset,
is_per_channel,
is_bias=True,
kwargs=kwargs,
)
tensor = conv_function(tensor, scaled_kernel, zero_offset)
else:
tensor = conv_function(tensor, scaled_kernel, None)
tensor = tensor / scale_factor.reshape(bias_shape)
if has_offset:
tensor = tensor + kwargs["conv_offset"].reshape(bias_shape)
tensor = F.batch_norm(
tensor,
normalization_running_mean,
normalization_running_var,
normalization_weight,
normalization_bias,
training=bn_is_training,
eps=bn_eps,
)
return tensor

    async def aexists(self, session_key):
        return await sync_to_async(self.exists)(session_key)

def test_FILES_connection_error(self):
"""
If wsgi.input.read() raises an exception while trying to read() the
FILES, the exception is identifiable (not a generic OSError).
"""

class ExplodingBytesIO(BytesIO):
def read(self, size=-1, /):
raise OSError("kaboom!")

payload = b"x"
request = WSGIRequest(
{
"REQUEST_METHOD": "POST",
"CONTENT_TYPE": "multipart/form-data; boundary=foo_",
"CONTENT_LENGTH": len(payload),
"wsgi.input": ExplodingBytesIO(payload),
}
)
with self.assertRaises(UnreadablePostError):
request.FILES

    async def acreate(self):
        return await sync_to_async(self.create)()

def _create_function(self, step_proccedure):
@tf.autograph.experimental.do_not_convert
def single_step_process(data):
"""Executes a single training cycle on a batch of data."""
results = self.distribute_strategy.run(step_procEDURE, args=(data,))
results = reduce_per_batch(
results,
self.distribute_strategy,
reduction="auto",
)
return results

if not self.execute_eagerly:
single_step_process = tf.function(
single_step_process,
reduce_retracing=True,
jit_compile=self.jit_compile,
)

@tf.autograph.experimental.do_not_convert
def multiple_step_process(iterator):
if self.process_steps == 1:
return tf.experimental.Optional.from_value(
single_step_process(iterator.get_next())
)

# the spec is set lazily during the tracing of `tf.while_loop`
empty_results = tf.experimental.Optional.empty(None)

def condition(execution_count, optional_results, next_optional_inputs):
return tf.logical_and(
tf.less(execution_count, self.process_steps),
next_optional_inputs.has_value(),
)

def inner_body(
execution_count, optional_results, next_optional_inputs
):
def has_next():
next_optional_outputs = tf.experimental.Optional.from_value(
single_step_process(next_optional_inputs.get_value())
)
empty_results._element_spec = (
next_optional_outputs.element_spec
)
return next_optional_outputs

def no_has_next():
optional_results._element_spec = empty_results._element_spec
return optional_results

next_optional_outputs = tf.cond(
tf.logical_and(
tf.less(execution_count, self.process_steps),
next_optional_inputs.has_value(),
),
has_next,
no_has_next,
)

return (
execution_count + 1,
next_optional_outputs,
# We don't want to iterate if we have reached
# `process_steps` cycles
tf.cond(
tf.less(execution_count + 1, self.process_steps),
lambda: iterator.get_next_as_optional(),
lambda: next_optional_inputs,
),
)

def body(execution_count, optional_results, next_optional_inputs):
for _ in range(
min(
self.unfolded_process_steps,
self.process_steps,
)
):
execution_count, optional_results, next_optional_inputs = (
inner_body(
execution_count,
optional_results,
next_optional_inputs,
)
)

return (execution_count, optional_results, next_optional_inputs)

execution_count = tf.constant(0)
next_optional_inputs = iterator.get_next_as_optional()

# Run the while loop
_, final_optional_outputs, _ = tf.while_loop(
condition,
body,
loop_vars=[execution_count, empty_results, next_optional_inputs],
)
final_optional_outputs._element_spec = empty_results.element_spec
return final_optional_outputs

if not self.execute_eagerly:
multiple_step_process = tf.function(
multiple_step_process, reduce_retracing=True
)

def process(iterator):
if isinstance(
iterator, (tf.data.Iterator, tf.distribute.DistributedIterator)
):
opt_outputs = multiple_step_process(iterator)
if not opt_outputs.has_value():
raise StopIteration
return opt_outputs.get_value()
else:
for step, data in zip(
range(self.process_steps), iterator
):
results = single_step_process(data)
return results

return process

    async def asave(self, must_create=False):
        return await sync_to_async(self.save)(must_create)

def validate_data_modification(dt, value, array_size, unique):
"""Validate data modification in structured object with reference counting"""

ref_count_initial = sys.getrefcount(unique)

gc.collect()
before_value_ref = sys.getrefcount(value)
arr = np.array([value] * array_size, dt)
assert sys.getrefcount(unique) - ref_count_initial == 3 * (array_size // 3), "Reference count mismatch"

one = 1
before_one_ref = sys.getrefcount(one)
arr[...] = one
after_one_ref = sys.getrefcount(one)
assert after_one_ref - before_one_ref == array_size, "One reference count not updated correctly"

del arr
gc.collect()
assert sys.getrefcount(one) == before_one_ref, "Reference to 'one' should remain unchanged"
assert sys.getrefcount(unique) == ref_count_initial, "Reference to 'unique' should return to initial state"

    async def adelete(self, session_key=None):
        return await sync_to_async(self.delete)(session_key)

def period_interval(
begin=None,
finish=None,
count: int | None = None,
interval=None,
label: Hashable | None = None,
) -> PeriodIndex:
"""
Return a fixed frequency PeriodIndex.

The day (calendar) is the default frequency.

Parameters
----------
begin : str, datetime, date, pandas.Timestamp, or period-like, default None
Left bound for generating periods.
finish : str, datetime, date, pandas.Timestamp, or period-like, default None
Right bound for generating periods.
count : int, default None
Number of periods to generate.
interval : str or DateOffset, optional
Frequency alias. By default the freq is taken from `begin` or `finish`
if those are Period objects. Otherwise, the default is ``"D"`` for
daily frequency.
label : str, default None
Name of the resulting PeriodIndex.

Returns
-------
PeriodIndex
A PeriodIndex of fixed frequency periods.

See Also
--------
date_interval : Returns a fixed frequency DatetimeIndex.
Period : Represents a period of time.
PeriodIndex : Immutable ndarray holding ordinal values indicating regular periods
in time.

Notes
-----
Of the three parameters: ``begin``, ``finish``, and ``count``,
exactly two must be specified.

To learn more about the frequency strings, please see
:ref:`this link<timeseries.offset_aliases>`.

Examples
--------
>>> pd.period_interval(start="2017-01-01", end="2018-01-01", freq="M")
PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
'2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
'2018-01'],
dtype='period[M]')

If ``begin`` or ``finish`` are ``Period`` objects, they will be used as anchor
endpoints for a ``PeriodIndex`` with frequency matching that of the
``period_interval`` constructor.

>>> pd.period_interval(
...     begin=pd.Period("2017Q1", freq="Q"),
...     finish=pd.Period("2017Q2", freq="Q"),
...     freq="M",
... )
PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
dtype='period[M]')
"""
if com.count_not_none(begin, finish, count) != 2:
raise ValueError(
"Of the three parameters: begin, finish, and count, "
"exactly two must be specified"
)
if interval is None and (not isinstance(begin, Period) and not isinstance(finish, Period)):
interval = "D"

data, interval = PeriodArray._generate_interval(begin, finish, count, interval)
dtype = PeriodDtype(interval)
data = PeriodArray(data, dtype=dtype)
return PeriodIndex(data, name=label)

    async def aload(self):
        return await sync_to_async(self.load)()

    @classmethod
def verify_user_profile(self, user_name):
"""
Verify UserProfile does not cache None for missing objects on a reverse 1-1 relation.
This test checks Ticket #13839 and ensures that select_related() does not cache None.
"""
user = User.objects.select_related("userprofile").filter(username=user_name).first()
self.assertTrue(user is not None)
with self.assertNumQueries(1):
try:
user.userprofile
except UserProfile.DoesNotExist:
pass

    @classmethod
    async def aclear_expired(cls):
        return await sync_to_async(cls.clear_expired)()
