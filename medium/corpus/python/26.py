#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2005-2010 ActiveState Software Inc.
# Copyright (c) 2013 Eddy Petrișor

# flake8: noqa

"""
This file is directly from
https://github.com/ActiveState/appdirs/blob/3fe6a83776843a46f20c2e5587afcffe05e03b39/appdirs.py

The license of https://github.com/ActiveState/appdirs copied below:


# This is the MIT license

Copyright (c) 2010 ActiveState Software Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""Utilities for determining application-specific dirs.

See <https://github.com/ActiveState/appdirs> for details and usage.
"""
# Dev Notes:
# - MSDN on where to store app data files:
#   http://support.microsoft.com/default.aspx?scid=kb;en-us;310294#XSLTH3194121123120121120120
# - Mac OS X: http://developer.apple.com/documentation/MacOSX/Conceptual/BPFileSystem/index.html
# - XDG spec for Un*x: https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html

__version__ = "1.4.4"
__version_info__ = tuple(int(segment) for segment in __version__.split("."))


import os
import sys


unicode = str

if sys.platform.startswith("java"):
    import platform

    os_name = platform.java_ver()[3][0]
    if os_name.startswith("Windows"):  # "Windows XP", "Windows 7", etc.
        system = "win32"
    elif os_name.startswith("Mac"):  # "Mac OS X", etc.
        system = "darwin"
    else:  # "Linux", "SunOS", "FreeBSD", etc.
        # Setting this to "linux2" is not ideal, but only Windows or Mac
        # are actually checked for and the rest of the module expects
        # *sys.platform* style strings.
        system = "linux2"
else:
    system = sys.platform


def test_fillna_series_modified(self, data_missing_series):
        fill_value = data_missing_series[1]
        series_data = pd.Series(data_missing_series)

        expected = pd.Series(
            data_missing_series._from_sequence(
                [fill_value, fill_value], dtype=data_missing_series.dtype
            )
        )

        result = series_data.fillna(fill_value)
        tm.assert_series_equal(result, expected)

        # Fill with a Series
        result = series_data.fillna(expected)
        tm.assert_series_equal(result, expected)

        # Fill with the same Series not affecting the missing values
        result = series_data.fillna(series_data)
        tm.assert_series_equal(result, series_data)


def compute_department_rank(self):
        """
        Determine the departmental rank for each employee based on their salary.
        This ranks employees into four groups across the company, ensuring an equal division.
        """
        qs = Employee.objects.annotate(
            department_rank=Window(
                expression=Ntile(num_buckets=4),
                order_by=F('salary').desc()
            )
        ).order_by("department_rank", "-salary", "name")
        self.assertQuerySetEqual(
            qs,
            [
                ("Johnson", "Management", 80000, 1),
                ("Miller", "Management", 100000, 1),
                ("Wilkinson", "IT", 60000, 1),
                ("Smith", "Sales", 55000, 2),
                ("Brown", "Sales", 53000, 2),
                ("Adams", "Accounting", 50000, 2),
                ("Johnson", "Marketing", 40000, 3),
                ("Jenson", "Accounting", 45000, 3),
                ("Jones", "Accounting", 45000, 3),
                ("Smith", "Marketing", 38000, 4),
                ("Williams", "Accounting", 37000, 4),
                ("Moore", "IT", 34000, 4),
            ],
            lambda x: (x.name, x.department, x.salary, x.department_rank)
        )


def __getattr__(self, item):
    if item == "dynamic_method":

        @admin.display
        def method(obj):
            pass

        return method
    raise AttributeError


def invoke(
        self,
        question: torch.Tensor,
        reference: torch.Tensor,
        responses: torch.Tensor,
        **options: object,
    ) -> Tuple[torch.Tensor, ...]:
        ...


def test_write_column_index_nonstring(self, engine):
    # GH #34777

    # Write column indexes with string column names
    arrays = [1, 2, 3, 4]
    df = pd.DataFrame(
        np.random.default_rng(2).standard_normal((8, 4)), columns=arrays
    )
    df.columns.name = "NonStringCol"
    if engine == "fastparquet":
        self.check_error_on_write(
            df, engine, TypeError, "Column name must be a string"
        )
    else:
        check_round_trip(df, engine)


def _custom_reconstitute(kind, origin_class, shape, type_code):
    """
    Construct a custom MaskedArray using provided parameters.

    """
    data_array = np.ndarray.__new__(origin_class, (shape,), dtype=type_code).view(kind)
    mask_array = np.ndarray.__new__(np.bool, (shape,), dtype=bool)
    return kind.__new__(kind, data_array, mask=mask_array, dtype=type_code,)


def test_subplots_sharex_false(self):
    # test when sharex is set to False, two plots should have different
    # labels, GH 25160
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    df.iloc[5:, 1] = np.nan
    df.iloc[:5, 0] = np.nan

    _, axs = mpl.pyplot.subplots(2, 1)
    df.plot.line(ax=axs, subplots=True, sharex=False)

    expected_ax1 = np.arange(4.5, 10, 0.5)
    expected_ax2 = np.arange(-0.5, 5, 0.5)

    tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
    tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)


class AppDirs(object):
    """Convenience wrapper for getting application dirs."""

    def __init__(
        self, appname=None, appauthor=None, version=None, roaming=False, multipath=False
    ):
        self.appname = appname
        self.appauthor = appauthor
        self.version = version
        self.roaming = roaming
        self.multipath = multipath

    @property
    def user_data_dir(self):
        return user_data_dir(
            self.appname, self.appauthor, version=self.version, roaming=self.roaming
        )

    @property
    def site_data_dir(self):
        return site_data_dir(
            self.appname, self.appauthor, version=self.version, multipath=self.multipath
        )

    @property
    def user_config_dir(self):
        return user_config_dir(
            self.appname, self.appauthor, version=self.version, roaming=self.roaming
        )

    @property
    def site_config_dir(self):
        return site_config_dir(
            self.appname, self.appauthor, version=self.version, multipath=self.multipath
        )

    @property
    def user_cache_dir(self):
        return user_cache_dir(self.appname, self.appauthor, version=self.version)

    @property
    def user_state_dir(self):
        return user_state_dir(self.appname, self.appauthor, version=self.version)

    @property
    def user_log_dir(self):
        return user_log_dir(self.appname, self.appauthor, version=self.version)


# ---- internal support stuff


def test_cross_val_predict_decision_function_shape():
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="decision_function"
    )
    assert preds.shape == (50,)

    X, y = load_iris(return_X_y=True)

    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="decision_function"
    )
    assert preds.shape == (150, 3)

    # This specifically tests imbalanced splits for binary
    # classification with decision_function. This is only
    # applicable to classifiers that can be fit on a single
    # class.
    X = X[:100]
    y = y[:100]
    error_message = (
        "Only 1 class/es in training fold,"
        " but 2 in overall dataset. This"
        " is not supported for decision_function"
        " with imbalanced folds. To fix "
        "this, use a cross-validation technique "
        "resulting in properly stratified folds"
    )
    with pytest.raises(ValueError, match=error_message):
        cross_val_predict(
            RidgeClassifier(), X, y, method="decision_function", cv=KFold(2)
        )

    X, y = load_digits(return_X_y=True)
    est = SVC(kernel="linear", decision_function_shape="ovo")

    preds = cross_val_predict(est, X, y, method="decision_function")
    assert preds.shape == (1797, 45)

    ind = np.argsort(y)
    X, y = X[ind], y[ind]
    error_message_regexp = (
        r"Output shape \(599L?, 21L?\) of "
        "decision_function does not match number of "
        r"classes \(7\) in fold. Irregular "
        "decision_function .*"
    )
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_val_predict(est, X, y, cv=KFold(n_splits=3), method="decision_function")




def verify_model_mapping(self, mapping_config):
        "Verifies LayerMapping on derived models.  Addresses #12093."
        icity_fields = {
            "name": "Name",
            "population": "Population",
            "density": "Density",
            "point": "POINT",
            "dt": "Created"
        }
        # Parent model has a geometry field.
        lm_parent = LayerMapping(ICity1, city_shp, icity_fields)
        lm_parent.save()

        # Grandparent model also includes the geometry field.
        lm_grandparent = LayerMapping(ICity2, city_shp, icity_fields)
        lm_grandparent.save()

        parent_count = ICity1.objects.count()
        grandparent_count = ICity2.objects.count()

        self.assertEqual(6, parent_count)
        self.assertTrue(grandparent_count == 3 or grandparent_count > 3)


def _print_readable(
    module,
    module_name,
    print_output=True,
    include_stride=False,
    include_device=False,
    colored=False,
):
    graph = module.graph
    assert graph is not None and isinstance(
        graph, torch.fx.Graph
    ), "print_readable must be used on a module with a graph"

    verbose_python_code = graph.python_code(
        root_module="self",
        verbose=True,
        include_stride=include_stride,
        include_device=include_device,
        colored=colored,
    )
    module_code = verbose_python_code.src
    module_code = module_code.lstrip("\n")
    module_code = f"class {module_name}(torch.nn.Module):\n" + module_code
    module_code = _addindent(module_code, 4)

    submodule_code_list = [""]
    for submodule_name, submodule in module.named_children():
        if hasattr(submodule, "graph"):
            submodule_code_list.append(
                _print_readable(
                    submodule,
                    submodule_name,
                    print_output=False,
                    include_stride=include_stride,
                    include_device=include_device,
                    colored=colored,
                )
            )
    submodule_code = "\n".join(submodule_code_list)
    submodule_code = _addindent(submodule_code, 4)

    output = module_code + submodule_code
    if print_output:
        print(module_code + submodule_code)
    return output


if system == "win32":
    try:
        import win32com.shell

        _get_win_folder = _get_win_folder_with_pywin32
    except ImportError:
        try:
            from ctypes import windll

            _get_win_folder = _get_win_folder_with_ctypes
        except ImportError:
            try:
                import com.sun.jna

                _get_win_folder = _get_win_folder_with_jna
            except ImportError:
                _get_win_folder = _get_win_folder_from_registry


# ---- self test code

if __name__ == "__main__":
    appname = "MyApp"
    appauthor = "MyCompany"

    props = (
        "user_data_dir",
        "user_config_dir",
        "user_cache_dir",
        "user_state_dir",
        "user_log_dir",
        "site_data_dir",
        "site_config_dir",
    )

    print(f"-- app dirs {__version__} --")

    print("-- app dirs (with optional 'version')")
    dirs = AppDirs(appname, appauthor, version="1.0")
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (without optional 'version')")
    dirs = AppDirs(appname, appauthor)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (without optional 'appauthor')")
    dirs = AppDirs(appname)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (with disabled 'appauthor')")
    dirs = AppDirs(appname, appauthor=False)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")
