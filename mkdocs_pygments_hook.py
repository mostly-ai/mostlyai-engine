# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MkDocs hook: Pygments 2.20+ HtmlFormatter expects filename to be a string.

pymdown-extensions can pass filename=None when highlighting block code with no
title (e.g. mkdocstrings signatures), which breaks html.escape().
"""


def on_config(config, **kwargs):
    import pymdownx.highlight as ph

    if not getattr(ph, "pygments", False):
        return config

    _orig = ph.BlockHtmlFormatter.__init__

    def __init__(self, **options):
        if options.get("filename") is None:
            options = {**options, "filename": ""}
        _orig(self, **options)

    ph.BlockHtmlFormatter.__init__ = __init__  # type: ignore[method-assign]
    return config
