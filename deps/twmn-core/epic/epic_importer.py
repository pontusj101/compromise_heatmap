import os
import re
import ast
import sys
import platform
import functools
import subprocess
from types import ModuleType
from typing import Any

import docopt
import importlib
from importlib.machinery import ModuleSpec
from importlib.abc import PathEntryFinder, Loader
from jinja2 import Template, StrictUndefined
from pathlib import Path

from .dotdict import DotDict

# class EpicList(list):
    # def add(self, item: Any)-> None:
        # super().add(item)
        # sys.path.append(item)

class EpicImporter(Loader, PathEntryFinder):
    bash_libs = list()
    help_tokens = dict()
    load_paths = set()

    @classmethod
    def epic_importer_factory(cls, path: str):
        for load_path in cls.load_paths:
            if Path(path).is_relative_to(load_path):
                return cls(path)
        raise ImportError

    @staticmethod
    def _resolve_file_interpreter(path):
        if not path.is_file():
            return None

        if not os.access(str(path), os.X_OK):
            return None

        with open(path) as f:
            first_line = f.readline()

        if first_line[0:2] != "#!":
            return None

        return first_line.split("/")[-1].split(" ")[-1].strip()

    def __init__(self, path):
        self.path = Path(path)

    def find_spec(self, fullname: str, path: str, target=None):
        """
        Return the module spec (i.e. import information) for a module, or `None` if the
        module can't be handled correctly.

        See more at https://docs.python.org/3/reference/import.html#the-meta-path.

        :param name: the fully qualified name of the module, e.g. `a.b.c`.
        :param path: akin to sys.path but for imports. Will be `None` for top-level
                     modules and `<parent>.__path__` for subpackages.
        :param target: an existing module object to be loaded later, only for reloading.
        """
        unqualified_name = fullname.split(".")[-1]

        if path:
            script_path = ".d".join([path, unqualified_name])
        else:
            script_path = self.path / unqualified_name

        if not (interpreter := self._resolve_file_interpreter(script_path)):
            return None

        with open(script_path) as f:
            script_contents = f.read()
            docopt_spec = self._get_docopt_string(script_contents, interpreter)

            if not docopt_spec:
                return None

        subcmds = list()
        if os.path.isdir(str(script_path)+'.d'):
            for item in os.listdir(str(script_path)+'.d'):
                item_path = os.path.join(str(script_path)+'.d', item)
                if os.path.isfile(item_path) and os.access(item_path, os.X_OK):
                    subcmds.append(item)

        m = ModuleSpec(
            fullname,
            self,
            origin=str(script_path),
            is_package=True,
            loader_state={
                "interpreter": interpreter,
                "contents": script_contents,
                "docopt": docopt_spec,
                "subcommands": subcmds
            },
        )
        m.has_location = True
        m.submodule_search_locations = [str(script_path) + ".d"]
        return m

    def create_module(self, spec: ModuleSpec):
        """
        Returning None uses the standard machinery for creating modules, but we want
        to include the documentation for docopt parsing.

        Any import-related module attributes (e.g. __spec__) are automatically set.
        """
        return ModuleType(spec.name, doc=spec.loader_state["docopt"])

    def exec_module(self, module: ModuleType) -> None:
        module._completions = functools.partial(EpicImporter._completions, module)
        module.run = functools.partial(self._run, module)


    def iter_modules(self, prefix=""):
        """
        Yield module names discovered under the path the importer manages.

        Looking at the source code for pkg_util.iter_modules(), available at:

        https://github.com/python/cpython/blob/3.11/Lib/pkgutil.py#L228

        it seems that pkgutils.iter_modules() expects the iter_modules() method
        of a finder to accept only a prefix and that the finder is instantiated
        with a path under which to look for modules. I.e. the expectation is
        that the finder is used via the sys.path_hooks mechanisms and not via
        sys.meta_path. I.e. pkgutils.iter_modules seems to work only with
        ancestors of PathEntryFinder.

        :param prefix: a string to output on the front of every module name on
        output. This is passed directly from the argument with the same name in
        pkgutils.iter_modules.
        """
        for item in self.path.iterdir():
            if self._resolve_file_interpreter(item):
                yield prefix + item.stem, False


    @staticmethod
    def parse_docopt_string(doc):
        options = docopt.parse_defaults(doc)
        pattern = docopt.parse_pattern(docopt.formal_usage(docopt.printable_usage(doc)), options)
        pattern_options = set(pattern.flat(docopt.Option))
        for ao in pattern.flat(docopt.AnyOptions):
            doc_options = docopt.parse_defaults(doc)
            ao.children = list(set(doc_options) - pattern_options)
        return [a.name for a in pattern.flat() if not a.name.startswith('<')]

    @staticmethod
    def _completions(module, words, cword, index, cursor):
        '''
        words: the current line as a list of words
        cword: the current word being completed
        index: the index of cword in the words list
        cursor: the cursor index in the cword
        '''
        if words and words[1] in module.__spec__.loader_state["subcommands"]:
            submod = importlib.import_module(f'{module.__name__}.{words[1]}')
            return submod._completions(words[1:], cword, index, cursor)

        completions = EpicImporter.parse_docopt_string(module.__spec__.loader_state["docopt"])
        completions += module.__spec__.loader_state["subcommands"]
        return completions

    @staticmethod
    def _run(module, argv: list, *, name: bool = True):
        if argv and argv[0] in module.__spec__.loader_state["subcommands"]:
            submod = importlib.import_module(f'{module.__name__}.{argv[0]}')
            submod.run(argv[1:])
            return

        kwargs = docopt.docopt(module.__spec__.loader_state["docopt"], argv, help=True)
        kwargs = {k.lstrip("-").strip("<>"): v for k, v in kwargs.items()}
        kwargs = {k.replace('-', '_'): v for k, v in kwargs.items()}
        kwargs = DotDict(kwargs)

        interpreter = module.__spec__.loader_state["interpreter"]

        if re.match('python', interpreter):
            # Compile the code before execution for performance and better
            # stacktraces.
            compiled_code = compile(
                source=Path(module.__file__).read_text(),
                filename=module.__file__,
                mode='exec',
            )

            module_name = module.__name__

            if name:
                module.__name__ = name
            prev_argv = sys.argv
            sys.argv = [module_name] + argv

            epic_vars = {
               "__epic_script__": Path(module.__file__).name,
            }

            exec(compiled_code, vars(module) | {'kwargs': kwargs} | epic_vars)

            if name:
                module.__name__ = module_name
            sys.argv = prev_argv

        elif interpreter == "bash":
            for k, v in kwargs.items():
                if isinstance(v, bool):
                    kwargs[k] = str(v).lower()
            kwargs["BASH_ENV"] = os.path.abspath("bash_env")
            if platform.system() == "Windows":
                shell = "C:\\Program Files\\Git\\bin\\bash.exe"
            else:
                shell = "/bin/bash"
            subprocess.run([shell, module.__file__], env=kwargs)
        else:
            raise ImportError("unsupported interpreter")


    def _get_docopt_string(self, script_contents: str, interpreter: str):
        '''
        If python, attempt to find the docstring in the typical __doc__
        style string.

        Otherwise, parse the script line by line reading the header
        comment section.
        '''
        if re.match('python', interpreter):
            match = re.search(r'"""(.*?)"""', script_contents, re.DOTALL)
            if match:
                docopt_spec = match.group(1)
            else:
                return None
        else:
            script_lines = script_contents.split("\n")
            for line in script_lines[1:]:
                if line.find("Usage:") > 1:
                    comment_char = line[0]
                    break
            else:
                return None

            line_iter = iter(script_lines[1:])

            comment = re.compile(f"{comment_char}")

            docopt_spec = []
            try:
                while comment.match(line := next(line_iter)):
                    docopt_spec.append(line[2:])
            except StopIteration:
                pass

            if not docopt_spec:
                return None

            docopt_spec = "\n".join(docopt_spec).lstrip()

        docopt_spec = Template(docopt_spec, undefined=StrictUndefined).render(
            **self.help_tokens
        )

        return docopt_spec

sys.path_hooks.insert(0, EpicImporter.epic_importer_factory)
