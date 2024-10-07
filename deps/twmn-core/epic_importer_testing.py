from epic import EpicImporter
import importlib
import sys

sys.path.append("libexec")

EpicImporter.load_paths.add("libexec")
mod1 = importlib.import_module("plugin1")
#mod3 = importlib.import_module("plugin1.plugin3")
mod1.run(['command1'])
#mod3.run(['--lol'])

#mod2 = importlib.import_module("plugin2")
#mod2.run(['--ll'])

