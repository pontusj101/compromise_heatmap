import os
import re
import sys
import cmd
import shlex
import platform
import subprocess
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import List, Dict
from docopt import docopt, DocoptExit
import readline

from .epic_importer import EpicImporter


class EpicShell(cmd.Cmd):
    prompt: str = 'epic> '
    _mods: Dict[str, ModuleType]
    _os_is_windows: bool
    _windows_bash_shell: str
    use_rawinput = True

    def __init__(self, plugin_dirs: List[str],
                 windows_bash_shell="C:\\Program Files\\Git\\bin\\bash.exe") -> None:
        super().__init__()
        readline.set_completer_delims(readline.get_completer_delims().replace('-', ''))

        self._mods = dict()
        self._os_is_windows = (platform.system() == "Windows")
        self._windows_bash_shell = windows_bash_shell

        self.plugin_dirs = plugin_dirs

        for path in self.plugin_dirs:
            sys.path.append(path)
            EpicImporter.load_paths.add(path)
        #self.load_plugins(plugin_dirs)

    def check_permissions(self, filepath):
        if self._os_is_windows:
            permissions = self.get_file_permissions(filepath)
            owner = self.get_file_owner(filepath)

            # Check if the third character in the permissions string is 'x', indicating execute permission for the owner
            if permissions[2] == 'x' and owner == os.getlogin():
                return True
            # Check if the ninth character in the permissions string is 'x', indicating execute permission for others
            elif permissions[9] == 'x':
                return True
            else:
                return False
        else:
            return os.access(filepath, os.X_OK)

    def get_file_permissions(self, filepath):
        try:
            # Run the `stat` command to get file permissions
            result = subprocess.run(['stat', '-c', '%A', filepath], capture_output=True, text=True, check=True)
            return result.stdout.strip()  # Strip any leading/trailing whitespace
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None

    def get_file_owner(self, filepath):
        try:
            # Run the `stat` command to get file own
            result = subprocess.run(['stat', '-c', '%U', filepath], capture_output=True, text=True, check=True)
            return result.stdout.strip()  # Strip any leading/trailing whitespace
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None

    def load_plugins(self, plugin_dirs):
        for path in plugin_dirs:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path) and os.access(item_path, os.X_OK):
                    self.load_plugin(item)


    def load_plugin(self, plugin_file: str):
        print(f"Loading {plugin_file}...")
        self._mods[plugin_file] = importlib.import_module(plugin_file)
        command_name = os.path.basename(plugin_file).replace('_', '-').split('.')[-1]
        setattr(self.__class__, 'do_' + command_name, lambda self, args: self.execute_plugin(self._mods[plugin_file], args))
        setattr(self.__class__, 'help_' + command_name, lambda self: print(self._mods[plugin_file].__doc__))
        setattr(self.__class__, 'complete_' + command_name, lambda self, text, line, begidx, endidx: self.complete_plugin(self._mods[plugin_file], text, line, begidx, endidx))

    def execute_plugin(self, mod: ModuleType, args):
        argv = shlex.split(args)
        try:
            mod.run(argv, name='__main__')
        except DocoptExit as de:
            print(de)
            return
        except Exception as e:
            print(f'Error when executing {mod.__spec__.name}: {type(e).__name__}: {e}')
            return

    def complete_plugin(self, mod: ModuleType, text, line, begidx, endidx) -> List[str] | None:
        # Split the line into words using shlex
        words = shlex.split(line)

        # Calculate the positions of each word in the original line
        word_positions = []
        current_position = 0

        for word in words:
            # Find the start position of the word in the original line
            start_idx = line.find(word, current_position)
            end_idx = start_idx + len(word)

            # Append the (start_idx, end_idx) tuple to word_positions
            word_positions.append((start_idx, end_idx))

            # Update current_position to be after this word
            current_position = end_idx

        line_cursor = endidx - begidx

        # Find the index in the words list of the current word
        index = None
        for i, wp in enumerate(word_positions):
            if wp[0] <= endidx <= wp[1]:
                index = i

        # If no index was found, this means the cursor is at the end of the current line.
        # Append an empty string to word list and set index accordingly.
        if index is None:
            words.append('')
            index = len(words) - 1

        return mod._completions(words, words[index], index, line_cursor)

    def do_add_load_path(self, args: str):
        argv = shlex.split(args)
        for path in argv:
            if path not in sys.path:
                sys.path.append(path)
            EpicImporter.load_paths.add(path)

    def do_load(self, args):
        if(args == "--all" or args == "-a"):
            self.load_plugins(self.plugin_dirs)
        else:
            try:
                arg = '.'.join(shlex.split(args))
                self.load_plugin(arg)
            except ModuleNotFoundError:
                print(f"Error: cmd {arg} not found.")
            except ValueError:
                print("Error: Cannot load Empty module name")

    def do_exit(self, args):
        """Exit the shell"""
        return True

    def do_help(self, line):
        try:
            cmd, *argv = shlex.split(line)
            mod = importlib.import_module(cmd)
            if len(argv) > 0:
                for i, arg in enumerate(argv):
                    if arg in mod.__spec__.loader_state["subcommands"]:
                        submod = importlib.import_module(f'{mod.__name__}.{arg}')
                        if len(argv[i+1:]) > 0:
                            self.do_help(argv[i+1:])
                        else:
                            print(submod.__doc__)
            else:
                print(mod.__doc__)
        except ModuleNotFoundError:
            super().do_help(line)

    def default(self, line: str):
        try:
            cmd, *argv = shlex.split(line)
            mod = importlib.import_module(cmd)
            self.execute_plugin(mod, line[len(cmd):])
        except DocoptExit as de:
            print(de)
            return
        except ModuleNotFoundError:
            print(f"Error: cmd {cmd} not found.")

    def completenames(self, text, *ignored):
        top_level_plugins = list()
        for path in self.plugin_dirs:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path) and os.access(item_path, os.X_OK):
                    top_level_plugins.append(item)

        return super().completenames(text, ignored) + top_level_plugins

    def completedefault(self, text, line, begidx, endidx) -> List[str]:
        try:
            origline = readline.get_line_buffer()
            line = origline.lstrip()
            stripped = len(origline) - len(line)
            begidx = readline.get_begidx() - stripped
            endidx = readline.get_endidx() - stripped
            cmd, *argv = shlex.split(line)
            mod = importlib.import_module(cmd)
            return self.complete_plugin(mod, text, line, begidx, endidx)
        except:
            return super().completedefault(text, line, begidx, endidx)

    def do_shell(self, line):
        try:
            # Use shlex.split to properly handle quoted strings
            parts = shlex.split(line)

            if not parts:
                return  # Handle empty input

            if parts[0] == "cd":
                # Join the rest of the parts to get the directory path
                if len(parts) > 1:
                    os.chdir(" ".join(parts[1:]))
                else:
                    print("cd: missing argument")
            else:
                # Use subprocess.run to execute other commands
                subprocess.run(parts, check=True)
        except ValueError as e:
            print(f'Error: {e}')
        except FileNotFoundError as e:
            print(f'Error: Command {line} not found')
        except OSError as e:
            print(f'Error: OSError when attempting to run {line}. \n{e}')
        except subprocess.CalledProcessError as e:
            print(f'Error: Command {line} failed with return code {e.returncode}')
