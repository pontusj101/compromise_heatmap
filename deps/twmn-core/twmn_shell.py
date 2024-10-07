from typing import List
from epic.epic_shell import EpicShell

class TwmnShell(EpicShell):
    def __init__(self, plugin_dirs: List[str]) -> None:
        super().__init__(plugin_dirs)
        self.prompt = 'twmn> '

if __name__ == '__main__':
    shell = TwmnShell(plugin_dirs=['libexec'])
    shell.cmdloop()
