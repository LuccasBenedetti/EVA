"""
EVA 4.0 — Logger Colorido
"""
from colorama import init, Fore, Style

init(autoreset=True)


class Logger:
    """Logger colorido por categoria."""

    @staticmethod
    def sys(msg):
        print(f"{Fore.CYAN}[SISTEMA]{Style.RESET_ALL} {msg}")

    @staticmethod
    def ia(msg):
        print(f"{Fore.MAGENTA}[EVA]{Style.RESET_ALL} {msg}")

    @staticmethod
    def mem(msg):
        print(f"{Fore.GREEN}[MEMÓRIA]{Style.RESET_ALL} {msg}")

    @staticmethod
    def vis(msg):
        print(f"{Fore.YELLOW}[VISÃO]{Style.RESET_ALL} {msg}")

    @staticmethod
    def err(msg):
        print(f"{Fore.RED}[ERRO]{Style.RESET_ALL} {msg}")

    @staticmethod
    def ptt(msg):
        print(f"{Fore.BLUE}[MIC]{Style.RESET_ALL} {msg}")

    @staticmethod
    def tool(msg):
        print(f"{Fore.WHITE}[TOOL]{Style.RESET_ALL} {msg}")
