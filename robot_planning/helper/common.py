def print_error(*message):
    print("\033[91m", "ERROR ", *message, "\033[0m")
    raise RuntimeError


def print_ok(*message):
    print("\033[92m", *message, "\033[0m")


def print_warning(*message):
    # yellow
    # print('\033[93m', *message, '\033[0m')
    # red
    print("\033[91m", "WARNING: ", *message, "\033[0m")


def print_info(*message):
    print("\033[96m", *message, "\033[0m")


class PrintBase:
    def prefix(self):
        return "[" + self.__class__.__name__ + "]: "

    @classmethod
    def print_error(cls, *message):
        print("\033[91m", "[" + cls.__name__ + "]: ", "ERROR ", *message, "\033[0m")
        raise RuntimeError

    @classmethod
    def print_ok(cls, message):
        print("\033[92m", "[" + cls.__name__ + "]: ", *message, "\033[0m")

    @classmethod
    def print_warning(cls, message):
        # yellow
        # print('\033[93m',"["+cls.__name__+"]: ", *message, '\033[0m')
        # red
        print("\033[91m", "[" + cls.__name__ + "]: ", "WARNING: ", *message, "\033[0m")

    @classmethod
    def print_info(cls, message):
        print("\033[96m", "[" + cls.__name__ + "]: ", *message, "\033[0m")


class PrintObject:
    def __init__(self):
        # print_ok(self.prefix() + "in use")
        pass

    def prefix(self):
        return "[" + self.__class__.__name__ + "]: "

    def print_error(self, *message):
        print("\033[91m", self.prefix(), "ERROR ", *message, "\033[0m")
        raise RuntimeError

    def print_ok(self, *message):
        print("\033[92m", self.prefix(), *message, "\033[0m")

    def print_warning(self, *message):
        # yellow
        # print('\033[93m',self.prefix(), *message, '\033[0m')
        # red
        print("\033[91m", self.prefix(), "WARNING: ", *message, "\033[0m")

    def print_info(self, *message):
        print("\033[96m", self.prefix(), *message, "\033[0m")
