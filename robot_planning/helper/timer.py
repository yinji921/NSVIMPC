from decimal import Decimal
from math import ceil, log10
from timeit import default_timer

_active_timer = None


class DummyTimer:
    def __init__(self, name: str, print_results=True, parent=None):
        pass

    @staticmethod
    def get_active() -> "DummyTimer":
        return DummyTimer("")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def child(self, name):
        return self

    def start(self):
        return self

    def stop(self):
        return self

    def print_results(self):
        pass

    def print_control_freq_result(self):
        pass

    def _format_results(self, indent="  "):
        return ""


class Timer:
    def __init__(self, name: str, print_results=True, parent=None):
        self.elapsed = Decimal()
        self._name = name
        self._print_results = print_results
        self._start_time = None
        self._children = {}
        self._parent = parent
        self._count = 0

        global _active_timer
        if _active_timer is None:
            _active_timer = self

    @staticmethod
    def get_active() -> "Timer":
        if _active_timer is not None:
            return _active_timer

        return Timer("")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        if self._print_results:
            self.print_results()

    def child(self, name):
        try:
            result = self._children[name]
            result._parent = self
            return result
        except KeyError:
            result = Timer(name, print_results=False, parent=self)
            self._children[name] = result
            return result

    def start(self):
        self._count += 1
        self._start_time = self._get_time()
        # If we aren't a child of the active timer, then add us to the active timer's children.
        global _active_timer
        if _active_timer is not self:
            _active_timer._children[self._name] = self
        self._parent = _active_timer
        _active_timer = self
        return self

    def stop(self):
        self.elapsed += self._get_time() - self._start_time
        # If we are the active timer, then set the active timer to our parent.
        global _active_timer
        if _active_timer is self:
            _active_timer = self._parent
        self._parent = None

        return self

    def print_results(self):
        print(self._format_results())
        # pass

    def print_control_freq_result(self):
        print("control frequency: ", 1 / self.elapsed)

    def _format_results(self, indent="  "):
        children = self._children.values()
        elapsed = self.elapsed or sum(c.elapsed for c in children)
        result = "%s: %.3fs" % (self._name, elapsed)
        max_count = max(c._count for c in children) if children else 0
        count_digits = 0 if max_count <= 1 else int(ceil(log10(max_count + 1)))
        for child in sorted(children, key=lambda c: c.elapsed, reverse=True):
            lines = child._format_results(indent).split("\n")
            child_percent = child.elapsed / elapsed * 100
            lines[0] += " (%d%%)" % child_percent
            if count_digits:
                # `+2` for the 'x' and the space ' ' after it:
                lines[0] = ("%dx " % child._count).rjust(count_digits + 2) + lines[0]
            for line in lines:
                result += "\n" + indent + line
        return result

    def _get_time(self):
        return Decimal(default_timer())


Timer = DummyTimer
