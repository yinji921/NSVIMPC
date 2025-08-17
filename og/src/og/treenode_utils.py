import equinox as eqx


def prettynode(cls):
    """Mixin for equinox pretty printing of nodes."""

    def __repr__(self):
        return eqx.tree_pformat(self)

    setattr(cls, "__repr__", __repr__)
    return cls
