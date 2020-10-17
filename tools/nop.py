class Nop:
    def __getattr__(self, _):
        return self.nop

    def nop(*args, **kwargs):
        pass
