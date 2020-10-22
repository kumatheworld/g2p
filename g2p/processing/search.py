class GreedySearch():
    def __init__(self, limit, st_idx=1, ed_idx=2):
        self.limit = limit
        self.st_idx = st_idx
        self.ed_idx = ed_idx

    def __call__(self, f, state_init):
        idx = self.st_idx
        state = state_init
        history = []
        for i in range(self.limit):
            prob, state = f(idx, state)
            idx = prob.argmax().item()
            if idx == self.ed_idx:
                break
            history.append(idx)
        return history
