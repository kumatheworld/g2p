def greedy_search(limit, st_dix=1, ed_idx=2):
    def greedy_search_(f, idx=st_dix, count=0, history=None):
        if history is None:
            history = []
        if count >= limit:
            return history
        likelihoods = f(idx)
        idx = likelihoods.argmax().item()
        if idx == ed_idx:
            return history
        history.append(idx)
        return greedy_search_(f, idx, count + 1, history)

    return greedy_search_
