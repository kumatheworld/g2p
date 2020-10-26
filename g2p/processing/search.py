from collections import namedtuple
from heapq import nlargest

Track = namedtuple('Track', 'len idx state prob_sum hist')

class GreedySearch:
    def __init__(self, limit, st_idx=1, ed_idx=2):
        self.limit = limit
        self.st_idx = st_idx
        self.ed_idx = ed_idx

    def __call__(self, f, state_init):
        idx = self.st_idx
        state = state_init
        history = []
        for _ in range(self.limit):
            prob, state = f(idx, state)
            idx = prob.argmax().item()
            if idx == self.ed_idx:
                break
            history.append(idx)
        return history

class BeamSearchLike:
    def __init__(self, beam_size, limit, st_idx=1, ed_idx=2):
        self.beam_size = beam_size
        self.limit = limit
        self.st_idx = st_idx
        self.ed_idx = ed_idx

    def _select_topk(self, tracks):
        return nlargest(self.beam_size, tracks,
                        lambda tr: tr.prob_sum / tr.len)

    def __call__(self, f, state_init):
        tracks = [Track(0, self.st_idx, state_init, 0, [])]
        tracks_done = []
        for _ in range(self.limit):
            tracks_new = []
            for track in tracks:
                prob, state = f(track.idx, track.state)
                for b in zip(*prob.topk(self.beam_size)):
                    idx = b[1].item()
                    hist = track.hist.copy()
                    if idx != self.ed_idx:
                        hist.append(idx)
                    tracks_new.append(Track(track.len + 1, idx, state,
                                            track.prob_sum + b[0].item(),
                                            hist))
            tracks = []
            for track in self._select_topk(tracks_new):
                (tracks_done if track.idx == self.ed_idx else tracks).append(track)
            if len(tracks_done) >= self.beam_size:
                break
        else:
            tracks_done.extend(tracks)
        tracks = self._select_topk(tracks_done)
        return tracks[0].hist
