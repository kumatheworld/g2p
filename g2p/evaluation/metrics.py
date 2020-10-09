import numpy as np

def levenshtein_distance(a, b, divide_by_len_b=True):
    table_width = len(b) + 1
    table_prev = np.arange(table_width)
    table_curr = np.empty_like(table_prev)
    for i, x in enumerate(a):
        table_curr[0] = i
        for j, y in enumerate(b):
            table_curr[j + 1] = min(table_prev[j] + (x != y),
                                    table_prev[j + 1] + 1,
                                    table_curr[j] + 1)
        table_prev, table_curr = table_curr, table_prev

    distance = table_prev[-1]
    if divide_by_len_b:
        distance /= len(b)
    return distance
