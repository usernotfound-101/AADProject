#THIS IS THE Fast Polynomial Time Approximation Scheme TO THE SUBSET SUM PROBLEM

def subset_sum_FPTAS(arr, target, eps=0.01):
    L = [0]
    for x in arr:
        L = sorted(set(L + [s + x for s in L]))
        new_L = [L[0]]
        for s in L[1:]:
            if s > new_L[-1] * (1 + eps):
                new_L.append(s)

        L = [s for s in new_L if s <= target]

    best = max(L)
    return best == target, best
