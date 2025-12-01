#THIS IS THE GREEDY FORCE APPROACH TO THE SUBSET SUM PROBLEM

def subset_sum_greedy(arr, target):
    arr = sorted(arr, reverse=True)
    total = 0
    subset = []

    for x in arr:
        if total + x <= target:
            total += x
            subset.append(x)
        if total == target:
            return True, subset

    return False, subset
