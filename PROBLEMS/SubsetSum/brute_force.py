#THIS IS THE BRUTE FORCE APPROACH TO THE SUBSET SUM PROBLEM

def subset_sum_bruteforce(arr, target, i=0, current_sum=0):
    if current_sum == target:
        return True

    if i == len(arr):
        return False

    if subset_sum_bruteforce(arr, target, i + 1, current_sum + arr[i]):
        return True

    if subset_sum_bruteforce(arr, target, i + 1, current_sum):
        return True

    return False

def get_input():
    arr = [3, 34, 4, 12, 5, 2]
    target = 9
    return arr, target


arr, target = get_input()
print(subset_sum_bruteforce(arr, target))