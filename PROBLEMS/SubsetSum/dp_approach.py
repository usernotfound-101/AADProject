#THIS IS THE DP-APPROACH TO THE SUBSET SUM PROBLEM (Pseudo-Polynomial)

def subset_sum_DP(arr, target):
    dp = [False] * (target + 1)
    dp[0] = True
    for x in arr:
        for s in range(target, x - 1, -1):
            if dp[s - x]:
                dp[s] = True
    return dp[target]
