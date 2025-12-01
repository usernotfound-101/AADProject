def is_power_of_two(n):
    return (n & (n - 1)) == 0 and n > 0

number = 8830192
print(is_power_of_two(number))