height = 100 * 3 / 5
number_of_bounces = 1
while number_of_bounces < 11:
    print(number_of_bounces, round(height, ndigits = 4))
    height = 3 / 5 * height
    round(height, ndigits = 4)
    number_of_bounces = number_of_bounces + 1

