"""
%┏━━━┓┏━━━┓┏━┓━┏┓┏━━━┓┏┓━━━┏━━━━┓┏┓━━┏┓━━━━━┏━━━┓┏━━━┓┏━━━┓┏━━━━┓┏━━━┓┏━━━┓
?┃┏━┓┃┃┏━━┛┃┃┗┓┃┃┃┏━┓┃┃┃━━━┃┏┓┏┓┃┃┗┓┏┛┃━━━━━┃┏━━┛┃┏━┓┃┃┏━┓┃┃┏┓┏┓┃┃┏━┓┃┃┏━┓┃
%┃┗━┛┃┃┗━━┓┃┏┓┗┛┃┃┃━┃┃┃┃━━━┗┛┃┃┗┛┗┓┗┛┏┛━━━━━┃┗━━┓┃┃━┃┃┃┃━┗┛┗┛┃┃┗┛┃┃━┃┃┃┗━┛┃
?┃┏━━┛┃┏━━┛┃┃┗┓┃┃┃┗━┛┃┃┃━┏┓━━┃┃━━━┗┓┏┛━━━━━━┃┏━━┛┃┗━┛┃┃┃━┏┓━━┃┃━━┃┃━┃┃┃┏┓┏┛
%┃┃━━━┃┗━━┓┃┃━┃┃┃┃┏━┓┃┃┗━┛┃━┏┛┗┓━━━┃┃━━━━━━┏┛┗┓━━┃┏━┓┃┃┗━┛┃━┏┛┗┓━┃┗━┛┃┃┃┃┗┓
?┗┛━━━┗━━━┛┗┛━┗━┛┗┛━┗┛┗━━━┛━┗━━┛━━━┗┛━━━━━━┗━━┛━━┗┛━┗┛┗━━━┛━┗━━┛━┗━━━┛┗┛┗━┛
%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
?━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script calculates the penalty factors for a given base value and growth rate.
It prints the base value and its square for each iteration, showing how the penalty factor grows over time.

You can use this script to see how different penalty factors evolve for determining the penalty in your network model.

For all of our experiments, we used a base value of 500 and a growth rate of 1.15. With the penalty factor squared within the fitness function.

Example usage:
    python check_penalty_factors.py
This will print the penalty factors to the console.

Make sure to adjust the `cover_penalty_base`, `growth_rate`, and `iterations` variables as needed for your specific use case.
"""

# Set the base value, growth rate, and number of iterations for the penalty factor calculation
cover_penalty_base = 500
growth_rate = 1.15
iterations = 25

# Set the header and table format for the output
print("Penalty Factor Calculation\n")
print("-" * 42)
print(f"{'Iter':>5} | {'Base':>15} | {'Squared':>15}")
print("-" * 42)

# Calculate and print the penalty factors for each iteration
base_value = cover_penalty_base
for i in range(iterations + 1):
    no_square = base_value
    squared = base_value ** 2
    print(f"{i:5d} | {base_value:15.2f} | {squared:15.2f}")
    base_value *= growth_rate