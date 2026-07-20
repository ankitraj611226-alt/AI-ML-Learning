"""
=========================================================
            Company Sales Analysis
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Library     : NumPy

Description:
This project analyzes a company's monthly sales using
NumPy. It calculates total sales, average sales,
highest & lowest sales, best and worst performing
months, and identifies months with above and below
average sales.
=========================================================
"""

import numpy as np


def main():

    months = np.array([
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])

    sales = []

    print("=" * 45)
    print("        COMPANY SALES ANALYSIS")
    print("=" * 45)
    print("Enter monthly sales (in $1000)\n")

    for month in months:
        value = float(input(f"{month}: "))
        sales.append(value)

    sales = np.array(sales)

    print("\n" + "=" * 45)
    print("           SALES REPORT")
    print("=" * 45)

    print(f"Total Sales          : ${np.sum(sales):.2f}k")
    print(f"Average Sales        : ${np.mean(sales):.2f}k")
    print(f"Highest Sales        : ${np.max(sales):.2f}k")
    print(f"Lowest Sales         : ${np.min(sales):.2f}k")

    best_month = months[np.argmax(sales)]
    worst_month = months[np.argmin(sales)]

    print(f"Best Performing Month: {best_month}")
    print(f"Worst Performing Month: {worst_month}")

    above_average = months[sales > np.mean(sales)]
    below_average = months[sales < np.mean(sales)]

    print("\nMonths Above Average Sales:")
    print(", ".join(above_average))

    print("\nMonths Below Average Sales:")
    print(", ".join(below_average))

    print("\nAnalysis Completed Successfully.")


if __name__ == "__main__":
    main()