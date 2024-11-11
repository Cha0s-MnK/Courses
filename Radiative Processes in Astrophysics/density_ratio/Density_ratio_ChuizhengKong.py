"""
Function: Calculate the ratio of electron volume density to hydrogen volume density in a fully ionized plasma
          with solar abundance.
Usage:    python3.11 Density_ratio_ChuizhengKong.py
Version:  Last edited by Cha0s_MnK on 2024-11-10 (UTC+08:00).
"""

###################
# SET ARGUMENT(S) #
###################

# element abundances in the present-day solar photosphere
# reference: Asplund et al. 2009 (https://arxiv.org/pdf/0909.0948)
data_element = [
    # [element, atomic number Z, log_varepsilon_X]
    ['H', 1, 12.00],
    ['He', 2, 10.93],
    ['Li', 3, 1.05],
    ['Be', 4, 1.38],
    ['B', 5, 2.70],
    ['C', 6, 8.43],
    ['N', 7, 7.83],
    ['O', 8, 8.69],
    ['F', 9, 4.56],
    ['Ne', 10, 7.93],
    ['Na', 11, 6.24],
    ['Mg', 12, 7.60],
    ['Al', 13, 6.45],
    ['Si', 14, 7.51],
    ['P', 15, 5.41],
    ['S', 16, 7.12],
    ['Cl', 17, 5.50],
    ['Ar', 18, 6.40],
    ['K', 19, 5.03],
    ['Ca', 20, 6.34],
    ['Sc', 21, 3.15],
    ['Ti', 22, 4.95],
    ['V', 23, 3.93],
    ['Cr', 24, 5.64],
    ['Mn', 25, 5.43],
    ['Fe', 26, 7.50],
    ['Co', 27, 4.99],
    ['Ni', 28, 6.22],
    ['Cu', 29, 4.19],
    ['Zn', 30, 4.56],
    ['Ga', 31, 3.04],
    ['Ge', 32, 3.65]]

#################
# MAIN FUNCTION #
#################

def main():
    ratio = 0.0

    # loop through elements and calculate contributions
    for element, Z, lg_varepsilon in data_element:
        num_e_per_H  = Z * 10**(lg_varepsilon - 12)
        ratio       += num_e_per_H

        print(f"{element} (Z = {Z}):")
        print(f"    log_varepsilon = {lg_varepsilon}")
        print(f"    number of electrons per H atom = {num_e_per_H:.3e}")
        print()

    print(f"ratio of electron density to hydrogen density n_e / n_H) = {ratio:.5f}")

if __name__ == "__main__":
    main()