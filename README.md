# Heat Exchanger Network Optimization
## Requirements
- Python 3.x
- pandas
- numpy
- scipy

## Problem
In the context of industrial processes, the optimization of heat exchange networks is crucial for enhancing energy efficiency, reducing operational costs, and minimizing environmental impact.
In this project, a heat exchanger network designed for a specific situation is considered. You can see the visual of this network below. For this HEN, we will try to find the minimum dT value that provides the most optimal cost.

![image](https://github.com/enes-yapici/hen_optimization/assets/125216116/8e256d03-9784-4614-867a-899742034120)

The input must necessarily include a value representing either the hot or cold utility, denoted as 'H/C'. It should be in the form of 'H{number}' for hot utility and 'C{number}' for cold utility. Values for n, Cp, and nCp must be entered. If there is a phase change, the energy required for this phase transition must be entered. If there is no phase change, it must be left blank. Tin and Tout values must be entered in Celsius degrees.

![image-2](https://github.com/enes-yapici/hen_optimization/assets/125216116/8b7792c7-2e1d-4591-8d21-321bd808592e)

## Output
The optimum dT_min value is : 4.0 degree Celsius.
The total annualized cost corresponding to that dT_min value is : 1.62 M$/year

![image](https://github.com/enes-yapici/hen_optimization/assets/125216116/7527e886-cbff-42c0-b883-8ec49613e2c9)

## Future Work
Initially, the check_phase_change function is quite basic. Although it currently only indicates that there is no problem, it should offer solutions for situations that may cause problems in later stages.
In addition, instead of us setting a heat exchanger network system, the program could be designed to suggest a network to us.
Additionally, it is possible to optimize several different variables simultaneously, such as dT_min and utility temperatures. This is done using the minimize function from scipy.optimize.

## For More Detail
Please read heat.ipynb

