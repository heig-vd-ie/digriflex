# README #

This code is written for the project DiGriFlex. More information regarding this project can be found 
[here](http://iese.heig-vd.ch/projets/digriflex).

The following papers present the modeling and the formulations of this project:
- [Rayati, Mohammad, Mokhtar Bozorg, Rachid Cherkaoui, and Mauro Carpita. "Distributionally robust chance constrained optimization for providing flexibility in an active distribution network." IEEE Transactions on Smart Grid 13, no. 4 (2022): 2920-2934.](https://ieeexplore.ieee.org/document/9721415).
- [Rayati, Mohammad, Mokhtar Bozorg, Mauro Carpita, Pasquale De Falco, Pierluigi Caramia, Antonio Bracale, Daniela Proto, and Fabio Mottola. "Real-Time Distribution Grid Control and Flexibility Provision under Uncertainties: Laboratory Demonstration." In 2022 IEEE 21st Mediterranean Electrotechnical Conference (MELECON), pp. 866-871. IEEE, 2022.](https://ieeexplore.ieee.org/abstract/document/9842979)
- [Rayati, Mohammad, Mokhtar Bozorg, Mauro Carpita, and Rachid Cherkaoui. "Stochastic optimization and Markov chain-based scenario generation for exploiting the underlying flexibilities of an active distribution network." Sustainable Energy, Grids and Networks (2023): 100999.](https://www.sciencedirect.com/science/article/pii/S2352467723000073)
- [Bozorg, Mokhtar, Antonio Bracale, Mauro Carpita, Pasquale De Falco, Fabio Mottola, and Daniela Proto. "Bayesian bootstrapping in real-time probabilistic photovoltaic power forecasting." Solar Energy 225 (2021): 577-590.](https://www.sciencedirect.com/science/article/pii/S0038092X21006393)
- [Bozorg, Mokhtar, Antonio Bracale, Pierluigi Caramia, Guido Carpinelli, Mauro Carpita, and Pasquale De Falco. "Bayesian bootstrap quantile regression for probabilistic photovoltaic power forecasting." Protection and Control of Modern Power Systems 5, no. 1 (2020): 1-12.](https://pcmp.springeropen.com/articles/10.1186/s41601-020-00167-7)
- [Bozorg, Mokhtar, Mauro Carpita, Pasquale De Falco, Davide Lauria, Fabio Mottola, and Daniela Proto. "A derivative-persistence method for real time photovoltaic power forecasting." In 2020 International Conference on Smart Grids and Energy Systems (SGES), pp. 843-847. IEEE, 2020.](https://ieeexplore.ieee.org/abstract/document/9364445)

### What is this repository for? ###
1. Forecasting loads and PV production in an active distribution grid 
2. Optimization of the resources withing the active distribution grid 
3. Running the optimization with different optimization algorithms 
4. Running the Optimization with the distribution-ally robust optimization algorithm 

### How do I get set up? ###
1. Install poetry
2. Install makefile
3. Install python3.9 of 64 bit 
4. Build a venv with python with the name ".venv":
    ```shell
    python -m venv .venv
    ```
5. Run the following command in terminal:
    ```shell
    poetry install
    ```
6. If you want to run the function with LabVIEW, you have to use python3.6 32bit version 
7. It needs gurobi installed with licence 
8. Run the following command in terminal:
    ```shell
    make -B
    ```

### Who do I talk to? ###
* Repo owner or admin
* Other community or team contact
