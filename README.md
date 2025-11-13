Project 1 — MACD Indicator  
Goal:  
Implement the MACD (Moving Average Convergence/Divergence) indicator from scratch using only NumPy for numerical operations.  

Key tasks:  
    -Load ~1000 samples of financial closing-price data (CSV).  
    -Implement EMA, MACD, and SIGNAL manually (no TA libraries).  
    -Detect buy and sell signals using MACD–SIGNAL crossings.  
    -Create plots showing:  
        ->Price history  
        ->MACD + SIGNAL + buy/sell markers  
        ->Example transactions  
    -Build an automatic trading simulation starting with 1000 units.  
    -Summarize profitability, number of winning/losing trades, and indicator usefulness.  


Project 2 — Solving Linear Systems (Jacobi, Gauss–Seidel, LU)  
Goal:  
Analyse and compare three methods for solving large linear systems: Jacobi, Gauss–Seidel, and LU factorization.  

Key tasks:  
    -Construct a band matrix A of size depending on student index (5 diagonals).  
    -Generate vector b using sine-based formula.  
    -Implement the Jacobi and Gauss–Seidel iterative methods.  
    -For each method:  
        ->Measure number of iterations to reach residual   
        ->Plot residual norm per iteration (log scale)  
    -Test convergence for different matrix parameters.  
    -Implement LU decomposition and compute solution for the non-convergent case.  
    -Compare performance of all methods for varying matrix sizes.  
    -Provide conclusions on accuracy, speed, and convergence behavior.  


Project 3 — Elevation Profile Interpolation  
Goal:  
Approximate real-world elevation data using interpolation methods.  

Key tasks:  
    -Collect elevation samples from various sources (e.g., Google Maps, GeoContext).  
    -Choose several routes with different characteristics (flat, hilly, many climbs).  
    -Apply two approximation methods:  
        ->Lagrange polynomial interpolation (scaled to [-1, 1] for stability)  
        ->Cubic spline interpolation  
    -For each method:  
        ->Analyse impact of number and placement of nodes  
        ->Plot original data, nodes, and interpolation curves  
    -Compare sensitivity to noise, measurement precision, and route profiles.  
    -Summarize which method is more suitable for elevation modeling and why.  
