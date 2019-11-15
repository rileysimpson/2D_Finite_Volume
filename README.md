# 2D Unstructured Finite Volume Solver

Solves the 2D Laplace equation d^2(phi)/dx^2 + d^2(phi)/dy^2 = 0 with dirichlet boundary conditions

- Finite volume formulation
- 2nd order accurate centered interpolation of interior nodes
- 1st order accurate interpolation of bounding nodes
- Solved via Gauss-Seidel iteration

Input:
- Fluent Mesh File (ascii)
- User settings

Output:
- 2D mesh plot
- Convergence plot
- Solution contour plot
- Solution 3D plots (cell centroids and vertices)
- Residuals 3D plot (cell centroids)
