///////////////////// Spherical slab geometry ////////////////////
//// \author Roberto Piersanti <roberto.piersanti@polimi.it> /////
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//// This work is licensed under a Creative Commons       ////
//// Attribution-NonCommercial 4.0 International License. ////
//////////////////////////////////////////////////////////////

// Parameters [mm].

h = 8; // Mesh size

R0 = 20; // Inner radius
d  = 5; // Thickness
R1 = R0 + d; // Outer radius

Point(1) = {0, 0, 0, h};
Point(2) = {0, 0, R0, h};
Point(3) = {0, R0, 0, h};
Point(4) = {0, 0, -R0, h};

Point(5) = {0, 0, R1, h};
Point(6) = {0, R1, 0, h};
Point(7) = {0, 0, -R1, h};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Line(3) = {4, 7};
Circle(4) = {7, 1, 6};
Circle(5) = {6, 1, 5};
Line(6) = {5, 2};

Line Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{1}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{28}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{50}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} { Surface{72}; }

Surface Loop(1) = {16, 82, 38, 60, 63, 41, 19, 85};
Volume(newv) = {1};

Physical Volume("Myocardium", 1) = {1, 2, 3, 4};
Physical Surface("Endocardium", 20) = {16, 82, 38, 60, 63, 41, 19, 85};
Physical Surface("Epicardium", 10) = {67, 70, 92, 89, 23, 26, 45, 48};

Mesh 3;
