/////////////////////// Cubic slab geometry //////////////////////
//// \author Nicola De March, Martina Raffaelli, Kaixi Matteo Chen 


//////////////////////////////////////////////////////////////
//// This work is licensed under a Creative Commons       ////
//// Attribution-NonCommercial 4.0 International License. ////
//////////////////////////////////////////////////////////////

// Parameters [mm].

h = 0.2; // Mesh size
Lx = 10;
Ly =  1;
Lz =  1;

/////////////////////////////

Point(1) = {0, 0, 0, h};
Point(2) = {Lx, 0, 0, h};
Point(3) = {Lx, Ly, 0, h};
Point(4) = {0, Ly, 0, h};

Line(5) = {1, 2};
Line(6) = {2, 3};
Line(7) = {3, 4};
Line(8) = {4, 1};

Line Loop(9) = {5, 6, 7, 8};
Plane Surface(10) = 9;

Extrude {0,0,Lz} {Surface{10};}

Surface Loop(1) = {10, 19, 23, 27, 31, 32};

Physical Volume("Myocardium", 1) = {1};
Physical Surface("Base up", 50) = {32};
Physical Surface("Base down", 60) = {10};
Physical Surface("Endocardium", 20) = {27};
Physical Surface("Epicardium",  10) = {19};
Physical Surface("Front wall", 30) = {23};
Physical Surface("Back wall",  40) = {31};

Mesh 3;
