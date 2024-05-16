//////////// Idealized left atrium: Hollow sphere geo ////////////
//// \author Roberto Piersanti <roberto.piersanti@polimi.it> /////
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//// This work is licensed under a Creative Commons       ////
//// Attribution-NonCommercial 4.0 International License. ////
//////////////////////////////////////////////////////////////

SetFactory("OpenCASCADE");

// Parameters [mm].

R_int = 28.1; // Internal radius of the hollow sphere
s =  2.5; // Width of the atrial wall
R = R_int + s; // External radius of the hollow sphere
dvp = 13; // Diameter of the pulmonary veins
phi = Pi / 6; // Rotation angle of the cones around y-axis
theta = Pi / 6; // Rotation angle of the cones around z-axis

Sphere(1) = {0, 0, 0, R, -Pi / 2, Pi / 2, 2 * Pi};
Cone(2) = {0, 0, 0,  R, 0, 0, 0, dvp / 2, 2 * Pi};
Cone(3) = {0, 0, 0,  R, 0, 0, 0, dvp / 2, 2 * Pi};
Cone(4) = {0, 0, 0,  R, 0, 0, 0, dvp / 2, 2 * Pi};
Cone(5) = {0, 0, 0,  R, 0, 0, 0, dvp / 2, 2 * Pi};

Rotate {{0, 1, 0}, {0, 0, 0}, -phi} {Volume{2};}
Rotate {{0, 0, 1}, {0, 0, 0}, theta} {Volume{2};}
Rotate {{0, 1, 0}, {0, 0, 0}, -phi} {Volume{3};}
Rotate {{0, 0, 1}, {0, 0, 0}, -theta} {Volume{3};}
Rotate {{0, 1, 0}, {0, 0, 0}, -(Pi - phi)} {Volume{4};}
Rotate {{0, 0, 1}, {0, 0, 0}, theta} {Volume{4};}
Rotate {{0, 1, 0}, {0, 0, 0}, -(Pi - phi)} {Volume{5};}
Rotate {{0, 0, 1}, {0, 0, 0}, -theta} {Volume{5};}

Sphere(6) = {0, 0, 0, R_int, -Pi / 2, Pi / 2, 2 * Pi};

BooleanDifference{ Volume{1}; Delete; }{ Volume{6}; Delete; }

Cone(7) = {0, 0, -R, 0, 0, R, R, 0, Pi / 2};
Rotate {{0, 0, 1}, {0, 0, -R}, Pi / 2} {Duplicata{Volume{7};}}
Rotate {{0, 0, 1}, {0, 0, -R}, Pi / 2} {Duplicata{Volume{8};}}
Rotate {{0, 0, 1}, {0, 0, -R}, Pi / 2} {Duplicata{Volume{9};}}

BooleanDifference {Volume{1}; Delete;}{Volume{3}; Volume{2}; Volume{4}; Volume{5}; Volume{8}; Volume{9}; Volume{10}; Volume{7}; Delete;}

Physical Volume("Miocardium", 1) = {1};
Physical Surface("Endocardium", 10) = {10};
Physical Surface("Epicardium", 30) = {1};
Physical Surface("MitralValveOpening", 40) = {2, 5, 4, 3};
Physical Surface("PulmonaryVeins_R", 20) = {6, 9};
Physical Surface("PulmonaryVeins_L", 50) = {7, 8};

Mesh 3;
