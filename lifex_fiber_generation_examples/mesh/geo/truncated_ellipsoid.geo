///////////////// Truncated ellipsoid geometry ///////////////////
//// \author Roberto Piersanti <roberto.piersanti@polimi.it> /////
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//// This work is licensed under a Creative Commons       ////
//// Attribution-NonCommercial 4.0 International License. ////
//////////////////////////////////////////////////////////////

// Parameters.

n = 2; // numbers of refinements

N  = 13 * n; // numbers of layers along the extrusion direction
N1 =  5 * n; // numbers of layers along the transmural direction
N2 = 20 * n; // numbers of layers along the ellipse length direction

d_focal = 45.0;
nu_epi   = 0.8;
nu_endo  = 0.6;
mu_base_gradi = 110.0;
mu_base_gradi_endo = 40;

//////////////////////////////////////////////////////////
// Function for point in Prolate Spheroidal Coordinates //
//////////////////////////////////////////////////////////

Function EllipsoidPoint
        Point(id) = {r1 * Sin(mu) * Cos(theta),
                     r1 * Sin(mu) * Sin(theta),
                     r2 * Cos(mu)};
Return

///////////////////////////////////////////////////////////

// Center, point (tag 1)
center = newp;
Point(center) = {0.0, 0.0, 0.0};

// Apex_L Endo, point (tag 2)
theta = 0.0;
r1 = d_focal * Sinh(nu_endo);
r2 = d_focal * Cosh(nu_endo);
mu = 0.0;
apex_endo = newp;
id = apex_endo;
Call EllipsoidPoint;

// Base_L Endo, point (tag 3)
mu_base = mu_base_gradi * Pi / 180.0;
mu = mu_base;
base_endo = newp;
id = base_endo;
Call EllipsoidPoint;

// Apex_L Epi, point (tag 4)
r1 = d_focal * Sinh(nu_epi);
r2 = d_focal * Cosh(nu_epi);
mu = 0.0;
apex_epi = newp;
id = apex_epi;
Call EllipsoidPoint;

// Base_L Epi, point (tag 5)
mu = Acos(Cosh(nu_endo) / Cosh(nu_epi) * Cos(mu_base));
base_epi = newp;
id = base_epi;
Call EllipsoidPoint;

// Apex Epi_R point (tag 6)
r1 = d_focal * Sinh(nu_epi);
r2 = d_focal * Cosh(nu_epi);
mu_base = mu_base_gradi_endo * Pi / 180.0;
mu = mu_base;
apex_R2 = newp;
id = apex_R2;
Call EllipsoidPoint;

// Apex Epi_R point (tag 7)
r1 = d_focal * Sinh(nu_endo);
r2 = d_focal * Cosh(nu_endo);
mu_base = mu_base_gradi_endo * Pi / 180.0;
mu = mu_base;
apex_R2 = newp;
id = apex_R2;
Call EllipsoidPoint;

/////////////////////////// Truncated Ventricle ///////////////////////////

Ellipse(1) = {7, 1, 2, 3};
Ellipse(2) = {6, 1, 4, 5};
Line(5) = {3, 5};
Line(6) = {7, 6};

Line Loop(1) = {1, 5, -2, -6};
Plane Surface(1) = {1};

Transfinite Line{5, 6} = N1;
Transfinite Line{1, 2} = N2;
Transfinite Surface{1};
Recombine Surface{1};

Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{1}; Layers{N}; Recombine;}

Physical Volume ("Myocardium", 1) = {1};
Physical Surface("Base down", 60) = {27};
Physical Surface("Base up", 50) = {19};
Physical Surface("Endocardium", 20) = {15};
Physical Surface("Epicardium", 10) = {23};
Physical Surface("Left wall", 30) = {1};
Physical Surface("Right wall", 40) = {28};

Mesh 3;
