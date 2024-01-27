////////////////// Idealized left ventricular geo ////////////////
//// \author Roberto Piersanti <roberto.piersanti@polimi.it> /////
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//// This work is licensed under a Creative Commons       ////
//// Attribution-NonCommercial 4.0 International License. ////
//////////////////////////////////////////////////////////////

// LV parameters [m].

h = 8; // Mesh size

d_focal = 45.0;
nu_epi   = 0.8;
nu_endo  = 0.6;
mu_base_gradi = 110.0;

//////////////////////////////////////////////////////////
// Function for point in Prolate Spheroidal Coordinates //
//////////////////////////////////////////////////////////

Function EllipsoidPoint
        Point(id) = { r1 * Sin(mu) * Cos(theta),
                      r1 * Sin(mu) * Sin(theta),
                      r2 * Cos(mu),
                      h };
Return

//////////////////////////////////////////////////////////

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

/////////////////////////// Left ventricle ///////////////////////////

Ellipse(1) = {3, 1, 2, 2};
Ellipse(2) = {5, 1, 4, 4};
Line(3) = {4, 2};
Line(4) = {3, 5};
Line Loop(1) = {1, -3, -2, -4};

Plane Surface(1) = {1};

Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{1}; }
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{21};}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{38};}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi / 2} {Surface{55};}

Physical Volume ("Myocardium", 1) = {1, 2, 3, 4};
Physical Surface ("Basal plane", 50) = {37, 54, 71, 20};
Physical Surface ("Endocardium", 20) = {29, 12, 46, 63};
Physical Surface ("Epicardium",  10) = {16, 33, 50, 67};

Mesh 3;
