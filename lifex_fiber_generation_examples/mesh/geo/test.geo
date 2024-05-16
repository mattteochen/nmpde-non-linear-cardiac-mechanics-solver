// Define the parameters of the ellipsoid
h = 1.0; // Height
a = 2.0; // Semi-axis length in x-direction
b = 1.5; // Semi-axis length in y-direction
c = 1.0; // Semi-axis length in z-direction

// Define the ellipsoid
Mesh.CharacteristicLengthMax = 0.1; // Max mesh element size
Point(1) = {0, 0, 0, Mesh.CharacteristicLengthMax}; // Center of the ellipsoid
Ellipse(1) = {1, a, b, c}; // Create the ellipsoid


Mesh 3;
