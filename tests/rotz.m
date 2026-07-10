function R = rotz(a)
% degree-based z rotation (Phased Array Toolbox convention)
R = [cosd(a) -sind(a) 0; sind(a) cosd(a) 0; 0 0 1];
end
