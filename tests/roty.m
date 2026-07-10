function R = roty(a)
% degree-based y rotation (Phased Array Toolbox convention)
R = [cosd(a) 0 sind(a); 0 1 0; -sind(a) 0 cosd(a)];
end
