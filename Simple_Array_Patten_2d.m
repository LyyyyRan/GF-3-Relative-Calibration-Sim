% 8阵元均匀线阵方向图，来波方向为0度
clc; clear all; close all;

% for Chinese charectors:
set(gca,'Fontname','Monospaced');

% 阵元数为16:
element_num = 16;

% 阵元间距d与波长lamda的关系:
d_lamda = 1 / 2;

% 来波方向:
theta = linspace(-pi / 2, pi / 2, 900);
theta0 = deg2rad(-3);
phi = linspace(-pi / 2, pi / 2, 900);
phi0 = deg2rad(5);

% 导向矢量:
SteeringVector_X = exp(1i * 2 * pi * d_lamda * sin(theta0) .* [0:element_num-1]');
SteeringVector_Y = exp(1i * 2 * pi * d_lamda * sin(phi0) .* [0:element_num-1]');

% Init Patten:
Gx = zeros(1, length(theta), 1);
Gy = zeros(1, length(phi), 1);

% Calculate Patten X:
for jdx = 1 : length(theta)
    a = exp(1j * 2 * pi * d_lamda * sin(theta(jdx)) .* [0 : element_num - 1]');
    Gx(:, jdx) = SteeringVector_X' * a;
end

% Calculate Patten Y:
for jdx = 1 : length(phi)
    a = exp(1j * 2 * pi * d_lamda * sin(phi(jdx)) .* [0 : element_num - 1]');
    Gy(:, jdx) = SteeringVector_Y' * a;
end

% 1d to 2d:
Gxy = Gx .* Gy.';

save('./mat_files/Simple_Gx.mat', 'Gx')
save('./mat_files/Simple_Gy.mat', 'Gy')
save('./mat_files/Simple_Gxy.mat', 'Gxy')

% show Gx:
figure(1);
plot(theta,abs(Gx));
grid on
xlabel('\theta(rad)')
ylabel('Modulus')
title('Gx(\theta)')

% show Gx_dB:
figure(2);
Gx_dB = db(abs(Gx) / max(abs(Gx)));
plot(theta, Gx_dB),
grid on
xlabel('\theta(rad)')
ylabel('Modulus(dB)')
title('Gx(\theta)')

% show Gy:
figure(3);
plot(phi, abs(Gy));
grid on
xlabel('\phi(rad)')
ylabel('Modulus')
title('Gy(\phi)')

% show Gy_dB:
figure(4);
Gy_dB = db(abs(Gy) / max(abs(Gy)));
plot(phi, Gy_dB),
grid on
xlabel('\phi(rad)')
ylabel('Modulus(dB)')
title('Gy(\phi)')

% show Gxy(θ, φ):
figure(5);
imagesc(abs(Gxy))
xlabel('\theta(rad)');
ylabel('\phi(rad)');
title("Gxy(\theta, \phi)");

% show Modulus as 3d:
figure(6);
[theta_3d, phi_3d] = meshgrid(theta, phi);
mesh(theta_3d, phi_3d, abs(Gxy));
xlabel('\theta(rad)');
ylabel('\phi(rad)');
zlabel('Modulus')
title('Gxy-3d');

% show Modulus as 3d-dB:
figure(7);
[theta_3d, phi_3d] = meshgrid(theta, phi);
mesh(theta_3d, phi_3d, db(abs(Gxy)));
xlabel('\theta(rad)');
ylabel('\phi(rad)');
zlabel('Modulus(dB)')
title('Gxy-3d');
