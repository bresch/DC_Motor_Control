clear all
close all
clc

% Motor parameters
R = 1.1;        % (Ohm)
L = 2.6e-3;     % (H)
J = 0.07;       % (kg/m^3)
b = 0.002;      % (N*m*s/rad)
Ki = 1.1;       % (N*m/A)
Kb = 1.1;       % (N*m/A)
T = 0.1;        % (N*m)

dt = 0.01;      % Sample time

% x = [
%       i;      % Motor current
%       omega;  % Rotor speed
%       theta;  % Rotor position
%       T];     % Load

% Continuous extended time model
F = [
    -R/L    -Kb/L   0   0;
    Ki/J    -b/J    0   -1/J;
    0       1       0   0;
    0       0       0   0];

G = [
    1/L;
    0;
    0;
    0];

H = [0 0 1 0];
   

% Model noise
stdI = 0.01;
std0mega = 0.01;
stdTheta = 0.01;
stdT = 0.1;

% Sensor noise
stdHall = 0.068*pi/180; % AS5047D Datasheet

Q = diag([stdI^2 std0mega^2 stdTheta^2 stdT^2]);
% Discretization
% Psik = G*dt;        % 1st order approx
Psik = [0.8198; 0.1054; 0.0004449; 0]; 
[Phik, Qk] = FQ2PhikQk(F, Q, dt);
% Phik = I + F*dt;  % 1st order approx
% Qk = diag([stdI^2 std0mega^2 stdTheta^2 stdT^2]*dt); % Approx
Rk = stdHall^2/dt;
Qk = eye(4)*dt;
Rk = 1;

t = 0:dt:10;
nEpochs = length(t);

% initialize vectors
i = zeros(1,nEpochs);
omega = zeros(1,nEpochs);
theta = zeros(1,nEpochs);
T = zeros(1,nEpochs);

% initial state
i(1) = 0;
omega(1) = 0;
theta(1) = 0;
T(1) = 0;

xState = [
    i(1);
    omega(1);
    theta(1);
    T(1)];

Pk = diag([0.1 0.1 0.5 1]);

inn_mem = zeros(1,nEpochs);
res_mem = zeros(1,nEpochs);
Kk_mem = zeros(4,nEpochs);
Pk_mem = zeros(4,nEpochs);
Pk_mem(:,1) = [Pk(1,1);Pk(2,2);Pk(3,3);Pk(4,4)];

% System simulation
Jm = 2*J;
Lm = L*1.2;
Fm = [
    -R/Lm    -Kb/Lm   0   0;
    Ki/Jm    -b/Jm    0   -1/Jm;
    0       1       0   0;
    0       0       0   0];

sys = ss(Fm,G,H,0);
% sys = ss(Phik, Psik, H, 0,dt);
% u = ones(length(t),1);
u = sin(t);
x0 = [0 0 pi 0.5]';
[ySim,t,statesSim] = lsim(sys,u,t,x0);
% ySim(500:end) = ySim(499)*ones(length(ySim(500:end)),1);
meas = ySim + stdHall*randn(size(t));

% Kalman simulation
for k = 2:nEpochs
    xPred = Phik*xState + Psik*u(k);    % State prediction
    PkPred = Phik*Pk*Phik'+Qk;          % State covariance prediction
    
    inn = meas(k) - H*xPred;            % Innovation
    
    S = H*PkPred*H' + Rk;               % Innovation covariance
    
%     Kk = PkPred*H'*S^(-1);              % Kalman gain
    p = PkPred(3,3);
    r = Rk;
    Kk = (1/(p+r))*PkPred(:,3);
    
    xState = xPred + Kk*inn;            % State update
    Pk = PkPred - Kk*S*Kk';             % State covariance update
    
    % Store values
    i(k) = xState(1);
    omega(k) = xState(2);
    theta(k) = xState(3);
    T(k) = xState(4);
    inn_mem(k) = inn;
    res_mem(k) = theta(k) - H*xPred;
    Pk_mem(:,k) = [Pk(1,1);Pk(2,2);Pk(3,3);Pk(4,4)];
    Kk_mem(:,k) = Kk;
    
    % Adaptive algorithm for Qk and Rk
    if mod(k,101) == 0
        C_dz = sum(res_mem(k-100:k).*res_mem(k-100:k))/100;
        Qk = Kk*C_dz*Kk';
        Rk = C_dz+H*Pk*H';
    end
end

figure;
subplot(4,1,1);
hold all;
plot(t,i);
plot(t,statesSim(:,1));
plot(t,i+sqrt(Pk_mem(1,:)),'g--');
plot(t,i-sqrt(Pk_mem(1,:)),'g--');
hold off
ylabel('Current (A)');
legend('Estimate', 'Reference', 'Confidence interval'); 

subplot(4,1,2);
hold all;
plot(t,omega);
plot(t,statesSim(:,2));
plot(t,omega+sqrt(Pk_mem(2,:)),'g--');
plot(t,omega-sqrt(Pk_mem(2,:)),'g--');
hold off
ylabel('Rotor speed (rad/s)');

subplot(4,1,3);
hold all;
plot(t,theta);
plot(t,statesSim(:,3));
plot(t,theta+sqrt(Pk_mem(3,:)),'g--');
plot(t,theta-sqrt(Pk_mem(3,:)),'g--');
hold off
ylabel('Rotor position (rad)');

subplot(4,1,4);
hold all;
plot(t,T);
plot(t,statesSim(:,4));
plot(t,T+sqrt(Pk_mem(4,:)),'g--');
plot(t,T-sqrt(Pk_mem(4,:)),'g--');
hold off
xlabel('Time (s)');
ylabel('External torque (N)');

figure;
stem(t,inn_mem);