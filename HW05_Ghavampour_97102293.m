%% Advance Nuero HW05 - Ali Ghavampour - 97102293
clear all; close all; clc;

%% Rescorla Wagner Rule ====================================================

%% Question 01
clear all; close all; clc;

nTrials = 200;
etha = 0.05;


% Extinction 
r = [ones(1,nTrials/2) , zeros(1,nTrials/2)];
w0 = 0;
u = ones(1,nTrials);
w = deltaRule(w0,nTrials,etha,u,r);
scatter(1:nTrials,w,'k','filled')
title("Extinction , etha = 0.05")
xlabel("Trial Number")
ylabel("W")


% Partial
alpha1 = 0.7;
alpha2 = 0.3;
r1 = rand(1,nTrials) < alpha1;
r2 = rand(1,nTrials) < alpha2;
w0 = 0;
u = ones(1,nTrials);
w1 = deltaRule(w0,nTrials,etha,u,r1);
w2 = deltaRule(w0,nTrials,etha,u,r2);

figure
hold all
scatter(1:nTrials,w1,'k')
yline(alpha1,'k','LineWidth',1.5,'HandleVisibility','Off');
scatter(1:nTrials,w2,'b')
yline(alpha2,'b','LineWidth',1.5,'HandleVisibility','Off');
hold off
ylim([0 1])
legend(sprintf("Alpha = %.1f",alpha1),sprintf("Alpha = %.1f",alpha2))
title("Prtial , etha = 0.05")
xlabel("Trial Number")
ylabel("W")


% Blocking
figure;
r = ones(1,nTrials);
w0 = [0;0];
u = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];
w = deltaRule(w0,nTrials,etha,u,r);
scatter(1:nTrials,w(1,:),'k','filled')
hold on
scatter(1:nTrials,w(2,:),'b','filled')
ylim([0 1])
legend("W1","W2")
title("Blocking , etha = 0.05")
xlabel("Trial Number")
ylabel("W")


% Inhibitory
figure;
u = [ones(1,nTrials) ; rand(1,nTrials) < 0.2];
r = 1-u(2,:);
w0 = [0;0];
w = deltaRule(w0,nTrials,etha,u,r);
scatter(1:nTrials,w(1,:),'k','filled')
hold on
scatter(1:nTrials,w(2,:),'b','filled')
ylim([-1 1])
legend("W1","W2")
title("Inhibitory , etha = 0.05")
xlabel("Trial Number")
ylabel("W")


% Overshadow
u = ones(2,nTrials);
r = ones(1,nTrials);
w0 = [0;0];
w = deltaRule(w0,nTrials,etha,u,r);
figure;
scatter(1:nTrials,w(1,:),'k','filled')
hold on
scatter(1:nTrials,w(2,:),'b')
ylim([0 1])
legend("W1","W2")
title("Overshadow , etha = 0.05")
xlabel("Trial Number")
ylabel("W")


%% Kalman Filter ==========================================================
clear all; close all; clc;

nTrials = 22;

% Figure 1B - Drift
s = rng;
v1 = normrnd(0,0.1,1,nTrials);
v2 = normrnd(0,0.1,1,nTrials);
w1 = zeros(1,nTrials);
w1(1) = 1;
w2 = zeros(1,nTrials);
w2(1) = 1;
for i = 2:nTrials
    w1(i) = w1(i-1) + v1(i-1);
    w2(i) = w2(i-1) + v2(i-1);
end
figure;
plot(0:nTrials-1,w1,'k','LineWidth',1.5)
hold on
plot(0:nTrials-1,w2,'--k','LineWidth',1.5)
ylim([0 2])
xlim([0 20])
legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
xlabel("Trial Number")
ylabel("W(t)")
title("Drift")


% Blocking
r = ones(1,nTrials);
w0 = [0;0];
C = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];
W_noise = eye(2)*0.01; % process noise
tau = 0.7; % Measurement noise
sigma0 = eye(2)*0.6;
[w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
sigma1 = [];
sigma2 = [];
for i = 1:nTrials
    tmp = sigma{i};
    sigma1 = [sigma1, tmp(1,1)];
    sigma2 = [sigma2, tmp(2,2)];
end

figure;
subplot(2,1,1)
plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
hold on
plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
hold on
xline(10,':k');
xlim([0 20])
ylim([0 1.2])
title("Blocking - mean")
ylabel("$\omega(t)$",'Interpreter','latex')
legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')

subplot(2,1,2)
plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
hold on 
plot(10:nTrials-1,sigma2(11:end),'--k','LineWidth',1.5)
hold on
xline(10,':k');
xlim([0 20])
ylim([0 1])
title("Blocking - Variance")
ylabel("$\sigma^2(t)$",'Interpreter','latex')
legend("${\sigma{^2}}_1$","${\sigma{^2}}_2$",'Interpreter','latex')


% Unblocking
r = [ones(1,nTrials/2),2*ones(1,nTrials/2)];
w0 = [0;0];
C = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];
W_noise = eye(2)*0.01;
tau = 0.6;
sigma0 = eye(2)*0.6;
[w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
sigma1 = [];
sigma2 = [];
for i = 1:nTrials
    tmp = sigma{i};
    sigma1 = [sigma1, tmp(1,1)];
    sigma2 = [sigma2, tmp(2,2)];
end

figure;
subplot(2,1,1)
plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
hold on
plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
hold on
xline(10,':k');
xlim([0 20])
ylim([0 1.2])
title("Unblocking - mean")
ylabel("$\omega(t)$",'Interpreter','latex')
legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')

subplot(2,1,2)
plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
hold on 
plot(10:nTrials-1,sigma2(11:end),'--k','LineWidth',1.5)
hold on
xline(10,':k');
xlim([0 20])
ylim([0 1])
title("Unblocking - Variance")
ylabel("$\sigma^2(t)$",'Interpreter','latex')
legend("${\sigma{^2}}_1$","${\sigma{^2}}_2$",'Interpreter','latex')



% Backward Blocking
r = ones(1,nTrials);
w0 = [0;0];
C = [ones(1,nTrials);ones(1,nTrials/2),zeros(1,nTrials/2)];
W_noise = eye(2)*0.02;
tau = 1.2;
sigma0 = eye(2)*0.6;
[w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
sigma1 = [];
sigma2 = [];
for i = 1:nTrials
    tmp = sigma{i};
    sigma1 = [sigma1, tmp(1,1)];
    sigma2 = [sigma2, tmp(2,2)];
end

figure;
subplot(2,1,1)
plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
hold on
plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
hold on
xline(10,':k');
xlim([0 20])
ylim([0 1.2])
title("Backward Blocking - Mean")
ylabel("$\omega(t)$",'Interpreter','latex')
legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')

subplot(2,1,2)
plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
hold on 
plot(10:nTrials-1,sigma2(11:end),'--k','LineWidth',1.5)
hold on
xline(10,':k');
xlim([0 20])
ylim([0 1])
title("Backward Blocking - Variance")
ylabel("$\sigma^2(t)$",'Interpreter','latex')
legend("${\sigma{^2}}_1$","${\sigma{^2}}_2$",'Interpreter','latex')


% Figure 2
sigma1 = sigma{1};
sigma9 = sigma{9};
sigma19 = sigma{19};
sigmaVec = {sigma1, sigma9, sigma19};
w1 = w(:,1);
w9 = w(:,9);
w19 = w(:,19);
wVec = {w1, w9, w19};

r = 0.3:0.3:3;
r = flip(r);
colorCode = linspace(0,1,length(r));
for i = 1:length(colorCode)
    color{i} = [colorCode(i),colorCode(i),colorCode(i)];
end
theta = 0:0.01:2*pi;
time = [1 9 19];
figure
for sub = 1:3
    subplot(1,3,sub)
    tmpW = wVec{sub};
    sigmaTmp = sigmaVec{sub};
    for i = 1:length(r)
        x = r(i)*sin(theta);
        y = r(i)*cos(theta);
        for j = 1:length(x)
            tmp = [x(j);y(j)];
            tmp = sigmaTmp * tmp;
            x(j) = tmp(1)+tmpW(1);
            y(j) = tmp(2)+tmpW(2);
        end
        fill(x,y,color{i},'LineStyle','none')
        axis square
        hold on
    end
    scatter(tmpW(1),tmpW(2),'*k')
    mu = wVec{sub}';
    sigma = sigmaVec{sub};
    rng('default')  % For reproducibility
    X = mvnrnd(mu,sigma,1000);
    scatter(mean(X(:,1)),mean(X(:,2)),'*r')
    hold on
    scatter(downsample(X(:,1),1),downsample(X(:,2),1),0.6,'r')
    
    xlim([-1 2])
    ylim([-1 2])
    set(gca,'Color','k')
    xlabel("$\omega_1$",'interpreter','LaTex')
    ylabel("$\omega_2$",'interpreter','LaTex')
    title(sprintf("t = %d",time(sub)))
end

% Question 5 
r = [ones(1,nTrials/2),-1*ones(1,nTrials/2)];
w0 = 0;
C = ones(1,nTrials);
W_noise = 0.02;
tau = 1.2;
sigma0 = 0.6;
[w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
sigma1 = [];
for i = 1:nTrials
    tmp = sigma{i};
    sigma1 = [sigma1, tmp];
end

figure;
subplot(2,1,1)
plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
hold on
xline(10,':k');
xlim([0 20])
ylim([-1.2 1.2])
title("s1->r and s1->-r , Mean")
ylabel("$\omega(t)$",'Interpreter','latex')
legend("$\omega_1(t)$",'Interpreter','latex')

subplot(2,1,2)
plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
hold on 
xline(10,':k');
xlim([0 20])
ylim([0 1])
title("s1->r and s1->-r , Variance")
ylabel("$\sigma^2(t)$",'Interpreter','latex')
legend("${\sigma{^2}}_1$",'Interpreter','latex')


%% Question02 - Changing Noise - Blocking
clear all; close all; clc;

taun = 0.1:0.4:2.2;
wn = 0:0.02:0.11;
nTrials = 22;

% Blocking
for param = 1:6
    r = ones(1,nTrials);
    w0 = [0;0];
    C = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];
    W_noise = eye(2)*wn(param); % process noise
    tau = 0.7; % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,param)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.2])
    title(sprintf("Blocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
end

figure;
for param = 1:6
    r = ones(1,nTrials);
    w0 = [0;0];
    C = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];
    W_noise = eye(2)*0.01; % process noise
    tau = taun(param); % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,param)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.2])
    title(sprintf("Blocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
end

figure;
for param = 1:2:6
    r = ones(1,nTrials);
    w0 = [0;0];
    C = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];
    W_noise = eye(2)*wn(param); % process noise
    tau = 0.7; % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,(param+1)/2)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.2])
    title(sprintf("Blocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
    
    subplot(2,3,(param+1)/2 + 3)
    plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
    hold on 
    plot(10:nTrials-1,sigma2(11:end),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1])
    title(sprintf("Blocking(variance) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\sigma^2(t)$",'Interpreter','latex')
    legend("${\sigma{^2}}_1$","${\sigma{^2}}_2$",'Interpreter','latex')
end

figure;
for param = 1:2:6
    r = ones(1,nTrials);
    w0 = [0;0];
    C = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];
    W_noise = eye(2)*0.01; % process noise
    tau = taun(param); % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,(param+1)/2)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.2])
    title(sprintf("Blocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
    
    subplot(2,3,(param+1)/2 + 3)
    plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
    hold on 
    plot(10:nTrials-1,sigma2(11:end),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1])
    title(sprintf("Blocking(variance) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\sigma^2(t)$",'Interpreter','latex')
    legend("${\sigma{^2}}_1$","${\sigma{^2}}_2$",'Interpreter','latex')
end


%% Question02 - Changing Noise - Unblocking
clear all; close all; clc;

taun = 0.1:0.4:2.2;
wn = 0:0.02:0.11;
nTrials = 22;

r = [ones(1,nTrials/2),2*ones(1,nTrials/2)];
w0 = [0;0];
C = [ones(1,nTrials);zeros(1,nTrials/2),ones(1,nTrials/2)];

for param = 1:6
    W_noise = eye(2)*wn(param); % process noise
    tau = 0.7; % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,param)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.5])
    title(sprintf("Unblocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
end

figure;
for param = 1:6
    W_noise = eye(2)*0.01; % process noise
    tau = taun(param); % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,param)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.5])
    title(sprintf("Unblocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
end

figure;
for param = 1:2:6
    W_noise = eye(2)*wn(param); % process noise
    tau = 0.7; % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,(param+1)/2)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.5])
    title(sprintf("Unblocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
    
    subplot(2,3,(param+1)/2 + 3)
    plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
    hold on 
    plot(10:nTrials-1,sigma2(11:end),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1])
    title(sprintf("Unblocking(variance) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\sigma^2(t)$",'Interpreter','latex')
    legend("${\sigma{^2}}_1$","${\sigma{^2}}_2$",'Interpreter','latex')
end

figure;
for param = 1:2:6
    W_noise = eye(2)*0.01; % process noise
    tau = taun(param); % Measurement noise
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end
    subplot(2,3,(param+1)/2)
    plot(0:nTrials-1,w(1,:),'k','LineWidth',1.5)
    hold on
    plot(0:nTrials-1,w(2,:),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1.5])
    title(sprintf("Unblocking(mean) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\omega(t)$",'Interpreter','latex')
    legend("$\omega_1(t)$","$\omega_2(t)$",'Interpreter','latex')
    
    subplot(2,3,(param+1)/2 + 3)
    plot(0:nTrials-1,sigma1,'k','LineWidth',1.5)
    hold on 
    plot(10:nTrials-1,sigma2(11:end),'--k','LineWidth',1.5)
    hold on
    xline(10,':k');
    xlim([0 20])
    ylim([0 1])
    title(sprintf("Unblocking(variance) , Process Noise = %.2f, Measure Noise = %.2f",W_noise(1,1),tau))
    ylabel("$\sigma^2(t)$",'Interpreter','latex')
    legend("${\sigma{^2}}_1$","${\sigma{^2}}_2$",'Interpreter','latex')
end


%% Question02 - Changing Noise - Backward Blocking
clear all; close all; clc;

taun = 0.4:0.2:2.2;
wn = 0.02:0.05:1;
nTrials = 22;

re = ones(1,nTrials);
w0 = [0;0];
C = [ones(1,nTrials);ones(1,nTrials/2),zeros(1,nTrials/2)];

subp = [0,3,6];
% Backward Blocking
for param = 1:2:6
    W_noise = eye(2)*wn(param);
    tau = 0.7;
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,re);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end

    % Figure 2
    sigma1 = sigma{1};
    sigma9 = sigma{9};
    sigma19 = sigma{19};
    sigmaVec = {sigma1, sigma9, sigma19};
    w1 = w(:,1);
    w9 = w(:,9);
    w19 = w(:,19);
    wVec = {w1, w9, w19};

    r = 0.3:0.3:2.4;
    r = flip(r);
    colorCode = linspace(0,1,length(r));
    for i = 1:length(colorCode)
        color{i} = [colorCode(i),colorCode(i),colorCode(i)];
    end
    theta = 0:0.01:2*pi;
    time = [1 9 19];
    for sub = 1:3
        subplot(3,3,sub+subp((param+1)/2))
        tmpW = wVec{sub};
        sigmaTmp = sigmaVec{sub};
        for i = 1:length(r)
            x = r(i)*sin(theta);
            y = r(i)*cos(theta);
            for j = 1:length(x)
                tmp = [x(j);y(j)];
                tmp = sigmaTmp * tmp;
                x(j) = tmp(1)+tmpW(1);
                y(j) = tmp(2)+tmpW(2);
            end
            fill(x,y,color{i},'LineStyle','none')
            axis square
            hold on
        end
        scatter(tmpW(1),tmpW(2),'*k')
        xlim([-1 2])
        ylim([-1 2])
        set(gca,'Color','k')
        xlabel("$\omega_1$",'interpreter','LaTex')
        ylabel("$\omega_2$",'interpreter','LaTex')
        title(sprintf("t = %d , Process Noise = %.2f , Measurment Noise = %.2f",time(sub),W_noise(1,1),tau))
    end
end

figure;
for param = 1:2:6
    W_noise = eye(2)*0.6;
    tau = taun(param);
    sigma0 = eye(2)*0.6;
    [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,re);
    sigma1 = [];
    sigma2 = [];
    for i = 1:nTrials
        tmp = sigma{i};
        sigma1 = [sigma1, tmp(1,1)];
        sigma2 = [sigma2, tmp(2,2)];
    end

    % Figure 2
    sigma1 = sigma{1};
    sigma9 = sigma{9};
    sigma19 = sigma{19};
    sigmaVec = {sigma1, sigma9, sigma19};
    w1 = w(:,1);
    w9 = w(:,9);
    w19 = w(:,19);
    wVec = {w1, w9, w19};

    r = 0.3:0.3:2.4;
    r = flip(r);
    colorCode = linspace(0,1,length(r));
    for i = 1:length(colorCode)
        color{i} = [colorCode(i),colorCode(i),colorCode(i)];
    end
    theta = 0:0.01:2*pi;
    time = [1 9 19];
    for sub = 1:3
        subplot(3,3,sub+subp((param+1)/2))
        tmpW = wVec{sub};
        sigmaTmp = sigmaVec{sub};
        for i = 1:length(r)
            x = r(i)*sin(theta);
            y = r(i)*cos(theta);
            for j = 1:length(x)
                tmp = [x(j);y(j)];
                tmp = sigmaTmp * tmp;
                x(j) = tmp(1)+tmpW(1);
                y(j) = tmp(2)+tmpW(2);
            end
            fill(x,y,color{i},'LineStyle','none')
            axis square
            hold on
        end
        scatter(tmpW(1),tmpW(2),'*k')
        xlim([-1 2])
        ylim([-1 2])
        set(gca,'Color','k')
        xlabel("$\omega_1$",'interpreter','LaTex')
        ylabel("$\omega_2$",'interpreter','LaTex')
        title(sprintf("t = %d , Process Noise = %.2f , Measurment Noise = %.2f",time(sub),W_noise(1,1),tau))
    end
end

%% Generalized Kalman Filter ===================================================
clear all; close all; clc;

nTrials = 100;
s = rng;
v = normrnd(0,0.1,1,nTrials);
vr = normrnd(0,0.5,1,nTrials);
phi = normrnd(0,2,1,nTrials);
% c = rand(1,nTrials) <= 0.02;
c = zeros(1,nTrials);
c(40) = 1;
c(90) = 1;
phi(40) = -2;
phi(90) = 4;
ind = find(c==1);
w = zeros(1,nTrials);
r = w;
w(1) = 0;
r(1) = w(1) + vr(1) + c(1)*phi(1);
for i = 2:nTrials
    w(i) = w(i-1) + v(i-1) + c(i-1) * phi(i-1);
    r(i) = w(i) + vr(i) + c(i)*phi(i);
end
figure;
subplot(3,1,1)
scatter(0:nTrials-1,w,20,'filled','k')
hold on
scatter(0:nTrials-1,r,'xk')
hold on
scatter([0,ind],[w(1),w(ind+1)],500,'k','LineWidth',2)
hold on
for i = 1:length(ind)
    xline(ind(i),':k');
end
xlim([0 nTrials])
ylim([-4 4])
title("Slow Draft and Dramatic Changes")
xlabel("t")
ylabel("w")
legend("w(t)","r(t)",'location','northwest')


% Learning Weights
gamma = 3.3;
W_noise = 0.01;
w0 = 0;
tau = 0.7;
sigma0 = 0.6;
u = ones(1,nTrials);
[w_learn,sigma,beta] = proKalman(gamma,W_noise,w0,nTrials,tau,sigma0,u,r);
subplot(3,1,2)
scatter(0:nTrials-1,w,20,'filled','k')
hold on
scatter(0:nTrials-1,r,'xk')
hold on
scatter(0:nTrials-1,w_learn,'k');
xlim([0 nTrials])
ylim([-4 4])
legend("theoretical w(t)","r(t)","estimated w(t)",'location','northwest')
title(sprintf("Learned w alongside r and theoretical w, gamma = %.2f",gamma))
xlabel("t")

subplot(3,1,3)
plot(sigma,'--k','LineWidth',1.5)
hold on
plot(beta,'k','LineWidth',1.5)
hold on
yline(gamma,'-.k','LineWidth',1.5);
ylim([0 10])
legend("ACh","NE","$gamma$",'Interpreter','LaTex','location','northwest')
xlabel("t")

% MSE
n = 100;
gammaVec = linspace(0,30,n);
mse = zeros(1,n);
for j = 1:1000
    nTrials = 100;
    s = rng;
    v = normrnd(0,0.1,1,nTrials);
    vr = normrnd(0,0.5,1,nTrials);
    phi = normrnd(0,2,1,nTrials);
    % c = rand(1,nTrials) <= 0.02;
    c = zeros(1,nTrials);
    c(40) = 1;
    c(90) = 1;
    phi(40) = -2;
    phi(90) = 4;
    ind = find(c==1);
    w = zeros(1,nTrials);
    r = w;
    w(1) = 0;
    r(1) = w(1) + vr(1) + c(1)*phi(1);
    for i = 2:nTrials
        w(i) = w(i-1) + v(i-1) + c(i-1) * phi(i-1);
        r(i) = w(i) + vr(i) + c(i)*phi(i);
    end
    mseTmp = [];
    for i1 = 1:n
        gamma = gammaVec(i1);
        [w_learn,sigma,beta] = proKalman(gamma,W_noise,w0,nTrials,tau,sigma0,u,r);
        tmp = sum((w_learn-w).^2);
        mseTmp = [mseTmp,tmp];
    end
    mse = mse + mseTmp;
end
mse = mse/1000;
figure
plot(gammaVec,mse,'k','LineWidth',1.5)
title("MSE for different gamma")
xlabel("$\gamma$",'interpreter','LaTex')
ylabel("MSE")


%% Functions 
function [w,sigma,betaVec] = proKalman(gamma,W_noise,w0,nTrials,tau,sigma0,u,r)
    w = zeros(1,nTrials);
    sigma = zeros(1,nTrials);
    betaVec = zeros(1,nTrials);
    w(1) = w0;
    sigma(1) = sigma0;
    for i = 2:nTrials
        % prediction
        sigmap = sigma(i-1) + W_noise;
        
        % update
        G = sigmap * u(i) * (u(i)*sigmap + tau^2)^-1;
        sigma(i) = sigmap - G*u(i)*sigmap;
        w(i) = w(i-1) + G*(r(i) - u(i)*w(i-1));
        
        % NE
        beta = (r(i) - u(i)*w(i))^2 / (u(i)*sigma(i) + tau^2);
        betaVec(i) = beta; 
        if beta > gamma
            sigma(i) = 100;
        end
    end
end

function [w,sigma] = myKalman(W_noise,w0,nTrials,tau,sigma0,C,r)
    w = zeros(size(C,1),nTrials);
    sigma = cell(1,nTrials);
    w(:,1) = w0;
    sigma{1} = sigma0;
    if size(C,1) > 1
        ind = find(C(2,:));
        ind = ind(1)-1;
    end
    for i = 2:nTrials
        % Prediction
        wp = w(:,i-1);
        sigmap = sigma{i-1} + W_noise;
        
        % Update
        G = sigmap * C(:,i) * (C(:,i)' * sigmap * C(:,i) + tau^2)^-1;
        sigma{i} = sigmap - G*C(:,i)'*sigmap;
        (r(i) - C(:,i)'*w(:,i-1));
        w(:,i) = w(:,i-1) + G * (r(i) - C(:,i)'*w(:,i-1));
        
        if size(C,1) > 1
            if (i == ind)
                tmp = sigma{i};
                tmp(2,2) = 0.6;
                sigma{i} = tmp;
            end
        end
    end
end

function w = deltaRule(w0,nTrials,etha,u,r)
    w = zeros(size(w0,1),nTrials);
    w(:,1) = w0;
    for i = 2:nTrials
        delta = r(i-1) - u(:,i-1)' * w(:,i-1);
        w(:,i) = w(:,i-1) + etha*delta*u(:,i-1);
    end
end









