%% Advance Neuro HW08 - Ali Ghavampour - 97102293

%% Part 1 - Simple Model
clear all; close all; clc;
rng shuffle

% Question 2
T = 1;
dt = 0.1;
bias = 1;
sigma = 1;
choiceVec = [];
xVec = [];
trlNum = 1000;
for trl = 1:trlNum
    [t,x,choice] = simple_model(bias,sigma,dt,T);
    choiceVec(trl) = choice;
end
fprintf("Correct Choice Percentage = %.4f\n",length(find(choiceVec==1))/trlNum*100)


T = 1;
dt = 0.01;
bias = 4;
sigma = 1;
choiceVec = [];
xVec = [];
trlNum = 1000;
for trl = 1:trlNum
    [t,x,choice] = simple_model(bias,sigma,dt,T);
    choiceVec(trl) = choice;
    xVec = [xVec;x];
end

meanX = mean(xVec,1);
varX = var(xVec,[],1);
subplot(2,1,1);
plot(t,varX,'k')
title("Variance of data over time , $varX(t) = t$",'interpreter','Latex')
xlabel("Time(s)")
ylabel("Variance")
subplot(2,1,2);
plot(t,meanX,'k')
title("Mean of data over time , $meanX(t) = bias\times t$ where $bias = 4$",'interpreter','Latex')
xlabel("Time(s)")
ylabel("Mean")


biasVec = [-1,0,0.1,1,10];
T = 10;
trlNum = 20;
dt = 0.01;
sigma = 1;
choiceVec = {};
xVec = {};
cnt = 1;
for bias = biasVec
    xVecTmp = [];
    choiceVecTmp = [];
    [t,x,choice] = simple_model(bias,sigma,dt,T);
    choiceVecTmp(trl) = choice;
    xVecTmp = [xVecTmp;x];
    choiceVec{cnt} = choiceVecTmp;
    xVec{cnt} = xVecTmp;
    cnt = cnt+1;
end
figure;
for i = 1:length(biasVec)
    tmp = xVec{i};
    plot(t,tmp,'linewidth',1.5);
    hold on
    ylim([-15,102])
end
xlabel("Time(s)")
ylabel("Decision Variable")
legend("B = -1","B = 0","B = 0.1","B = 1","B = 10",'location','northwest');

%% Question 3
clear all; close all; clc;

b = 0.1;
sigma = 1;
dt = 0.1;
TVec = linspace(0.5,10,50);
trlNum = 1000;
err = [];
errTheory = [];
for T = TVec
    rng shuffle
    choiceVec = [];
    % Simulation
    for trl = 1:trlNum
        [~,~,choice] = simple_model(b,sigma,dt,T);
        choiceVec = [choiceVec, choice];
    end
    tmp = length(find(choiceVec == 1));
    err = [err, 1-tmp/trlNum];
    
    %theoretical
    fun = @(x) 1/(sqrt(2*pi*sigma*T)) * exp(-1/(2*sigma*T) * (x-b*T).^2);
    tmp = integral(fun,-inf,0);
    errTheory = [errTheory,tmp];
    
end

plot(TVec,err,'k','linewidth',1.5)
hold on
plot(TVec,errTheory,'r','linewidth',1.5)
legend("Simulation","Theoretical")
xlabel("Duration(s)")
ylabel("error")
title("Error vs Sim Duration")
ylim([0,0.6])
xlim([0.5,10])

%% Question 4
clear all; close all; clc;
rng shuffle

b = 0.1;
sigma = 1;
dt = 0.1;
T = 10;
trlNum = 1000;

xVec = [];
for trl = 1:trlNum
    [t,x,~] = simple_model(b,sigma,dt,T);
    xVec = [xVec;x];
end
stdX = std(xVec,[],1);
meanX = mean(xVec,1);
meanSim = b*t;
stdSim = sqrt(sigma*t);

figure;
plot(t,xVec(1,:),'color',[0.7,0.7,0.7]);
hold on
plot(t,xVec(2:50:end,:),'color',[0.7,0.7,0.7],'handlevisibility','off');
hold on
plot(t,meanX,'k','linewidth',1.5)
hold on
plot(t,meanX+stdX,'--k','linewidth',1.5,'handlevisibility','off')
hold on
plot(t,meanX-stdX,'--k','linewidth',1.5,'handlevisibility','off')
hold on
plot(t,meanSim,'r')
hold on
plot(t,meanSim+stdSim,'--r','handlevisibility','off')
hold on
plot(t,meanSim-stdSim,'--r','handlevisibility','off')

title("Mean and std of decision variable")
xlabel("Time(s)")
ylabel("Desision Variable")
legend("decision variable","Simulation Mean and std","Theoretical Mean and std"...
    ,'location','northwest')

%% Question 5
clear all; close all; clc;
rng shuffle

bias = 0.1;
sigma = 1;
startVec = linspace(-6,6,51);
TVec = [0.1,1,10,20,50,100];
trlNum = 5000;
for i = 1:length(TVec)
    T = TVec(i);
    choiceVec = [];
    tic
    for start = startVec
        fprintf("start = %.4f\n",start);
        choiceVecTmp = [];
        for trl = 1:trlNum
            choice = simple_model2(bias,sigma,start,T);
            choiceVecTmp(trl) = choice;
        end
        choiceVec = [choiceVec,length(find(choiceVecTmp==1))/trlNum*100];
    end
    toc

    subplot(2,3,i)
    plot(startVec,choiceVec,'k','linewidth',1.5)
    text(0.3,choiceVec(find(startVec==0)),sprintf("P = %.2f percent",choiceVec(find(startVec==0))));
    hold on
    scatter(0,choiceVec(find(startVec==0)),'r','filled')
    hold on
    xline(0,'--r');
    xlabel("starting point")
    ylabel("Percentage of Right Choice")
    title(sprintf("Percentage of Right Choice - Time = %.2f s",T))
    ylim([0 100])
end


%% Question 7
clear all; close all; clc;

trlNum = 10000;
posTh = 5;
negTh = -5;
bias = 0.1;
sigma = 1;
x0 = 0;
dt = 0.01;
RTVec = [];
choiceVec = [];
for trl = 1:trlNum
    trl
    [RT,choice] = two_choice_trial(posTh,negTh,sigma,x0,bias,dt);
    RTVec(trl) = RT;
    choiceVec(trl) = choice;
end

crctInd = find(choiceVec == 1);
wrInd = find(choiceVec == -1);
nbin = 100;
[cnt,x] = hist(RTVec,nbin);
[cntCorrect,xCorrect] = hist(RTVec(crctInd),nbin);
[cntWR,xWR] = hist(RTVec(wrInd),nbin);

figure;
subplot(1,3,1)
plot(x,cnt,'k','LineWidth',1.5)
xlim([0,max(RTVec)])
ylim([0,700])
xlabel("RT(s)")
ylabel("Count")
title("RT distribution")

subplot(1,3,2)
plot(xCorrect,cntCorrect,'k','LineWidth',1.5)
xlim([0,max(RTVec)])
ylim([0,700])
xlabel("RT(s)")
ylabel("Count")
title("RT distribution for correct trials")

subplot(1,3,3)
plot(xWR,cntWR,'k','LineWidth',1.5)
xlim([0,max(RTVec)])
ylim([0,700])
xlabel("RT(s)")
ylabel("Count")
title("RT distribution for  wrong trials")

figure;
cntCorrect = cntCorrect/max(cntCorrect);
cntWR = cntWR/max(cntWR);
plot(xCorrect,cntCorrect,'k','LineWidth',1.5)
hold on
plot(xWR,cntWR,'r','LineWidth',1.5)
xlabel("RT(s)")
ylabel("Count")
title("Normalized Count")
legend("Correct","Wrong")
xlim([0,max(RTVec)])


%% Question 8
clear all; close all; clc;
rng shuffle

th1 = 5;
th2 = 5;
sigma1 = 1;
sigma2 = 1;
b1 = 0.1;
b2 = 0.1;
dt = 0.01;
trlNum = 500;

b1Vec = linspace(0.1,5,20);
choiceVec = [];
for b1 = b1Vec
    choiceVecTmp = [];
    for trl = 1:trlNum
        trl
        [RT,choice] = race_trial(th1,th2,sigma1,sigma2,b1,b2,dt,1000);
        choiceVecTmp(trl) = choice;
    end
    tmp = length(find(choiceVecTmp==1));
    choiceVec = [choiceVec,tmp];
end
choiceVec = choiceVec/trlNum*100;


figure;
plot(b1Vec,choiceVec,'k','linewidth',1.5)
hold on
plot(b1Vec,100-choiceVec,'r','linewidth',1.5)
xlabel("bias1")
ylabel("percentage of winning")
title("percentage of winning vs bias1")
legend("Racer 1","Racer 2")


%% Question 9
clear all; close all; clc;
rng shuffle

th1 = 5;
th2 = 5;
sigma1 = 1;
sigma2 = 1;
b1 = 0.1;
b2 = 0.1;
dt = 0.1;
trlNum = 500;
TVec = linspace(10,100,20);

TReach = [];
for T = TVec
    RTVec = [];
    for trl = 1:trlNum
        trl
        [RT,~] = race_trial(th1,th2,sigma1,sigma2,b1,b2,dt,T);
        RTVec(trl) = [RT];
    end
    TReach =  [TReach, length(find(RTVec >= T))];
end
TReach = TReach / trlNum;

plot(TVec,TReach,'k','linewidth',1.5)
title("Percentage of Reaching Max Time vs Max Time")
xlabel("Max Time(s)")




%% Part 2 =================================================================

%% Question 1 
clear all; close all; clc;
rng shuffle

mtP = [0.1;0.05];
lipW = [0.1;-0.15];
lipTh = 50;
[time,mt1,mt2,lip,RT] = lip_activity(mtP,lipW,lipTh);

w = 0.2;
ind = find(lip==1);
tTmp = time(ind);
tmp = lip(ind);
p1 = plot([tTmp;tTmp],[1.5+tmp+w;1.5+tmp-w],'k');

ind = find(mt2==1);
tTmp = time(ind);
tmp = mt2(ind);
hold on
p2 = plot([tTmp;tTmp],[1+tmp+w;1+tmp-w],'r');

ind = find(mt1==1);
tTmp = time(ind);
tmp = mt1(ind);
hold on
p3 = plot([tTmp;tTmp],[0.5+tmp+w;0.5+tmp-w],'b','DisplayName','yo');

ylim([1 3.1])
legend([p1(1),p2(1),p3(1)],"LIP","MT Inh","MT Exc")
title("Raster plot of MT and LIP neurons")
xlabel("time(s)")

%% Question 2
clear all; close all; clc;
rng shuffle

% Preferred Orientations of MT neurons
theta = 0:0.01:pi;
act1 = sin(theta+pi/4).^10;
act2 = sin(theta-pi/4).^10;
plot(theta,act1,'k','linewidth',1.5)
hold on
plot(theta,act2,'r','linewidth',1.5)
legend("Neuron 1","Neuron 2")
xlim([0,pi])
title("Preferred Orientations of MT neurons")
ylabel("Probablity of firing")
xlabel("stimulus")

% simulation
act1 = @(x) sin(x+pi/4)^10;
act2 = @(y) sin(y-pi/4)^10;
dt = 0.001;
T = 0.5;
L = T/dt;
lipW = [0.1;-0.1];
mtTheta = [[ones(1,200)*0.2,ones(1,50)*0.8,ones(1,250)*0.2] ; ones(1,L)*1.85];
[time,mt1,mt2,lip1,lip2] = lip_activity2(mtTheta,act1,act2,lipW,T);


w = 0.2;
ind = find(lip1==1);
tTmp = time(ind);
tmp = lip1(ind);
p1 = plot([tTmp;tTmp],[2+tmp+w;2+tmp-w],'k');

ind = find(lip2==1);
tTmp = time(ind);
tmp = lip2(ind);
hold on
p2 = plot([tTmp;tTmp],[1.5+tmp+w;1.5+tmp-w],'k');

ind = find(mt1==1);
tTmp = time(ind);
tmp = mt1(ind);
hold on
p3 = plot([tTmp;tTmp],[1+tmp+w;1+tmp-w],'r');

ind = find(mt2==1);
tTmp = time(ind);
tmp = mt2(ind);
hold on
p4 = plot([tTmp;tTmp],[0.5+tmp+w;0.5+tmp-w],'b','DisplayName','yo');
xlim([0,T])
ylim([1 3.6])
title("Raster plot of MT and LIP neurons")
xlabel("time(s)")
legend([p1(1),p2(1),p3(1),p4(1)],"LIP1 = +MT1 - MT2","LIP2 = -MT1 + MT2","MT1","MT2")





%% Functions ==============================================================
function [time,mt1,mt2,lip1,lip2] = lip_activity2(mtTheta,act1,act2,lipW,T)
    dt = 0.001;
    t = 0;
    N = [0;0];
    mt1 = [];
    mt2 = [];
    lip1 = [];
    lip2 = [];
    time = [];
    cnt = 1;
    lipW1 = lipW;
    lipW2 = flip(lipW);
    for i = 1:round(T/dt)
        time = [time,t];
        
        theta = mtTheta(:,cnt);
        dN = rand(2,1) < [act1(theta(1));act2(theta(2))];
        mt1 = [mt1,dN(1)];
        mt2 = [mt2,dN(2)];
        N = N + dN;
        p_lip1 = sum(N .* lipW1);
        p_lip2 = sum(N .* lipW2);
        lipEvent1 = rand(1)<p_lip1;
        lipEvent2 = rand(1)<p_lip2;
        lip1 = [lip1,lipEvent1];
        lip2 = [lip2,lipEvent2];
        t = t + dt;
        cnt = cnt+1;
    end
end

function [time,mt1,mt2,lip,RT] = lip_activity(mtP,lipW,lipTh)
    dt = 0.001;
    t = 0;
    N = [0;0];
    rate = 0;
    mt1 = [];
    mt2 = [];
    lip = [];
    time = [];
    lipT = [];
    M = 100;
    while rate < lipTh
        time = [time,t];
        dN = rand(2,1) < mtP;
        mt1 = [mt1,dN(1)];
        mt2 = [mt2,dN(2)];
        N = N + dN;
        p_lip = sum(N .* lipW);
        lipEvent = rand()<p_lip;
        lip = [lip,lipEvent];
        if (lipEvent == 1)
            lipT = [lipT,t];
        end
        
        % Check lip neuron mean rate for last M spikes
        if (length(lipT) >= M)
            rate = M / (t-lipT(end-M+1));
        end
        t = t + dt
    end
    RT = t;
end

function [RT,choice] = race_trial(th1,th2,sigma1,sigma2,b1,b2,dt,T)
    x1 = 0;
    x2 = 0;
    t = 0;
    flag = 1;
    while (x1 < th1 && x2 < th2 && flag)
        x1 = x1 + b1 * dt + sigma1 * normrnd(0,sqrt(dt));
        x2 = x2 + b2 * dt + sigma2 * normrnd(0,sqrt(dt));
        t = t+dt;
        if (t >= T)
            flag = 0;
        end
    end
    RT = t;
    if (x1 >= th1)
        choice = 1;
    else
        choice = 2;
    end
    
    if (t >= T)
        if (abs(x1-th1)<abs(x2-th2))
            choice = 1;
        else
            choice = 2;
        end
    end
    
end

function [RT,choice] = two_choice_trial(posTh,negTh,sigma,x0,bias,dt)
    x = x0;
    t = 0;
    while (x <= posTh && x >= negTh)
        x = x + bias * dt + sigma * normrnd(0,sqrt(dt));
        t = t+dt;
    end
    RT = t;
    if (x >= posTh)
        choice = 1;
    else
        choice = -1;
    end
end

function choice = simple_model2(bias,sigma,start,T)
    mu = start + bias*T;
    std = sqrt(sigma*T);
    p = cdf('normal',0,mu,std);
    num = rand();
    if (num>p)
        choice = 1;
    else
        choice = -1;
    end
end

function [t,x,choice] = simple_model(bias,sigma,dt,T)
    len = round(T/dt);
    t = linspace(0,T,len);
    x = zeros(1,len);
    for i = 2:len
        x(i) = x(i-1) + bias * dt + sigma * normrnd(0,sqrt(dt));
    end
    choice = sign(x(end)); 
end

