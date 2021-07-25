%% Advance Neuro HW06 - Ali Ghavampour - 97102203
clear all; close all; clc;


% initialize
r = zeros(15,15); % reward matrix
ir = 10; jr = 10;
ip = 8; jp = 5;

r(ir,jr) = 10;
r(ip,jp) = -10;

% actions: 1:Right, 2:Bottom, 3:Left, 4:Up
Q = zeros(15*15,4);
Q = init_Q();

% Simulation
trialNum = 100;
etha = 0.5;
gamma = 0.8;
dir = [[1 0];[0 -1];[-1 0];[0 1]];
endPoint = [];
posHolder = {};
QHolder = {};
for trl = 1:trialNum 
    disp(sprintf("trial %d",trl))
    s = rng;
    i = randi(15);
    j = randi(15);
    [i,j] = check_ij(i,j,ir,ip,jr,jp);
    % Fixed starting point here!!!!!!!!!!!!! ===============
    i = 3;
    j = 7;
    is = i;
    js = j;
    
    flag = 0;
    posVec = [i;j];
    iLast = 0;
    jLast = 0;
    time = 1;
    QTime = {};
    while(flag ~= 1)
        % Choosing action -----------------------
        ind = index(i,j);
        availActions = find(~isnan(Q(ind,:)));
        
        % deterministic policy:
%         maxA = max(Q(ind,availActions));
%         action = find(Q(ind,:) == maxA);
%         if (length(action)>1)
%             s = rng;
%             rndInd = randi(length(action));
%             action = action(rndInd);
%         end

        % Softmax policy:
        T = 0.05;
        action = softmax(Q,availActions,ind,T);
        
        % Performing the action
        newPos = [i,j] + dir(action,:);
        iLast = i;
        jLast = j;
        i = newPos(1);
        j = newPos(2);
        posVec = [posVec,[i;j]];
        
        % Updating Q
        ind = index(i,j);
        ind2 = index(iLast,jLast);
        delta = r(i,j) + gamma*max(Q(ind,:)) - Q(ind2,action);
        Q(ind2,action) = Q(ind2,action) + etha * delta;
        QTime{time} = Q;
        
        flag = isequal([i,j],[ir,jr]) | isequal([i,j],[ip,jp]);
        time = time + 1;
    end
    endPoint = [endPoint,[i;j]];
    posHolder{trl} = posVec;
    QHolder{trl} = QTime;
end

cnt = 0;
for i = 1:length(endPoint)
    tmp = endPoint(:,i);
    if isequal(tmp,[ir;jr])
        cnt = cnt + 1;
    end
end
disp(sprintf("reward 1 occurance = %d (%.2f percent)",cnt,cnt/length(endPoint)*100))
disp(sprintf("reward 2 occurance = %d (%.2f percent)",length(endPoint)-cnt,100-cnt/length(endPoint)*100))

% checking if the value is learned
for i = 1:trialNum-3
    z = i;
    pos1 = posHolder{i};
    pos2 = posHolder{i+1};
    pos3 = posHolder{i+2};
    pos4 = posHolder{i+3};
    % 3 of 4 are equal
    cond1 = isequal(pos1,pos2,pos3) || isequal(pos1,pos2,pos4) || isequal(pos2,pos3,pos4);
    
    % distnace condition
    dist = (ir-is) + (jr-js);
    th = 3;
    cond2 = (size(pos1,2) <= dist+th) && (size(pos2,2) <= dist+th) ...
        && (size(pos3,2) <= dist+th) && (size(pos4,2) <= dist+th);
    
    if (cond1 || cond2)
        break
    end
end

% Question 1 - plot before and after training
n = 20;
trial = round(linspace(1,trialNum,n));
% trial = 180:199;
for i = 1:n
    trl = trial(i);
    subplot(4,5,i)
    posVec = posHolder{trl};
    plot(posVec(1,:),posVec(2,:),'k')
    hold on;
    scatter(ir,jr,'k','filled')
    hold on
    scatter(ip,jp,'r','filled')
    hold on
    scatter(posVec(1,1),posVec(2,1),'y','filled')
    xlim([1,15])
    ylim([1,15])
    title(sprintf("Trial Number %d",trl))
end

val = max(Q');
val = reshape(val,[15 15]);
val(ir,jr) = 5;
% val(ip,jp) = -1;
xb = 1:15;
yb = 1:15;
figure;
subplot(1,2,1)
colormap(jet)
contourf(xb,yb,log10(val+0.02)),colorbar
axis square
hold on
scatter(ir,jr,20,'k','filled')
hold on
scatter(ip,jp,20,'r','filled')
title("contour plot of learned Q")

subplot(1,2,2)
colormap(jet)
pcolor(log10(val+0.05)),colorbar
axis square
hold on
scatter(ir,jr,20,'k','filled')
hold on
scatter(ip,jp,20,'r','filled')
title("Log of learned Q")

[fx,fy] = gradient(val);
fx(ir,jr) = 0;
fy(ir,jr) = 0;
figure;
q = quiver(xb,yb,fx,fy,'k','AutoScaleFactor',0.6);
% % q.ShowArrowHead = 'off';
% q.AutoScale = 'on';
hold on
scatter(ir,jr,20,'k','filled')
hold on
scatter(ip,jp,20,'r','filled')
xlim([1 15])
ylim([1 15])


%% Animation - RUN THE LAST SECTION BEFORE THIS SECTION!!!! ===============
clc; close all;
flag = 0;
h = figure('Position',[500 150 700 600]);
xb = 1:15;
yb = 1:15;
v = VideoWriter('myVideo.avi');
open(v)
for trl = [24,26,28,29,35,40,60,70,100]
    if ~ishghandle(h)
        break
    end
    if (flag == 1)
        break
    end
    tmpPath = posHolder{trl};
    x = tmpPath(1,:);
    y = tmpPath(2,:);
    for i = 1:length(tmpPath)-1
        if ~ishghandle(h)
            break
            flag = 1;
        end
        
        % Path
        scatter(ir,jr,50,'k','filled')
        hold on
        scatter(ip,jp,50,'r','filled')
        hold on
        scatter(x(1),y(1),50,'y','filled')
        hold on
        plot(x,y,'color',[.7 .7 .7])
        hold on
        quiver(x(i),y(i),(x(i+1)-x(i)),(y(i+1)-y(i)),'k','linewidth',1.5,'MaxHeadSize',0.5)
        hold off
        xlim([1 15])
        ylim([1 15])
        title(sprintf("Trial %d",trl))
%         pause(0.00001)
        frame = getframe(gcf);
        for i = 1:5
            writeVideo(v,frame);
        end
    end
end
close(v)

%% Animation - Gradient and Value =========================================
clc; close all;
flag = 0;
h = figure('Position',[0 0 2160 920]);
xb = 1:15;
yb = 1:15;
colormap(jet);
v = VideoWriter('myVideo2.avi');
open(v)
for trl = 1:50
    if ~ishghandle(h)
        break
    end
    if (flag == 1)
        break
    end
    QTmp = QHolder{trl};
    QTmp = QTmp{1};
    
    % Contour
    val = max(QTmp');
    val = reshape(val,[15 15]);
    val(ir,jr) = 5;
    subplot(1,2,1)
    contourf(xb,yb,log10(val+0.02)),colorbar
    axis square
    hold on
    scatter(ir,jr,20,'k','filled')
    hold on
    scatter(ip,jp,20,'r','filled')
    title(sprintf("contour plot of learned Q - Trial = %d",trl))
    
    % Gradient
    subplot(1,2,2)
    [fx,fy] = gradient(val);
    fx(ir,jr) = 0;
    fy(ir,jr) = 0;
    q = quiver(xb,yb,fx,fy,'k','AutoScaleFactor',0.6);
    hold on
    scatter(ir,jr,20,'k','filled')
    hold on
    scatter(ip,jp,20,'r','filled')
    xlim([1 15])
    ylim([1 15])
    title(sprintf("Gradient - Trial = %d",trl))
    hold off

    frame = getframe(gcf);
    for i = 1:5
        writeVideo(v,frame);
    end
end
close(v)


%% Question03 - effect of etha and gamma on learning
clear all; close all; clc;

r = zeros(15,15); % reward matrix
ir = 10; jr = 10;
r(ir,jr) = 10;
ip = 8; jp = 5;
r(ip,jp) = 2;
trialNum = 200;

% actions: 1:Right, 2:Bottom, 3:Left, 4:Up
dir = [[1 0];[0 -1];[-1 0];[0 1]];

gammaVec = 0.5;
ethaVec = linspace(0.1,0.95,10);
trlVec = [];
itNum = 2;
for it = 1:itNum
    disp(sprintf("iteration %d",it))
    trlVecTmp = [];
    for etha = ethaVec
        for gamma = gammaVec
            disp(sprintf("gamma = %.4f , etha = %.4f",gamma,etha))
            Q = zeros(15*15,4);
            Q = init_Q();
            posHolder = {};
            trl = 1;
            for trl = 1:trialNum
                i = 5;
                j = 8;
                is = i;
                js = j;
                flag = 0;
                posVec = [i;j];
                iLast = 0;
                jLast = 0;
                while(flag ~= 1)
                    % Choosing action
                    ind = index(i,j);
                    availActions = find(~isnan(Q(ind,:)));
                    % deterministic policy:
%                     maxA = max(Q(ind,availActions));
%                     action = find(Q(ind,:) == maxA);
%                     if (length(action)>1)
%                         s = rng;
%                         rndInd = randi(length(action));
%                         action = action(rndInd);
%                     end

                    % Softmax policy:
                    T = 0.05;
                    action = softmax(Q,availActions,ind,T);

                    % Performing the action
                    newPos = [i,j] + dir(action,:);
                    iLast = i;
                    jLast = j;
                    i = newPos(1);
                    j = newPos(2);
                    posVec = [posVec,[i;j]];

                    % Updating Q
                    ind = index(i,j);
                    ind2 = index(iLast,jLast);
                    delta = r(i,j) + gamma*max(Q(ind,:)) - Q(ind2,action);
                    Q(ind2,action) = Q(ind2,action) + etha * delta;

                    flag = isequal([i,j],[ir,jr]) | isequal([i,j],[ip,jp]);
                end
                posHolder{trl} = posVec;
            end
            % checking if the value is learned
            for i5 = 1:trialNum-3
                z = i5;
                pos1 = posHolder{i5};
                pos2 = posHolder{i5+1};
                pos3 = posHolder{i5+2};
                pos4 = posHolder{i5+3};
                % 3 of 4 are equal
                cond1 = isequal(pos1,pos2,pos3) || isequal(pos1,pos2,pos4) ...
                    || isequal(pos2,pos3,pos4);
                % distnace condition
                dist = (ir-is) + (jr-js);
                th = 3;
                cond2 = (size(pos1,2) <= dist+th) && (size(pos2,2) <= dist+th) ...
                    && (size(pos3,2) <= dist+th) && (size(pos4,2) <= dist+th);
                if (cond1 || cond2)
                    break
                end
            end
            trlVecTmp = [trlVecTmp,z];
        end
    end
    trlVec = [trlVec;trlVecTmp];
end

% use this this section for the set of parameters:
% gammaVec = 0.5;
% ethaVec = linspace(0.1,0.95,10); 
% itNum = arbitraty;
% UNCOMMENT FROM HERE!!!!!!!!!
x = mean(trlVec,1);
err = std(trlVec)/sqrt(itNum);
errorbar(ethaVec,x,err,'k','LineWidth',1.5)
title(sprintf("effect of learning rate - gamma = %.4f",gamma))
ylabel("average trials sufficient to learn")
xlabel("learning rate")


% use this this section for the set of parameters:
% gammaVec = linspace(0.1,0.95,10);
% ethaVec = 0.5; 
% itNum = arbitraty;
% UNCOMMENT FROM HERE!!!!!!!!!
% x = mean(trlVec,1);
% err = std(trlVec)/sqrt(itNum);
% errorbar(gammaVec,x,err,'k','LineWidth',1.5)
% title(sprintf("effect of gamma - learning rate = %.4f",etha))
% ylabel("average trials sufficient to learn")
% xlabel("gamma")

% use this this section for the set of parameters:
% gammaVec = linspace(0.25,0.95,5);
% ethaVec = linspace(0.2,0.95,5); 
% itNum = arbitrary;
% UNCOMMENT FROM HERE!!!!!!!!!
% x = mean(trlVec,1);
% x = reshape(x,[length(ethaVec),length(gammaVec)]);
% colormap(jet)
% imagesc(gammaVec,ethaVec,log10(x)), colorbar
% title(sprintf("effect of gamma and learning rate"))
% ylabel("learning rate")
% xlabel("gamma")




%% Question 4 =============================================================
clear all; close all; clc;

% initialize
r = zeros(15,15); % reward matrix
ir = 10; jr = 10;
ip = 8; jp = 5;

r(ir,jr) = 10;
r(ip,jp) = 1;

% actions: 1:Right, 2:Bottom, 3:Left, 4:Up

% Simulation
ethaVec = linspace(0.1,0.9,10);
gamma = 0.5;
% gammaVec = linspace(0.05,0.95,10);
dir = [[1 0];[0 -1];[-1 0];[0 1]];
prob = [];
for etha = ethaVec
    trialNum = 250;
    posHolder = {};
    itNum = 10;
    endPoint = [];
    for it = 1:itNum
        fprintf("iteration %d \n",it)   
        Q = zeros(15*15,4);
        Q = init_Q();
        for trl = 1:trialNum 
            disp(sprintf("trial %d",trl))
            s = rng;
            i = randi(15);
            j = randi(15);
            [i,j] = check_ij(i,j,ir,ip,jr,jp);
            % Fixed starting point here!!!!!!!!!!!!! ===============
            i = 6;
            j = 12;
            is = i;
            js = j;

            flag = 0;
            posVec = [i;j];
            iLast = 0;
            jLast = 0;
            while(flag ~= 1)
                % Choosing action -----------------------
                ind = index(i,j);
                availActions = find(~isnan(Q(ind,:)));

                % deterministic policy:
        %         maxA = max(Q(ind,availActions));
        %         action = find(Q(ind,:) == maxA);
        %         if (length(action)>1)
        %             s = rng;
        %             rndInd = randi(length(action));
        %             action = action(rndInd);
        %         end

                % Softmax policy:
                T = 0.05;
                action = softmax(Q,availActions,ind,T);

                % Performing the action
                newPos = [i,j] + dir(action,:);
                iLast = i;
                jLast = j;
                i = newPos(1);
                j = newPos(2);
                posVec = [posVec,[i;j]];

                % Updating Q
                ind = index(i,j);
                ind2 = index(iLast,jLast);
                delta = r(i,j) + gamma*max(Q(ind,:)) - Q(ind2,action);
                Q(ind2,action) = Q(ind2,action) + etha * delta;

                flag = isequal([i,j],[ir,jr]) | isequal([i,j],[ip,jp]);
            end
            endPoint = [endPoint,[i;j]];
            posHolder{trl} = posVec;
        end
    end

    cnt = 0;
    for i = 1:length(endPoint)
        tmp = endPoint(:,i);
        if isequal(tmp,[ir;jr])
            cnt = cnt + 1;
        end
    end
%     disp(sprintf("reward 1 occurance = %d (%.2f percent)",cnt,cnt/length(endPoint)*100))
%     disp(sprintf("reward 2 occurance = %d (%.2f percent)",length(endPoint)-cnt,100-cnt/length(endPoint)*100))
    prob = [prob,cnt/length(endPoint)*100];
end

% plot(gammaVec,prob,'k','LineWidth',1.5)
plot(ethaVec,prob,'k','LineWidth',1.5)
xlim([0.1,0.9])
title("effect of gamma on choosing reward")
xlabel("gamma")
ylabel("percentage of choosing higher reward")


%% Question 5 - TD Lambda Rule ============================================
clear all; close all; clc;


% initialize
r = zeros(15,15); % reward matrix
ir = 10; jr = 10;
ip = 8; jp = 5;

r(ir,jr) = 10;
r(ip,jp) = -10;

% actions: 1:Right, 2:Bottom, 3:Left, 4:Up
Q = zeros(15*15,4);
Q = init_Q();

% Simulation
trialNum = 200;
etha = 0.5;
gamma = 0.8;
dir = [[1 0];[0 -1];[-1 0];[0 1]];
endPoint = [];
posHolder = {};

lambda = 0.9; % For TD(lambda) rule

for trl = 1:trialNum 
    disp(sprintf("trial %d",trl))
    s = rng;
    i = randi(15);
    j = randi(15);
    [i,j] = check_ij(i,j,ir,ip,jr,jp);
    % Fixed starting point here!!!!!!!!!!!!! ===============
%     i = 2;
%     j = 3;
    is = i;
    js = j;
    
    flag = 0;
    posVec = [i;j];
    iLast = 0;
    jLast = 0;
    iLast2 = 0;
    jLast2 = 0;
    iLast3 = 0;
    jLast3 = 0;
    iLast4 = 0;
    jLast4 = 0;
    
    time = 1;
    while(flag ~= 1)
        % Choosing action -----------------------
        ind = index(i,j);
        availActions = find(~isnan(Q(ind,:)));
        
        % deterministic policy:
%         maxA = max(Q(ind,availActions));
%         action = find(Q(ind,:) == maxA);
%         if (length(action)>1)
%             s = rng;
%             rndInd = randi(length(action));
%             action = action(rndInd);
%         end

        % Softmax policy:
        T = 0.05;
        action = softmax(Q,availActions,ind,T);
        
        % Performing the action
        newPos = [i,j] + dir(action,:);
        
        iLast4 = iLast3;
        jLast4 = jLast3;
        iLast3 = iLast2;
        jLast3 = jLast2;
        iLast2 = iLast;
        jLast2 = jLast;
        iLast = i;
        jLast = j;
        i = newPos(1);
        j = newPos(2);
        posVec = [posVec,[i;j]];
        
        
        % Updating Q
        ind = index(i,j);
        ind2 = index(iLast,jLast);
        ind3 = index(iLast2,jLast2);
        ind4 = index(iLast3,jLast3);
        ind5 = index(iLast4,jLast4);
        delta = r(i,j) + gamma*max(Q(ind,:)) - Q(ind2,action);
        Q(ind2,action) = Q(ind2,action) + etha * delta;
        if (time >= 2 && ind3~=ind2)
            Q(ind3,action) = Q(ind3,action) + etha*lambda*delta;
        end
        if (time >= 3 && ind4~=ind3)
            Q(ind4,action) = Q(ind4,action) + etha*lambda^2*delta;
        end
        if (time >= 4 && ind5~=ind4)
            Q(ind5,action) = Q(ind5,action) + etha*lambda^3*delta;
        end
        
        time = time + 1;
        flag = isequal([i,j],[ir,jr]) | isequal([i,j],[ip,jp]);
    end
    endPoint = [endPoint,[i;j]];
    posHolder{trl} = posVec;
end

cnt = 0;
for i = 1:length(endPoint)
    tmp = endPoint(:,i);
    if isequal(tmp,[ir;jr])
        cnt = cnt + 1;
    end
end
disp(sprintf("reward 1 occurance = %d (%.2f percent)",cnt,cnt/length(endPoint)*100))
disp(sprintf("reward 2 occurance = %d (%.2f percent)",length(endPoint)-cnt,100-cnt/length(endPoint)*100))

% checking if the value is learned
for i = 1:trialNum-3
    z = i;
    pos1 = posHolder{i};
    pos2 = posHolder{i+1};
    pos3 = posHolder{i+2};
    pos4 = posHolder{i+3};
    % 3 of 4 are equal
    cond1 = isequal(pos1,pos2,pos3) || isequal(pos1,pos2,pos4) || isequal(pos2,pos3,pos4);
    
    % distnace condition
    dist = (ir-is) + (jr-js);
    th = 3;
    cond2 = (size(pos1,2) <= dist+th) && (size(pos2,2) <= dist+th) ...
        && (size(pos3,2) <= dist+th) && (size(pos4,2) <= dist+th);
    
    if (cond1 || cond2)
        break
    end
end

% Question 1 - plot before and after training
n = 20;
trial = round(linspace(1,trialNum,n));
% trial = 180:199;
for i = 1:n
    trl = trial(i);
    subplot(4,5,i)
    posVec = posHolder{trl};
    plot(posVec(1,:),posVec(2,:),'k')
    hold on;
    scatter(ir,jr,'k','filled')
    hold on
    scatter(ip,jp,'r','filled')
    hold on
    scatter(posVec(1,1),posVec(2,1),'y','filled')
    xlim([1,15])
    ylim([1,15])
    title(sprintf("Trial Number %d , lambda = %.2f",trl,lambda))
end


val = max(Q');
val = reshape(val,[15 15]);
val(ir,jr) = 5;
% val(ip,jp) = -1;
xb = 1:15;
yb = 1:15;
figure;
subplot(1,2,1)
colormap(jet)
contourf(xb,yb,log10(val+0.02)),colorbar
axis square
hold on
scatter(ir,jr,20,'k','filled')
hold on
scatter(ip,jp,20,'r','filled')
title("contour plot of learned Q - lambda = " + num2str(lambda))

subplot(1,2,2)
colormap(jet)
pcolor(log10(val+0.05)),colorbar
axis square
hold on
scatter(ir,jr,20,'k','filled')
hold on
scatter(ip,jp,20,'r','filled')
title("Log of learned Q - lambda = " + num2str(lambda))

[fx,fy] = gradient(val);
fx(ir,jr) = 0;
fy(ir,jr) = 0;
figure;
q = quiver(xb,yb,fx,fy,'k','AutoScaleFactor',0.6);
% % q.ShowArrowHead = 'off';
% q.AutoScale = 'on';
hold on
scatter(ir,jr,20,'k','filled')
hold on
scatter(ip,jp,20,'r','filled')
xlim([1 15])
ylim([1 15])




%% Question 5 - Effect of lambda ==========================================
clear all; close all; clc;

% initialize
r = zeros(15,15); % reward matrix
ir = 10; jr = 10;
ip = 8; jp = 5;

r(ir,jr) = 10;
r(ip,jp) = -10;

% actions: 1:Right, 2:Bottom, 3:Left, 4:Up


% Simulation
trialNum = 200;
etha = 0.5;
gamma = 0.8;
dir = [[1 0];[0 -1];[-1 0];[0 1]];
endPoint = [];

lambdaVec = linspace(0.1,0.9,10); % For TD(lambda) rule

zVec = [];
for it = 1:2
    fprintf("iteration %d \n",it)
    zVecTmp = [];
    for lambda = lambdaVec
        Q = zeros(15*15,4);
        Q = init_Q();
        posHolder = {};
        for trl = 1:trialNum 
            disp(sprintf("trial %d",trl))
            s = rng;
            i = randi(15);
            j = randi(15);
            [i,j] = check_ij(i,j,ir,ip,jr,jp);
            % Fixed starting point here!!!!!!!!!!!!! ===============
            i = 2;
            j = 3;
            is = i;
            js = j;

            flag = 0;
            posVec = [i;j];
            iLast = 0;
            jLast = 0;
            iLast2 = 0;
            jLast2 = 0;
            iLast3 = 0;
            jLast3 = 0;
            iLast4 = 0;
            jLast4 = 0;

            time = 1;
            while(flag ~= 1)
                % Choosing action -----------------------
                ind = index(i,j);
                availActions = find(~isnan(Q(ind,:)));

                % deterministic policy:
        %         maxA = max(Q(ind,availActions));
        %         action = find(Q(ind,:) == maxA);
        %         if (length(action)>1)
        %             s = rng;
        %             rndInd = randi(length(action));
        %             action = action(rndInd);
        %         end

                % Softmax policy:
                T = 0.05;
                action = softmax(Q,availActions,ind,T);

                % Performing the action
                newPos = [i,j] + dir(action,:);

                iLast4 = iLast3;
                jLast4 = jLast3;
                iLast3 = iLast2;
                jLast3 = jLast2;
                iLast2 = iLast;
                jLast2 = jLast;
                iLast = i;
                jLast = j;
                i = newPos(1);
                j = newPos(2);
                posVec = [posVec,[i;j]];


                % Updating Q
                ind = index(i,j);
                ind2 = index(iLast,jLast);
                ind3 = index(iLast2,jLast2);
                ind4 = index(iLast3,jLast3);
                ind5 = index(iLast4,jLast4);
                delta = r(i,j) + gamma*max(Q(ind,:)) - Q(ind2,action);
                Q(ind2,action) = Q(ind2,action) + etha * delta;
                if (time >= 2 && ind3~=ind2)
                    Q(ind3,action) = Q(ind3,action) + etha*lambda*delta;
                end
                if (time >= 3 && ind4~=ind3)
                    Q(ind4,action) = Q(ind4,action) + etha*lambda^2*delta;
                end
                if (time >= 4 && ind5~=ind4)
                    Q(ind5,action) = Q(ind5,action) + etha*lambda^3*delta;
                end

                time = time + 1;
                flag = isequal([i,j],[ir,jr]) | isequal([i,j],[ip,jp]);
            end
            endPoint = [endPoint,[i;j]];
            posHolder{trl} = posVec;
        end

        for i = 1:trialNum-3
            z = i;
            pos1 = posHolder{i};
            pos2 = posHolder{i+1};
            pos3 = posHolder{i+2};
            pos4 = posHolder{i+3};
            % 3 of 4 are equal
            cond1 = isequal(pos1,pos2,pos3) || isequal(pos1,pos2,pos4) || isequal(pos2,pos3,pos4);

            % distnace condition
            dist = (ir-is) + (jr-js);
            th = 3;
            cond2 = (size(pos1,2) <= dist+th) && (size(pos2,2) <= dist+th) ...
                && (size(pos3,2) <= dist+th) && (size(pos4,2) <= dist+th);

            if (cond1 || cond2)
                break
            end
        end
        zVecTmp = [zVecTmp,z];
    end
    zVec = [zVec;zVecTmp];
end

cnt = 0;
for i = 1:length(endPoint)
    tmp = endPoint(:,i);
    if isequal(tmp,[ir;jr])
        cnt = cnt + 1;
    end
end
disp(sprintf("reward 1 occurance = %d (%.2f percent)",cnt,cnt/length(endPoint)*100))
disp(sprintf("reward 2 occurance = %d (%.2f percent)",length(endPoint)-cnt,100-cnt/length(endPoint)*100))

%%
plot(lambdaVec,mean(zVec,1),'k','LineWidth',1.5)
title("Effect of lambda on learning - gamma = 0.8, eta = 0.5")
ylabel("trials neede to learn")
xlabel("lambda")


%% Functions
function action = softmax(Q,availActions,ind,T)
    if length(availActions) == 2
        ind1 = availActions(1);
        ind2 = availActions(2);
        denum = exp(Q(ind,ind1)/T) + exp(Q(ind,ind2)/T);
        p1 = exp(Q(ind,ind1)/T) / denum;
        s = rng;
        if (rand() <= p1)
            action = ind1;
        else
            action = ind2;
        end
    elseif length(availActions) == 3
        ind1 = availActions(1);
        ind2 = availActions(2);
        ind3 = availActions(3);
        denum = exp(Q(ind,ind1)/T) + exp(Q(ind,ind2)/T) + exp(Q(ind,ind3)/T);
        p1 = exp(Q(ind,ind1)/T) / denum;
        p2 = exp(Q(ind,ind2)/T) / denum;
        s = rng;
        x = rand();
        if (x <= p1)
            action = ind1;
        elseif (x > p1 && x <= p1+p2)
            action = ind2;
        else
            action = ind3;
        end
    elseif length(availActions) == 4
        ind1 = availActions(1);
        ind2 = availActions(2);
        ind3 = availActions(3);
        ind4 = availActions(4);
        denum = exp(Q(ind,ind1)/T) + exp(Q(ind,ind2)/T) + exp(Q(ind,ind3)/T)...
            + exp(Q(ind,ind4)/T);
        p1 = exp(Q(ind,ind1)/T) / denum;
        p2 = exp(Q(ind,ind2)/T) / denum;
        p3 = exp(Q(ind,ind3)/T) / denum;
        s = rng;
        x = rand();
        if (x <= p1)
            action = ind1;
        elseif (x > p1 && x <= p1+p2)
            action = ind2;
        elseif (x > p1+p2 && x <= p1+p2+p3)
            action = ind3;
        else
            action = ind4;
        end
    end
end

function ind = index(i,j)
    ind = (i-1)*15 + j;
end

function [i_new,j_new] = actor(v,sap,i,j,iLast,jLast)
    val = v(i,j);
    ind = (i-1)*15 + j; 
    dir = [[1,0]; [0,-1]; [-1,0]; [0,1]];
    p = sap(ind,:);
    x1 = [iLast-i, jLast-j, i-iLast, j-jLast];
    p(find(x1)==1) = 0;
    pMax = max(p);
    indx = find(p == pMax);
    if (length(indx) > 1)
        s = rng;
        x = randi(length(indx));
        indx = indx(x);
    end
    tmpInd = [i,j] + dir(indx,:);
    i_new = tmpInd(1);
    j_new = tmpInd(2);
end

function v_new = value(v,sap,r)
    dir = [[1,0]; [0,-1]; [-1,0]; [0,1]];
    [ir,jr] = find(r==1);
    [ip,jp] = find(r==-1);
    v_new = zeros(size(v));
    v_new(ir,jr) = 1;
    v_new(ip,jp) = -1;
    for i = 1:size(v,1)
        for j = 1:size(v,2)
            if (~(isequal([i,j],[ir,jr]) || isequal([i,j],[ip,jp])))
                ind = (i-1)*15 + j;
                tmpInd = find(sap(ind,:));
                tmp = 0;
                for i1 = 1:length(tmpInd)
                    pTmp = sap(ind,tmpInd(i1));
                    ijTmp = [i,j] + dir(tmpInd(i1),:);
                    vTmp = v(ijTmp(1),ijTmp(2));
                    tmp = tmp + pTmp*vTmp;
                end
                v_new(i,j) = tmp;
            end
        end
    end
end

function [i,j] = check_ij(i,j,ir,ip,jr,jp)
    if (i==ir || i==ip)
        i = i-2;
    end
    if (j==jr || j==jp)
        j = j-2;
    end
end

function Q = init_Q()
    sap = zeros(15*15,4); % state action probability
    for i = 1:15
        for j = 1:15
            ind = (i-1)*15 + j;
            if (isequal([i,j],[1,1]))
                sap(ind,3) = nan;
                sap(ind,2) = nan;
            end

            if (isequal([i,j],[1,15]))
                sap(ind,3) = nan;
                sap(ind,4) = nan;
            end

            if (isequal([i,j],[15,1]))
                sap(ind,1) = nan;
                sap(ind,2) = nan;
            end

            if (isequal([i,j],[15,15]))
                sap(ind,1) = nan;
                sap(ind,4) = nan;
            end

            if (i>=2 && i<=14 && j == 1)
                sap(ind,2) = nan;
            end

            if (i>=2 && i<=14 && j == 15)
                sap(ind,4) = nan;
            end

            if (j>=2 && j<=14 && i == 1)
                sap(ind,3) = nan;
            end

            if (j>=2 && j<=14 && i == 15)
                sap(ind,1) = nan;
            end
        end
    end
    Q = sap;
end







