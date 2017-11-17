function tcsvmQP_demo
clc
clear all
close all

%% generate data
nsamples = 200;
% training data
[x, y] = tcdataGenerator(nsamples);
% testing data
[xt, yt] = tcdataGenerator(nsamples);
[m n] = size(x);


%% Quadratic Programming Solver
w = tcsvmQP(x, y);

%% Visualize Results
xmin = min(x(:))-1;
xmax = max(x(:))+1;
data_pos = x(find(y==1),:);
data_neg = x(find(y==-1),:);

scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
hold on
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight

margin = xmin:0.1:xmax;
plot(margin, (-w(3)-margin*w(1))/w(2), 'r', 'LineWidth', 2);
plot(margin, (1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
plot(margin, (-1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);