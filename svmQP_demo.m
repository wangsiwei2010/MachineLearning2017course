function tcsvmQP_demo
clc
clear all
close all

%% generate data

iris_data = load('iris_dataset.mat')
x = iris_data.irisInputs'
x = x(:,1:2)
y = iris_data.irisTargets'
y = y(:,1)
y(y==0) = -1;
%% Quadratic Programming Solver
w = QPsvm(x, y);

%% Visualize Results
xmin = min(x(:))-1;
xmax = max(x(:))+1;
data_pos = x(find(y==1),:);
data_neg = x(find(y==-1),:);

scatter(data_pos(:, 1), data_pos(:, 2), 'b*', 'SizeData', 2, 'LineWidth', 0.1);
hold on
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 2, 'LineWidth', 0.1);
axis tight

margin = xmin:0.1:xmax;
plot(margin, (-w(3)-margin*w(1))/w(2), 'r', 'LineWidth', 2);
plot(margin, (1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
plot(margin, (-1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);