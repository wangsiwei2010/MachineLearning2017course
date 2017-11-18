function [ w ] = QPsvm( x,y )
%QPSVM �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[m, n] = size(x);
x = [x,ones(m, 1)];

H = zeros(n+1, n+1);
H(1:n, 1:n) = eye(n);
f = zeros(1, n+1);
A = -repmat(y, 1, n+1).*x;
b = -ones(m, 1);
opt = optimset('Algorithm','active-set');
[w,fval,exitflag,output,lambda] = quadprog(H,f,A,b,[],[],[],[],[],opt);
end

