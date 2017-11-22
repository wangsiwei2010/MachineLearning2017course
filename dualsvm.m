function svm = dualsvm(x,y,c)
    n = length(Y);
    H = (Y'*Y).*(X'*X);
    f = -ones(n,1); %f为1*n个-1,f相当于Quadprog函数中的c
    A = [];
    b = [];
    Aeq = Y; %相当于Quadprog函数中的A1,b1
    beq = 0;
    lb = zeros(n,1); %相当于Quadprog函数中的LB，UB
    ub = c*ones(n,1);
    a0 = zeros(n,1);  % a0是解的初始近似值
    [a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0);
    
    epsilon = 1e-8;                     
    sv_label = find(abs(a)>epsilon);  %0<a<a(max)则认为x为支持向量
    svm.w = (a.*y)*x;
    svm.a = a(sv_label);
    svm.Xsv = X(:,sv_label);
    svm.Ysv = Y(sv_label);
    svm.svnum = length(sv_label);
   %svm.label = sv_label;
end