clear all
clf
close all

rng('default')
u1 = [0 0]';
u2 = [2 2]';
covMat = [1 0.25;0.25 1];
%1.1
x_1 = mvnrnd(u1,covMat,500)';
x_2 = mvnrnd(u2,covMat,500)';

X = cat(2,x_1,x_2);
y = cat(2,ones(1,500),2*ones(1,500));

%colormap flag;
gscatter(X(1,:),X(2,:),y,'rb',[],15);
legend('class1','class2');      

%1.2
for i = 1:1000
    p_1(i) = (1/(2*pi*sqrt(det(covMat))))*exp(-0.5*(X(:,i)-u1)'*inv(covMat)*(X(:,i)-u1));
	p_2(i) = (1/(2*pi*sqrt(det(covMat))))*exp(-0.5*(X(:,i)-u2)'*inv(covMat)*(X(:,i)-u2));
    
    if p_1(i) > p_2(i)
        pred(i)=1;
    else
        pred(i)=2;
    end
end
figure

gscatter(X(1,:),X(2,:),pred,'rb',[],15);
legend('class1','class2');

%1.3
p_error = sum(pred ~= y)/1000;
fprintf("error classification probability = %f\n",p_error)

%1.4
L=[0 1; .005 0];
for i = 1:1000
    if L(1,2)*p_1(i) > L(2,1)*p_2(i)
        pred_loss(i)=1;
    else
        pred_loss(i)=2;
    end
end

figure
gscatter(X(1,:),X(2,:),pred_loss,'rb',[],15);
legend('class1','class2');

%1.5
ar = (pred_loss ~= y) .* cat(2,L(1,2)*ones(1,500),L(2,1)*ones(1,500));
ar = sum(ar)/1000;
fprintf("average risk = %f\n",ar)