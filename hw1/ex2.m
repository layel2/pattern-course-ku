clear all
clf
close all

u1 = [0 2]';
u2 = [0 0]';
%covMat1 = [4 1.8;1.8 1];
%covMat2 = [4 1.8;1.8 1];

%covMat1 = [4 0;0 1];
%covMat2 = [4 0;0 1];
n1 = 5000;
n2 = 500;
n = n1+n2;
for cases = 1:2
    rng('default')
    if cases==1
        covMat1 = [4 1.8;1.8 1];
        covMat2 = [4 1.8;1.8 1];
    else
        covMat1 = [4 0;0 1];
        covMat2 = [4 0;0 1];
    end
    %1.1
    x_1 = mvnrnd(u1,covMat1,n1)';
    x_2 = mvnrnd(u2,covMat2,n2)';

    X = cat(2,x_1,x_2);
    y = cat(2,ones(1,n1),2*ones(1,n2));
    
    figure
    colormap flag;
    gscatter(X(1,:),X(2,:),y,'rb',[],10);

    %1.2
    for i = 1:n
        p_1(i) = (1/(2*pi*sqrt(det(covMat1))))*exp(-0.5*(X(:,i)-u1)'*inv(covMat1)*(X(:,i)-u1));
        p_2(i) = (1/(2*pi*sqrt(det(covMat2))))*exp(-0.5*(X(:,i)-u2)'*inv(covMat2)*(X(:,i)-u2));

        if (n1/n)*p_1(i) > (n2/n)*p_2(i)
            pred(i)=1;
        else
            pred(i)=2;
        end
    end
    figure

    colormap flag;
    gscatter(X(1,:),X(2,:),pred,'rb',[],10);

    %1.3
    p_error = sum(pred ~= y)/n;
    fprintf("(%d) error classification probability = %f\n",cases,p_error)

    %1.4
    for i = 1:n
        p_naive1(i) = (1/sqrt(2*pi*covMat1(1,1))*exp(-(X(1,i)-u1(1)).^2/(2*covMat1(1,1)))) * (1/sqrt(2*pi*covMat1(2,2))*exp(-(X(2,i)-u1(2)).^2/(2*covMat1(2,2))));
        p_naive2(i) = (1/sqrt(2*pi*covMat2(1,1))*exp(-(X(1,i)-u2(1)).^2/(2*covMat2(1,1)))) * (1/sqrt(2*pi*covMat2(2,2))*exp(-(X(2,i)-u2(2)).^2/(2*covMat2(2,2))));

        if (n1/n)*p_naive1(i) > (n2/n)*p_naive2(i)
            pred_naive(i)=1;
        else
            pred_naive(i)=2;
        end
    end
    figure
    colormap flag;
    gscatter(X(1,:),X(2,:),pred_naive,'rb',[],10);

    %1.5
    p_error_navie = sum(pred_naive ~= y)/n;
    fprintf("(%d) error classification probability(naive bayes) = %f\n",cases,p_error_navie)
end