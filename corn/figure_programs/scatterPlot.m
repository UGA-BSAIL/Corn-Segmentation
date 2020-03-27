x = [];
pred = zeros(1,50); 
conf_gt = strings(1,25);
conf_pred = strings(1,25);
% -1 for left, 0 for equal, 1 for right
files = {'C:\Users\shrin\Documents\GitHub\Corn_Detection\logs\cons_est\300\300_1.mat', 'C:\Users\shrin\Documents\GitHub\Corn_Detection\logs\cons_est\300\300_2.mat', 'C:\Users\shrin\Documents\GitHub\Corn_Detection\logs\cons_est\300\300_3.mat', 'C:\Users\shrin\Documents\GitHub\Corn_Detection\logs\cons_est\300\300_4.mat'};
for i  = 1 : 4
    %filename = [];
    filename = strcat(strcat('C:\Users\shrin\Documents\GitHub\Corn_Detection\logs\cons_est\300\300_', int2str(i)),'.mat');
    %filename = string(files(i));
    load(filename);
    disp(filename);
    disp(GT_Left(1,1));
    disp(Pred_Left(1,1));
    Pred_Left(Pred_Left>100) = 100;
    GT_Left(GT_Left>100) = 100;
    Pred_Right(Pred_Right>100) = 100;
    GT_Right(GT_Right>100) = 100;
    
    
    x = [GT_Left GT_Right];
    %disp(x);
    
   
    for j = 1:25
        if  GT_Left(1,j) < GT_Right(1,j) 
            conf_gt(1,j) = 'Left';
   
        else
            if GT_Left(1,j) > GT_Right(1,j)
            conf_gt(1,j) = 'Right';
            else
                conf_gt(1,j) = 'Equal';
       
            end
        end
         if  Pred_Left(1,j) < Pred_Right(1,j) 
            conf_pred(1,j) = 'Left';
   
        else
            if Pred_Left(1,j) > Pred_Right(1,j)
            conf_pred(1,j) = 'Right';
            else
                conf_pred(1,j) = 'Equal';
            end
        end
    
    end
    %Ch = confusionchart(conf_gt,conf_pred,'Order',{'Left','Equal','Right'});
    [C,order] = confusionmat(conf_gt,conf_pred, 'Order',{'Left','Equal','Right'});
    
pred(i,:) = [Pred_Left Pred_Right];
end
x(x>100) = 100;
pred(pred>100) = 100;
%scatter(x,median(pred), 15,'b', 'filled');
%xlab = xlabel('Ground Truth Consumption');
%ylab = ylabel('Average Predicted Consumption by 5 models');
%tit = title('(e) No. of training images = 250');

% xlab.FontWeight = 'bold';
% ylab.FontWeight = 'bold';
% xlim([-2 105]);
% ylim([-2 105]);

mpred = mean(pred);
medpred = median(pred);
%slope = x(:)\medpred(:);

%coeff = polyfit(x, median(pred), 1);

difference = x - medpred;
xout = [];
predout = [];
xnew = [];
prednew = [];
for i = 1 : 50
    
%     if(difference(i) < 0)
%         for col = 1:5
%             if (pred(col,i) < medpred(i)) && abs(x(i) - pred(col,i)) < abs(difference(i))
%                 medpred(i) = pred(col,i);
%                 difference(i) = x(i) - pred(col,i);
%                         
%             end
%         end
%     end
%     if(difference(i) > 0)
%         for col = 1:5
%             if (pred(col,i) > medpred(i)) && abs(x(i) - pred(col,i)) < abs(difference(i))
%                 medpred(i) = pred(col,i);
%                 difference(i) = x(i) - pred(col,i);
%                         
%             end
%         end
%     end
    
    if(abs(difference(i)) > 15)
        xout = [xout, x(i)];
        predout = [predout, medpred(i)];
    else
        xnew = [xnew, x(i)];
        prednew = [prednew, medpred(i)];
    end
end

slopenew = xnew(:)\prednew(:);

scatter(xnew,prednew, 20,'b', 'filled');
xlab = xlabel('Ground Truth Consumption(%)');
ylab = ylabel('Average Predicted Consumption(%) by 5 models');
tit = title('(d) No. of training images = 200');

xlab.FontWeight = 'bold';
ylab.FontWeight = 'bold';
xlim([-2 105]);
ylim([-2 105]);

hold on;
scatter(xout,predout, 20,'r', 'filled');
hold on;
x2 =linspace(0, 100, 10);
plot(x2, x2, 'r-', 'LineWidth', 1);

dlm = fitlm(xnew, prednew, 'Intercept', false);
x1 = linspace(min(x), max(x), 1000);
y1 = x1 * dlm.Coefficients.Estimate;
%y1 = polyval(coeff, x1);
hold on;
plot(x1, y1, 'b--', 'LineWidth', 2);
hold on;

str = {};
str = [str, ['Y = ', num2str(dlm.Coefficients.Estimate, '%1.4f'), ' * X ']];
str = [str, ['R^{2} = ' , num2str(dlm.Rsquared.Adjusted, '%1.4f')]];
%str = [str, ['Adjusted R^{2} = ' , num2str(dlm.Rsquared.Adjusted, '%1.4f')]];
dim = [.2 .5 .3 .3];
t = annotation('textbox',dim,'String',str,'FitBoxToText','on');
%t = annotation('textbox');
t.FontSize = 12;
hold on;

iou_legend={};
iou_legend=[ iou_legend, ['predictions']];
iou_legend=[ iou_legend, ['outliers']];
iou_legend=[ iou_legend, ['Y=X']];
iou_legend=[ iou_legend, ['Fitting Curve']];
legend(iou_legend,'Location','northwest','FontSize',12,'NumColumns',2);
lgd=legend;
set(gcf,'Position',[100 100 500 500])
