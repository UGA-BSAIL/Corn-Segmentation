clear;
load('PRCurve.mat')

iou_legend={};
for i=1:5
    PR=squeeze(precision_recall(i,:,:));
    x=PR(2,:);
    y=PR(1,:);
    ln=plot(x,y);
    ln.LineWidth = 2;
    IOU=0.7+0.05*(i-1);
    iou_legend=[iou_legend, ['IOU_{', num2str(IOU, '%.2f'), '}=', num2str(AP(i), '%.2f')]];
    hold on;
end

legend(iou_legend,'Location','northeast','FontSize',20,'NumColumns',2);
lgd=legend;
lgd.Title.String =['Mean Average Precision'];
xlabel('Recall','FontSize',20);
ylabel('Precision','FontSize',20);
set(gca,'linewidth',2);
set(gca,'fontsize',20,'fontweight','bold','fontangle','normal');

mIOU=mean(IOUs{1});

