clc
clear

k_range = 2:10;

num_samples = 1000;
num_features = 5;

mu = zeros(num_features,1,2);
sigma = zeros(num_features,num_features,2);
mu(:,1,1) = [5;10;10;20;25];                   %class 1 features
mu(:,1,2) = [0;5;5;6;9];                  %class 1 features
sigma(:,:,1) = [5 0 0 0 0;...
                0 5 0 0 0;...
                0 0 5 0 0;...
                0 0 0 5 0;...
                0 0 0 0 5];
sigma(:,:,2) = [8 0 0 0 0;...
                0 8 0 0 0;...
                0 0 8 0 0;...
                0 0 0 8 0;...
                0 0 0 0 8];
for k_cluster = k_range;
    [t,x] = gen_data(mu,sigma,num_features,num_samples);
    x=x';
    % trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
    opts = statset('Display','final');
    [idx,C] = kmeans(x,k_cluster,'Replicates',5,'Options',opts);
    
    % figure;
    % plot3(x(idx==1,1),x(idx==1,2),x(idx==1,3),'r.','MarkerSize',12)
    % hold on
    % plot3(x(idx==2,1),x(idx==2,2),x(idx==2,3),'b.','MarkerSize',12)
    % plot3(C(:,1),C(:,2),C(:,3),'kx',...
    %      'MarkerSize',15,'LineWidth',3) 
    % legend('Cluster 1','Cluster 2','Centroids',...
    %        'Location','NW')
    % title 'Cluster Assignments and Centroids'
    % hold off
    
    figure;
    
    plot3(x(idx==1,1),x(idx==1,2),x(idx==1,3),'.','MarkerSize',12)
    hold on
    for k = 2:k_cluster
        plot3(x(idx==k,1),x(idx==k,2),x(idx==k,3),'.','MarkerSize',12)
    end
    plot3(C(:,1),C(:,2),C(:,3),'kx',...
         'MarkerSize',15,'LineWidth',3) 
    
     Legend = cell(length(k_cluster)+1,1)
     for iter=1:k_cluster
       Legend{iter}=strcat("Cluster ", num2str(iter));
     end
     Legend{iter+1}='Centroids';
     legend(Legend)
    xlabel('Feature 1')
    ylabel('Feature 2')
    zlabel('Feature 3')
    Title = strcat('K=',int2str(k_cluster)," Cluster Assignments and Centroids");
    title(Title)
    hold off
end