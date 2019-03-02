function [A, Y, EigenValues] = PCA_transformation(TrainImages, N)
%Function that return the PCA transformed features, transformation matrix and 
%the corresponding eigen values
%% Input arguments: 
%Train Images (3500 x 400): N (Number of PCA Features)
%% Output arguments: 
%A :(Transformation Matrix) (400 x N)
%Y : (Train Images after PCA Transformation) (N x 3500)
%Eigen Values: (up to ‘N’ PCA features) (N x 1)

%%
                 
Covariance = cov(TrainImages);
[A EigenValues] = eig(Covariance);
[EigenValues order] = sort(diag(EigenValues), 'descend'); 


A = A(:,order(1:N));


% Y = (TrainImages*A(:,1:end));
Y = (TrainImages*A);


end

