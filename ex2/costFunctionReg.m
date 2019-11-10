function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_x = sigmoid(X * theta);
%disp(h_x);

theta_raw = theta;
theta_raw(1) = 0;

disp(theta);

J = (-1 / m) * (y' * log(h_x) + (ones(size(y')) - y') * log(ones(size(h_x)) - h_x)) + lambda / (2 * m) * ones(1, size(theta_raw, 1)) * (theta_raw .^ 2);

disp(J);

grad = (1 / m) * ((h_x - y)' * X)' + lambda / m * theta_raw;

disp((1 / m) * ((h_x - y)' * X)');
disp(lambda / m * theta_raw);

% =============================================================

end
