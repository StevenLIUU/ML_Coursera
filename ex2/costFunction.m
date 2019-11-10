function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%J = 1/(2*m) * ((((X * theta) - y) .^ 2)' * ones(size(X)(1), 1));

h_x = sigmoid(X * theta);

J = (-1 / m) * (y' * log(h_x) + (ones(size(y')) - y') * log(ones(size(h_x)) - h_x));

%grad(0) = (-1 / m) * ((h_x - y)' * ones(size(h_x)))

grad = (1 / m) * ((h_x - y)' * X);


% =============================================================

end
