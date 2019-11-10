function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
disp("Theta1 : ");
disp(Theta1);
disp("Theta2 : ");
disp(Theta2);
disp("X : ");
disp(X);
disp("y : ");
disp(y);

a1 = X;
a2 = sigmoid(Theta1 * [ones(size(X, 1), 1) X]');
disp("Hidden_layer : ");
disp(a2);
h_x = sigmoid(Theta2 * [ones(1, size(a2, 2)) ; a2]);
matrix_y = zeros(num_labels, m);
for i = 1:size(y, 1)
  label = y(i);
  matrix_y(label, i) = 1;
endfor
disp("matrix_y : ");
disp(matrix_y);
cost_matrix = matrix_y .* log(h_x) + (ones(size(matrix_y)) - matrix_y) .* (log(ones(size(h_x)) - h_x));
disp("cost_matrix : ");
disp(cost_matrix);
J = (-1 / m) * ones(1, num_labels) * (matrix_y .* log(h_x) + (ones(size(matrix_y)) - matrix_y) .* (log(ones(size(h_x)) - h_x))) * ones(m, 1);
%J = 0;
%for i = 1:m
%  for k = 1:num_labels
%    J = J + (-1/m) * (matrix_y(k, i) * log(h_x(k, i)) + (i - matrix_y(k, i)) * log(1 - h_x(k, i)));
%  endfor
%endfor
disp(J);

#for i = 1:m
#  disp(1)
#  delta3(:,i) = h_x(:,i) - matrix_y(:,i)
#  disp(2);
#  delta2(:,i) = (Theta2(:, 2:end)' * delta3(:,i)) .* (a2(:,i) .* (ones(size(a2(:,i))) - a2(:,i)))
#  disp(3)
#  delta1(:,i) = (Theta1(:, 2:end)' * delta2(:,i)) .* (a1'(:,i) .* (ones(size(a1'(:,i))) - a1'(:,i)))
#endfor

delta3 = h_x - matrix_y;
delta2 = (Theta2(:,2:end)' * delta3) .* (a2 .* (ones(size(a2)) - a2))

Theta1_grad = delta2 * a1
Theta2_grad = delta3 * a2'

#Theta1_grad = (1/m) * delta2 * a1 + lambda * Theta1(:, 2:end);
#Theta2_grad = (1/m) * delta3 * a2' + lambda * Theta2(:, 2:end);





 







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
