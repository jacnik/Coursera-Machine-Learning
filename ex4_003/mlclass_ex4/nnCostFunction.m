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
X = [ones(m, 1), X]; % prepend column of ones to X

h1 = sigmoid(X * Theta1'); % first hidden layer activations (5000 x 25)
h1 = [ones(m, 1), h1]; % prepend column of ones to h1 (5000 x 26)

h2 = sigmoid(h1 * Theta2'); % output layer activations (5000 x 10)

% used to be => bin_y = repmat(1:num_labels, m, 1) == y; 
% but bsxfun is used to suppress the warning 'automatic broadcasting operation applied'
bin_y = bsxfun("eq", repmat(1:num_labels, m, 1), y); % this creates matrix of walues 1 in y'th column for each row (5000 x 10)
%   [1, 2, 3, 4;        [1;       [1, 0, 0, 0;
%    1, 2, 3, 4;   ==    2;   ->   0, 1, 0, 0;
%    1, 2, 3, 4]         3]        0, 0, 1, 0]

% todo try to vectorize this element vise multiplications and sum function
J = 1/m * sum(( -bin_y .* log(h2) - (1 - bin_y) .* log(1 - h2) )(:));

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
for t = 1:m
    % step 1: for each t-th training example perform a feedforward pass computing
    %         the activations for each layer
    a1 = X(t, :); % row vector (one set of attributes to learn), bias unit is already added. (1 x 401)
    
    z2 = a1 * Theta1'; % (1 x 25)
    a2 = sigmoid(z2); % first hidden layer activations (1 x 25)
    a2 = [1, a2]; % add one as a first column (bias unit) (1 x 26)

    a3 = sigmoid(a2 * Theta2'); % output layer activations (1 x 10)
    
    % step 2: calculate error terms for output layer
    d3 = a3 - bin_y(t, :);  %(1 x 10)
    
    % step 3: calculate error terms in hidden layer
    % todo: dimensions don't match (d3 * Theta2 is 1 x 26 and z2 is 1 x 25)
    %       so either use a2 instead of z2 
    %       but more probably just remove first column from d3 * Theta2, because it's the bias unit
    d2 = (d3 * Theta2)(2:end) .* sigmoidGradient(z2); % (1 x 25)
    
    % step 4: accumulate the gradient
    Theta1_grad += d2' * a1; % (25 x 401)
    Theta2_grad += d3' * a2; % (10 x 26)
    
end
    % step 5: Obtain the unregularized gradient by dividing accumulated gradients by m
Theta1_grad *= 1/m;
Theta2_grad *= 1/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% add regularization
% Theta1(:,2:end)(:)' * Theta1(:,2:end)(:) is the sum of squares off all the elements except first column
J += lambda/(2 * m) * ( Theta1(:,2:end)(:)' * Theta1(:,2:end)(:) + Theta2(:,2:end)(:)' * Theta2(:,2:end)(:) );

% not regularize first (bias therms) column of Theta's
Theta1_grad(:,2:end) += lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) += lambda/m * Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
