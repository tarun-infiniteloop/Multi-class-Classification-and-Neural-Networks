function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(size(X,1), 1) X]; % column of ones is added to X, now it is  5000*401
a2 = zeros(25,5000);
a2 = sigmoid(Theta1*X'); % 25*5000
q = size(a2,2);
a2 = [ones(1, q); a2]; % column of ones is added 26*5000

a3 = zeros(10, 5000);
a3 = sigmoid(Theta2*a2); % 10*5000

[m,i] = max(a3);

p = i';

% =========================================================================


end
