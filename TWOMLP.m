% Custom MLP Implementation - Function Approximation
clc; clear all; close all;

% Generate input data (3 inputs) and target data (2 outputs)
input_all = rand(3,1000);
in1 = input_all(1,:); in2 = input_all(2,:); in3 = input_all(3,:);
target_out1 = 2*cos(in1) + in2 + 3*in3;
target_out2 = in2.^2 + in1.^2 + in1.*in3 + sin(in3);
target_all  = [target_out1; target_out2];

% Data Partitioning
num_samples_train = 700; num_samples_test = 200; num_samples_check = 100;

input_train  = input_all(:,1:num_samples_train);     
target_train = target_all(:,1:num_samples_train);

input_test   = input_all(:,num_samples_train+1:num_samples_train+num_samples_test); 
target_test  = target_all(:,num_samples_train+1:num_samples_train+num_samples_test);

input_check  = input_all(:,num_samples_train+num_samples_test+1:end); 
target_check = target_all(:,num_samples_train+num_samples_test+1:end);

% Network Architecture
num_inputs  = 3;  num_outputs = 2;
num_hidden1 = 10; num_hidden2 = 5;
learning_rate = 0.0001;
num_epochs    = 100;

% Initialize Weights and Biases
weights1 = rand(num_hidden1,num_inputs)*2-1; bias1 = rand(num_hidden1,1)*2-1;
weights2 = rand(num_hidden2,num_hidden1)*2-1; bias2 = rand(num_hidden2,1)*2-1;
weights3 = rand(num_outputs,num_hidden2)*2-1;  bias3 = rand(num_outputs,1)*2-1;

% Pre-allocate Arrays
mse_train_history = zeros(1,num_epochs);
mse_test_history  = zeros(1,num_epochs);

pred_train = zeros(num_outputs,num_samples_train);
pred_test  = zeros(num_outputs,num_samples_test);

%% Training Loop
figure(1) 
for epoch = 1:num_epochs
    current_train_error = [];

    for k = 1:num_samples_train
        % 1) Feedforward
        layer0_out = input_train(:,k);
        layer1_net = weights1*layer0_out + bias1;
        layer1_out = tansig(layer1_net);
        layer2_net = weights2*layer1_out + bias2;
        layer2_out = tansig(layer2_net);
        layer3_net = weights3*layer2_out + bias3;
        layer3_out = purelin(layer3_net);

        pred_train(:,k) = layer3_out;
        error_output = target_train(:,k) - layer3_out;
        current_train_error = [current_train_error, error_output];

        % 2) Backpropagation
        % 2.1) Output Layer Delta
        deriv_f3 = 1; 
        delta3 = deriv_f3 * error_output; 

        weights3_prev = weights3;
        weights3 = weights3 + learning_rate * (delta3 * layer2_out');
        bias3 = bias3 + learning_rate * delta3;

        % 2.2) Second Hidden Layer Delta
        deriv_f2 = 1 - layer2_out.^2;
        delta2 = zeros(num_hidden2,1);
        for i = 1:num_outputs
            for j = 1:num_hidden2
                delta2(j) = delta2(j) + deriv_f2(j) * weights3_prev(i,j) * delta3(i);
            end
        end

        weights2_prev = weights2;
        weights2 = weights2 + learning_rate * (delta2 * layer1_out');
        bias2 = bias2 + learning_rate * delta2;

        % 2.3) First Hidden Layer Delta
        deriv_f1 = 1 - layer1_out.^2;
        delta1 = zeros(num_hidden1,1);
        for i = 1:num_hidden2
            for j = 1:num_hidden1
                delta1(j) = delta1(j) + deriv_f1(j) * weights2_prev(i,j) * delta2(i);
            end
        end

        weights1 = weights1 + learning_rate * (delta1 * layer0_out');
        bias1 = bias1 + learning_rate * delta1;
    end

    mse_train_history(epoch) = mse(current_train_error);

    % Test Evaluation
    current_test_error = [];
    for k = 1:num_samples_test
        layer0_out = input_test(:,k);
        layer1_net = weights1*layer0_out + bias1;
        layer1_out = tansig(layer1_net);
        layer2_net = weights2*layer1_out + bias2;
        layer2_out = tansig(layer2_net);
        layer3_net = weights3*layer2_out + bias3;
        layer3_out = purelin(layer3_net);
        pred_test(:,k) = layer3_out;
        current_test_error = [current_test_error, target_test(:,k)-layer3_out];
    end
    mse_test_history(epoch) = mse(current_test_error);
    
    subplot(2,1,1)
    plot(mse_train_history(1:epoch))
    title('MSE error for training data')
    
    subplot(2,1,2)
    plot(mse_test_history(1:epoch))
    title('MSE error for test data')
    
    drawnow
end

% Final Errors
final_mse_train = mse(current_train_error)
final_mse_test  = mse(current_test_error)

%% Check Data
check_error_vector = [];
pred_check = [];
for i = 1:num_samples_check
    layer0_out = input_check(:,i);
    layer1_net = weights1*layer0_out + bias1;
    layer1_out = tansig(layer1_net);
    layer2_net = weights2*layer1_out + bias2;
    layer2_out = tansig(layer2_net);
    layer3_net = weights3*layer2_out + bias3;
    layer3_out = purelin(layer3_net);
    check_error_vector = [check_error_vector, target_check(:,i)-layer3_out];
    pred_check = [pred_check, layer3_out];
end
mse_check_final = mse(check_error_vector)

%% Plotting
figure
plot(pred_train'); hold on; plot(target_train','r');
title('Outputs for training data & target')
legend('y1 network', 'y2 network', 'y1 target', 'y2 target')

figure
plot(pred_test'); hold on; plot(target_test','r');
title('Outputs for testing data & target')
legend('y1 network', 'y2 network', 'y1 target', 'y2 target')

figure
plot(pred_check'); hold on; plot(target_check','r');
title('Outputs for checking data & target')
legend('y1 network', 'y2 network', 'y1 target', 'y2 target')
