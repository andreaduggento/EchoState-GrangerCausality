function mse_value = mean_squared_error(actual, predicted)
    % Validate the dimensions
    if size(actual) ~= size(predicted)
        error('Dimensions of actual and predicted must match.');
    end
    
    % Calculate MSE
    mse_value = mean((actual - predicted).^2);
end
