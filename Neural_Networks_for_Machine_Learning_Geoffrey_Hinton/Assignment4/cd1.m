function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    % error('not yet implemented');
    flag_real_visible_data = 1; % e.g. pixel intensities in image
    if flag_real_visible_data == 1
      visible_data = sample_bernoulli(visible_data);
    end  
    
    h0_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    h0_data = sample_bernoulli(h0_probabilities);
    reconstructed_probabilities = hidden_state_to_visible_probabilities(rbm_w, h0_data);
    reconstructed_data = sample_bernoulli(reconstructed_probabilities);
    h1_probabilities = visible_state_to_hidden_probabilities(rbm_w, reconstructed_data);
    
    positive_gradient = configuration_goodness_gradient(visible_data, h0_data);
    flag_use_sample = 0;
    if flag_use_sample == 1
      h1_data = sample_bernoulli(h1_probabilities);
      negative_gradient = configuration_goodness_gradient(reconstructed_data, h1_data);
    else
      % instead of sample for hidden data at step 1, we use conditional probabilities
      negative_gradient = configuration_goodness_gradient(reconstructed_data, h1_probabilities);
    end  
    ret = positive_gradient - negative_gradient;
end
