% computes log of partition(Z)
function ret = compute_partition(rbm_w, n_hidden_states, n_visible_states)
    flag_iterate_over_visible = 0
    ret = 0;
    
    if flag_iterate_over_visible
        % iterate over visible states
        % Note: Slower as number of visible states(256) is much bigger than number of hidden states(10)
        visible_matrix = binary_matrix(n_visible_states);    
        for row = 1:size(visible_matrix,1)
            visible_vector = visible_matrix(row,:);
            
            prod_over_hidden_states = 1;
            for j = 1:n_hidden_states
                prod_over_hidden_states *= 1 + exp(rbm_w(j,:) * visible_vector');
            end
            
            ret += prod_over_hidden_states;
        end
        
        ret = log(ret);
    else
        % iterate over hidden states
        hidden_matrix = binary_matrix(n_hidden_states);
        for row = 1:size(hidden_matrix,1)
            hidden_vector = hidden_matrix(row,:);
            
            prod_over_visible_states = 1;
            for i = 1:n_visible_states
                prod_over_visible_states *= 1 + exp(hidden_vector * rbm_w(:,i));
            end
            
            ret += prod_over_visible_states;
        end
        
        ret = log(ret);
    end
    
end

function mat = binary_matrix(n_states)
    mat = zeros(pow2(n_states),n_states);
    mat(2,1) = 1;
    
    for cur_max_col = 2:n_states
        cur_max_row = pow2(cur_max_col-1);
        for row = 1:cur_max_row
            new_row = row+cur_max_row;
            for col = 1:cur_max_col-1
                mat(new_row,col) = mat(row,col);
            end
            mat(new_row,cur_max_col) = 1; 
        end
    end
end

% resource: https://www.coursera.org/learn/neural-networks/discussions/weeks/13/threads/E4SUt55lEea8DBIggA9NJg
% Notes provided above: http://www.crim.ca/perso/patrick.kenny/BMNotes.pdf  (page 9)
% Also helpful: http://deeplearning.net/tutorial/rbm.html