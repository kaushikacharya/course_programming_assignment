function movie_recommender(vote_matrix, r_matrix, weight_calculation_method)
% Author: Kaushik Acharya
% Date: July 10,2012
% Paper: 
% 1. Empirical Analysis of Predictive Algorithms for Collaborative
%       Filtering (1998) by Breese et al.
% 2. Towards a Scalable kNN CF Algorithm: Exploring Effective Applications
%       of Clustering (2007) by Rashid et al.

% load E:\course_programming_assignment\ml_class_coursera\ex8\ex8\ex8_movies.mat
% row - movies
% col - users

n_movies = size(vote_matrix,1);
n_users = size(vote_matrix,2);

% mean vote for each user
mean_vote_user_array = sum(vote_matrix .* r_matrix) ./ sum(r_matrix);

% compute the user correlation matrix 
user_corr_matrix = zeros(n_users);

% N.B. use of notation of i,j,k different from the paper(1). Don't get
% confused

switch weight_calculation_method
    case 1 % Pearson correlation coefficient
        case_amplification = false; rho = 2.5;
        for user_i = 1:n_users-1
            for user_j = user_i+1:n_users
                flag_common_array = r_matrix(:,user_i) .* r_matrix(:,user_i);
                diff_user_i_array = (flag_common_array .* vote_matrix(:,user_i)) - (mean_vote_user_array(user_i) * flag_common_array);
                diff_user_j_array = (flag_common_array .* vote_matrix(:,user_j)) - (mean_vote_user_array(user_j) * flag_common_array);
                
                user_corr_matrix(user_i,user_j) = (diff_user_i_array' * diff_user_j_array) / ...
                    sqrt( (diff_user_i_array' * diff_user_i_array) * (diff_user_j_array' * diff_user_j_array)  );
                
                if case_amplification % section 2.2.3 paper(1)
                    user_corr_matrix(user_i,user_j) = power(user_corr_matrix(user_i,user_j),rho);
                end
                user_corr_matrix(user_j,user_i) = user_corr_matrix(user_i,user_j);
            end
        end
    case 2 % vector similarity
        for user_i = 1:n_users-1
            % assumption: no negative vote present
            denom_user_i = sqrt((r_matrix(:,user_i) .* vote_matrix(:,user_i))' * vote_matrix(:,user_i));
            for user_j = user_i+1:n_users
                denom_user_j = sqrt((r_matrix(:,user_j) .* vote_matrix(:,user_j))' * vote_matrix(:,user_j));
                flag_common_array = r_matrix(:,user_i) .* r_matrix(:,user_i);
                
                user_corr_matrix(user_i,user_j) = (flag_common_array .* (vote_matrix(:,user_i)/denom_user_i))' * ...
                    (flag_common_array .* (vote_matrix(:,user_j)/denom_user_j));
                user_corr_matrix(user_j,user_i) = user_corr_matrix(user_i,user_j);
            end
        end
    case 3 % kNN algorithm -paper(2)
        n_user_clusters = 5; % user-defined value
        count_user_clusters = 1;
        
        user_clusters_struct(count_user_clusters).user_indices = 1:n_users;
        user_clusters_struct(count_user_clusters).user_n = n_users;
        
        while count_user_clusters < n_user_clusters
            % bisect the largest cluster
            [junk,cluster_to_split] = max([user_clusters_struct.user_n]);
            
            % apply 2 means
            % randomly choose two cluster center in cluster_to_split
            center_array = ceil(rand(1,2)*user_clusters_struct(cluster_to_split).user_n);
            
            cluster_1_center_index = user_clusters_struct(cluster_to_split).user_indices(center_array(1));
            cluster_2_center_index = user_clusters_struct(cluster_to_split).user_indices(center_array(2));
            
            for user_i = user_clusters_struct(cluster_to_split).user_indices
                
            end
            
            count_user_clusters = count_user_clusters + 1;
        end
end

predict_matrix = NaN(n_movies,n_users);

user_a = 1;
for item_j = 1:n_movies
    wt_sum = 0; k_norm = 0;

    for user_i = 1:n_users
        k_norm = k_norm + abs(user_corr_matrix(user_a,user_i));
        wt_sum = wt_sum + user_corr_matrix(user_a,user_i) * ( vote_matrix(item_j,user_i) - mean_vote_user_array(user_i) );
    end

    predict_matrix(item_j,user_a) =  mean_vote_user_array(user_a) + wt_sum/k_norm;
    % vote_matrix(item_j,user_a)
end

% analysing visually
[sorted_corr_values, sorted_user_indices] = sort(user_corr_matrix(user_a,:),'descend');
[sorted_corr_values(1:50); sorted_user_indices(1:50); mean_vote_user_array(sorted_user_indices(1:50)); vote_matrix(item_j,sorted_user_indices(1:50))-mean_vote_user_array(sorted_user_indices(1:50))];

user_b = 2;
% compare user_a with user_b movie ratings
common_movie_indices = find(r_matrix(:,user_a) & r_matrix(:,user_b))
figure; hold on; grid on; 
plot(1:length(common_movie_indices),vote_matrix(common_movie_indices,user_a),'b*');  
plot(1:length(common_movie_indices),vote_matrix(common_movie_indices,user_b),'rs');
figure; hold on; grid on;
plot(1:length(common_movie_indices),vote_matrix(common_movie_indices,user_a)-vote_matrix(common_movie_indices,user_b),'b*');

% now compare with the user with which user_a has the best correlation with
[max_corr,user_c] = max(user_corr_matrix(user_a,:));
common_movie_indices = find(r_matrix(:,user_a) & r_matrix(:,user_c))
figure; hold on; grid on; 
plot(1:length(common_movie_indices),vote_matrix(common_movie_indices,user_a),'b*');  
plot(1:length(common_movie_indices),vote_matrix(common_movie_indices,user_c),'rs');

end % main