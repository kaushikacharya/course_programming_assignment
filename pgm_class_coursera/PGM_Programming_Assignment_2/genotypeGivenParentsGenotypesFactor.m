function genotypeFactor = genotypeGivenParentsGenotypesFactor(numAlleles, genotypeVarChild, genotypeVarParentOne, genotypeVarParentTwo)
% This function computes a factor representing the CPD for the genotype of
% a child given the parents' genotypes.

% THE VARIABLE TO THE LEFT OF THE CONDITIONING BAR MUST BE THE FIRST
% VARIABLE IN THE .var FIELD FOR GRADING PURPOSES

% When writing this function, make sure to consider all possible genotypes 
% from both parents and all possible genotypes for the child.

% Input:
%   numAlleles: int that is the number of alleles
%   genotypeVarChild: Variable number corresponding to the variable for the
%   child's genotype (goes in the .var part of the factor)
%   genotypeVarParentOne: Variable number corresponding to the variable for
%   the first parent's genotype (goes in the .var part of the factor)
%   genotypeVarParentTwo: Variable number corresponding to the variable for
%   the second parent's genotype (goes in the .var part of the factor)
%
% Output:
%   genotypeFactor: Factor in which val is probability of the child having 
%   each genotype (note that this is the FULL CPD with no evidence 
%   observed)

% The number of genotypes is (number of alleles choose 2) + number of 
% alleles -- need to add number of alleles at the end to account for homozygotes

genotypeFactor = struct('var', [], 'card', [], 'val', []);

% Each allele has an ID.  Each genotype also has an ID.  We need allele and
% genotype IDs so that we know what genotype and alleles correspond to each
% probability in the .val part of the factor.  For example, the first entry
% in .val corresponds to the probability of having the genotype with
% genotype ID 1, which consists of having two copies of the allele with
% allele ID 1, given that both parents also have the genotype with genotype
% ID 1.  There is a mapping from a pair of allele IDs to genotype IDs and 
% from genotype IDs to a pair of allele IDs below; we compute this mapping 
% using generateAlleleGenotypeMappers(numAlleles). (A genotype consists of 
% 2 alleles.)

[allelesToGenotypes, genotypesToAlleles] = generateAlleleGenotypeMappers(numAlleles);

% One or both of these matrices might be useful.
%
%   1.  allelesToGenotypes: n x n matrix that maps pairs of allele IDs to 
%   genotype IDs, where n is the number of alleles -- if 
%   allelesToGenotypes(i, j) = k, then the genotype with ID k comprises of 
%   the alleles with IDs i and j
%
%   2.  genotypesToAlleles: m x 2 matrix of allele IDs, where m is the 
%   number of genotypes -- if genotypesToAlleles(k, :) = [i, j], then the 
%   genotype with ID k is comprised of the allele with ID i and the allele 
%   with ID j

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INSERT YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

% Fill in genotypeFactor.var.  This should be a 1-D row vector.
genotypeFactor.var = [genotypeVarChild  genotypeVarParentOne  genotypeVarParentTwo];
% Fill in genotypeFactor.card.  This should be a 1-D row vector.
genotypeFactor.card  = size(genotypesToAlleles,1)*ones(1,3);

genotypeFactor.val = zeros(1, prod(genotypeFactor.card));
% Replace the zeros in genotypeFactor.val with the correct values.
assignments = IndexToAssignment(1:prod(genotypeFactor.card), genotypeFactor.card);

for i = 1:size(assignments,1)
    genotypeChild = assignments(i,1);
    genotypeParentOne = assignments(i,2);
    genotypeParentTwo = assignments(i,3);
    
    % [from the comment section] A genotype consists of 2 alleles.
    % allele_pair_child = genotypesToAlleles(genotypeChild,:);
    allele_pair_ParentOne = genotypesToAlleles(genotypeParentOne,:);
    allele_pair_ParentTwo = genotypesToAlleles(genotypeParentTwo,:);
    
    prob = 0;
    for row_i = 1:size(allelesToGenotypes,1)
        col_i = find(allelesToGenotypes(row_i,:) == genotypeChild, 1);

        if isempty(col_i)
            continue;
        end
        % probs
        allele_pair_child_current = [row_i col_i];
        prob_allele_1 = sum(allele_pair_child_current(1) == allele_pair_ParentOne)/2;
        prob_allele_2 = sum(allele_pair_child_current(2) == allele_pair_ParentTwo)/2;
        
        prob = prob + prob_allele_1*prob_allele_2;
    end
    
    genotypeFactor.val(i) = prob;
    
%     if genotypeParentOne == 1
%         % case: parent1: AA
%         switch  genotypeParentTwo
%             case 1 % parent2: AA
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 1;
%                     case 2 % Aa
%                         val = 0;
%                     case 3 % aa
%                         val = 0;
%                 end
%             case 2 % parent2: Aa
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 0.5;
%                     case 2 % Aa
%                         val = 0.5;
%                     case 3 % aa
%                         val = 0;
%                 end
%             case 3 % parent2: aa
%                 switch genotypeChild
%                     case 1
%                         val = 0;
%                     case 2
%                         val = 1;
%                     case 3
%                         val = 0;
%                 end
%         end
%     elseif genotypeParentOne == 2
%         % case: parent1: Aa
%         switch  genotypeParentTwo
%             case 1 % parent2: AA
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 0.5;
%                     case 2 % Aa
%                         val = 0.5;
%                     case 3 % aa
%                         val = 0;
%                 end
%             case 2 % parent2: Aa
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 0.25;
%                     case 2 % Aa
%                         val = 0.5;
%                     case 3 % aa
%                         val = 0.25;
%                 end
%             case 3 % parent2: aa
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 0;
%                     case 2 % Aa
%                         val = 0.5;
%                     case 3 % aa
%                         val = 0.5;
%                 end
%         end
%     elseif genotypeParentOne == 3
%         % case: parent1: aa
%         switch  genotypeParentTwo
%             case 1 % parent2: AA
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 0;
%                     case 2 % Aa
%                         val = 1;
%                     case 3 % aa
%                         val = 0;
%                 end
%             case 2 % parent2: Aa
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 0;
%                     case 2 % Aa
%                         val = 0.5;
%                     case 3 % aa
%                         val = 0.5;
%                 end
%             case 3 % parent2: aa
%                 switch genotypeChild
%                     case 1 % AA
%                         val = 0;
%                     case 2 % Aa
%                         val = 0;
%                     case 3 % aa
%                         val = 1;
%                 end
%         end
%     end
%     
%     genotypeFactor.val(i) = val;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%