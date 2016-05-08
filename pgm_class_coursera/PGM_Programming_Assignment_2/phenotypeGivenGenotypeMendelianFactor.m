function phenotypeFactor = phenotypeGivenGenotypeMendelianFactor(isDominant, genotypeVar, phenotypeVar)
% This function computes the probability of each phenotype given the
% different genotypes for a trait.  It assumes that there is 1 dominant
% allele and 1 recessive allele.
%
% If you do not have much background in genetics, you should read the
% on-line Appendix or watch the Khan Academy Introduction to Heredity Video
% (http://www.khanacademy.org/video/introduction-to-heredity?playlist=Biology)
% before you start coding.
%
% For the genotypes, assignment 1 maps to homozygous dominant, assignment 2
% maps to heterozygous, and assignment 3 maps to homozygous recessive.  For
% example, say that there is a gene with two alleles, dominant allele A and
% recessive allele a.  Assignment 1 would map to AA, assignment 2 would
% make to Aa, and assignment 3 would map to aa.  For the phenotypes, 
% assignment 1 maps to having the physical trait, and assignment 2 maps to 
% not having the physical trait.
%
% THE VARIABLE TO THE LEFT OF THE CONDITIONING BAR MUST BE THE FIRST
% VARIABLE IN THE .var FIELD FOR GRADING PURPOSES
%
% Input:
%   isDominant: 1 if the trait is caused by the dominant allele (trait 
%   would be caused by A in example above) and 0 if the trait is caused by
%   the recessive allele (trait would be caused by a in the example above)
%   genotypeVar: The variable number for the genotype variable (goes in the
%   .var part of the factor)
%   phenotypeVar: The variable number for the phenotype variable (goes in
%   the .var part of the factor)
%
% Output:
%   phenotypeFactor: Factor denoting the probability of having each 
%   phenotype for each genotype

phenotypeFactor = struct('var', [], 'card', [], 'val', []);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INSERT YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

% Fill in phenotypeFactor.var.  This should be a 1-D row vector.
phenotypeFactor.var = [phenotypeVar genotypeVar];
% Fill in phenotypeFactor.card.  This should be a 1-D row vector.
phenotypeFactor.card = [2 3]; % assumed 2 allele
% KA - check: https://class.coursera.org/pgm/forum/thread?thread_id=473
% KA - Also read in Section 2 of PA2Appendix.pdf about combination of alleles

phenotypeFactor.val = zeros(1, prod(phenotypeFactor.card));
% Replace the zeros in phentoypeFactor.val with the correct values.
assignments = IndexToAssignment(1:prod(phenotypeFactor.card), phenotypeFactor.card);
for i = 1:size(assignments,1)
    phenotype = assignments(i,1);
    genotype = assignments(i,2);
    
    % phenotype: 1 :: gets the trait
    % phenotype: 2 :: doesn't gets the trait
    
    switch genotype
        case 1
            if isDominant
                phenotypeFactor.val(i) = phenotype == 1;
            else
                phenotypeFactor.val(i) = phenotype == 2;
            end
        case 2
            if isDominant
                phenotypeFactor.val(i) = phenotype == 1;
            else
                phenotypeFactor.val(i) = phenotype == 2;
            end
        case 3
            if isDominant
                phenotypeFactor.val(i) = phenotype == 2;
            else
                phenotypeFactor.val(i) = phenotype == 1;
            end
        otherwise
    end
    
%     if genotype == 1
%         % case: AA
%         if phenotype == 1
%             phenotypeFactor.val(i) = true;
%         else
%             phenotypeFactor.val(i) = false;
%         end
%     elseif genotype == 2
%         % case: Aa
%         if phenotype == 1
%             phenotypeFactor.val(i) = isDominant;
%         else
%             phenotypeFactor.val(i) = ~isDominant;
%         end
%     elseif genotype == 3
%         % case: aa
%         if phenotype == 1
%             phenotypeFactor.val(i) = false;
%         else
%             phenotypeFactor.val(i) = true;
%         end
%     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
