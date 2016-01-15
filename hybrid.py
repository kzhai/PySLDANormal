"""
Hybrid Update for Vanilla LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy;
import scipy
import nltk;
import sys;

from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer
from variational_bayes import VariationalBayes;

"""
This is a python implementation of vanilla lda, based on a lda approach of variational inference and gibbs sampling, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.

References:
[1] D. Mimno, M. Hoffman, D. Blei. Sparse Stochastic Inference for Latent Dirichlet Allocation. Internal Conference on Machine Learning, Jun 2012.
"""
class Hybrid(VariationalBayes, Inferencer):
    """
    """
    def __init__(self,
                 hyper_parameter_optimize_interval=1,
                 symmetric_alpha_alpha=True,
                 symmetric_alpha_beta=True,
                 ):
        Inferencer.__init__(self, hyper_parameter_optimize_interval);

        self._symmetric_alpha_alpha = symmetric_alpha_alpha
        self._symmetric_alpha_beta = symmetric_alpha_beta

        # self._local_parameter_iteration = local_parameter_iteration
        # assert(self._local_parameter_iteration>0)
                
    def e_step(self,
               parsed_corpus_response=None,
               number_of_samples=10,
               burn_in_samples=5,
               approximate_phi=True):
        
        if parsed_corpus_response == None:
            word_idss = self._parsed_corpus;
            responses = self._responses
        else:
            word_idss, responses = parsed_corpus_response;
        
        number_of_documents = len(word_idss);
        
        document_log_likelihood = 0;
        words_log_likelihood = 0;
        
        # initialize a V-by-K matrix phi sufficient statistics
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_types));
        E_A_sufficient_statistics = numpy.zeros((number_of_documents, self._number_of_topics))
        E_AA_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_topics))
        
        # initialize a D-by-K matrix gamma values
        gamma_values = numpy.zeros((number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;
        
        E_log_beta = compute_dirichlet_expectation(self._beta);
        assert E_log_beta.shape == (self._number_of_topics, self._number_of_types);
        if parsed_corpus_response != None:
            E_log_prob_eta = E_log_beta - scipy.misc.logsumexp(E_log_beta, axis=1)[:, numpy.newaxis]
        exp_E_log_beta = numpy.exp(E_log_beta);

        for doc_id in xrange(number_of_documents):
            
            phi = numpy.random.random((self._number_of_topics, len(word_idss[doc_id])));
            phi = phi / numpy.sum(phi, axis=0)[numpy.newaxis, :];
            phi_sum = numpy.sum(phi, axis=1)[:, numpy.newaxis];
            assert(phi_sum.shape == (self._number_of_topics, 1));
            
            document_phi = numpy.zeros((len(word_idss[doc_id]), self._number_of_topics));
            
            for iter in xrange(number_of_samples):
                for word_pos in xrange(len(word_idss[doc_id])):
                    word_id = word_idss[doc_id][word_pos];
                    
                    phi_sum -= phi[:, word_pos][:, numpy.newaxis];
                    
                    # this is to get rid of the underflow error from the above summation, ideally, phi will become all integers after few iterations
                    phi_sum *= (phi_sum > 0);
                    # assert(numpy.all(phi_sum >= 0));

                    temp_phi = (phi_sum.T + self._alpha_alpha) * exp_E_log_beta[:, [word_id]].T;
                    assert(temp_phi.shape == (1, self._number_of_topics));
                    temp_phi /= numpy.sum(temp_phi);

                    # sample a topic for this word
                    temp_phi = numpy.random.multinomial(1, temp_phi[0])[:, numpy.newaxis];
                    assert(temp_phi.shape == (self._number_of_topics, 1));
                    
                    phi[:, word_pos][:, numpy.newaxis] = temp_phi;
                    phi_sum += temp_phi;

                    # discard the first few burn-in sweeps
                    if iter < burn_in_samples:
                        continue;
                    
                    phi_sufficient_statistics[:, word_id] += temp_phi[:, 0];
                    document_phi[word_pos, :] += temp_phi[:, 0];

            gamma_values[doc_id, :] = self._alpha_alpha + phi_sum.T[0, :];
            # batch_document_topic_distribution[doc_id, :] = self._alpha_alpha + phi_sum.T[0, :];
            
            document_phi /= (number_of_samples - burn_in_samples);
            # this is to prevent 0 during log()
            document_phi += 1e-100;
            assert document_phi.shape == (len(word_idss[doc_id]), self._number_of_topics);
            
            phi_mean = numpy.mean(document_phi, axis=0)
            assert phi_mean.shape == (self._number_of_topics,);
            
            # Note: all terms including E_q[p(\theta|\alpha)], i.e., terms involving \Psi(\gamma), are cancelled due to \gamma updates
            # Note: all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates in M-step
            
            # compute the alpha terms
            document_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha_alpha)) - numpy.sum(scipy.special.gammaln(self._alpha_alpha))
            # compute the gamma terms
            document_log_likelihood += numpy.sum(scipy.special.gammaln(gamma_values[doc_id, :])) - scipy.special.gammaln(numpy.sum(gamma_values[doc_id, :]));
            # compute the phi terms
            document_log_likelihood -= numpy.sum(numpy.log(document_phi) * document_phi);

            # compute the eta terms
            document_log_likelihood -= 0.5 * numpy.log(2 * numpy.pi * self._sigma_square)
            document_log_likelihood -= 0.5 * (responses[doc_id] ** 2 - 2 * responses[doc_id] * numpy.sum(self._eta[0, :] * phi_mean) + numpy.dot(numpy.dot(self._eta, numpy.dot(phi_mean[:, numpy.newaxis], phi_mean[numpy.newaxis, :])), self._eta.T)) / self._sigma_square
            
            # Note: all terms including E_q[p(\_eta | \_beta)], i.e., terms involving \Psi(\_eta), are cancelled due to \_eta updates in M-step
            if parsed_corpus_response != None:
                # compute the p(w_{dn} | z_{dn}, \_eta) terms, which will be cancelled during M-step during training
                words_log_likelihood += numpy.sum(phi.T * E_log_prob_eta[:, word_idss[doc_id]]);
            
            E_A_sufficient_statistics[doc_id, :] = phi_mean;
            E_AA_sufficient_statistics += numpy.dot(phi_mean[:, numpy.newaxis], phi_mean[numpy.newaxis, :])
            
            if (doc_id + 1) % 1000 == 0:
                print "successfully processed %d documents..." % (doc_id + 1);

        phi_sufficient_statistics /= (number_of_samples - burn_in_samples);
        
        # compute mean absolute error
        mean_absolute_error = numpy.abs(numpy.dot(E_A_sufficient_statistics, self._eta.T) - responses[:, numpy.newaxis]).sum()
        
        if parsed_corpus_response == None:
            self._gamma = gamma_values;
            return document_log_likelihood, phi_sufficient_statistics, E_A_sufficient_statistics, E_AA_sufficient_statistics
        else:
            return words_log_likelihood, gamma_values, numpy.dot(E_A_sufficient_statistics, self._eta.T)
        
if __name__ == "__main__":
    print "not implemented..."
