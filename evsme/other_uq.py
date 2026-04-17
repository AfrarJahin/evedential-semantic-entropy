import numpy as np


def mass_on_total_set(log_liks, responses):

    responses_set = list(set(responses))

    # NO Length normalization of generation probabilities.
    log_liks_agg_not_normalized = [np.sum(log_lik) for log_lik in log_liks]
    not_normalized_sequence_prob = [np.exp(log_liks_agg_not_normalized[i]) for i in range(len(log_liks_agg_not_normalized))] 
    not_normalized_sequence_prob_set = [not_normalized_sequence_prob[responses.index(str(response))] for response in responses_set]

    return 1 - sum(not_normalized_sequence_prob_set)




def semantic_entropy_correction(log_liks, semantic_ids, responses, sequence_prob_type, do_correct,  **kwargs):
    responses_set = list(set(responses))

    if sequence_prob_type == "raw":
        sequence_logprob = [np.sum(log_lik) for log_lik in log_liks]
        sequence_prob = [np.exp(sequence_logprob[i]) for i in range(len(sequence_logprob))]

    if sequence_prob_type == "length_normalization_as_for_se":
        sequence_logprob = [np.mean(log_lik) for log_lik in log_liks]
        sequence_prob = [np.exp(sequence_logprob[i]) for i in range(len(sequence_logprob))]
        

    if not do_correct:

        cluster_list = list(set(semantic_ids))
        proba_cluster = [0 for i in range(len(cluster_list))]

        for sequence_index in range(len(semantic_ids)):

            cluster = semantic_ids[sequence_index]

            proba_cluster[cluster] += sequence_prob[sequence_index]

        proba_cluster = [i/sum(proba_cluster) for i in proba_cluster]

        
        log_proba_clusterTIMESproba_cluster = [0 for i in range(len(cluster_list))] 
        for cluster in cluster_list:
            log_proba_clusterTIMESproba_cluster[cluster] += np.log(proba_cluster[cluster])*proba_cluster[cluster]

        return(-np.sum(log_proba_clusterTIMESproba_cluster))


    else:
        sequence_prob_uniqueEvents = [sequence_prob[responses.index(response)] for response in responses_set]
        semantic_ids_uniqueEvents = [semantic_ids[responses.index(response)] for response in responses_set]
        # e = - sum [p(c)*log(p(c))]
        # log(p(c)) = sum(log(p(s))), s in c (log_proba_cluster)
        # p(c) = sum(p(s)), s in c (proba_cluster)

        cluster_list = list(set(semantic_ids_uniqueEvents))

        proba_cluster = [0 for i in range(len(cluster_list))] 

        for event_index in range(len(semantic_ids_uniqueEvents)):

            cluster_index = semantic_ids_uniqueEvents[event_index]

            proba_cluster[cluster_index] += sequence_prob_uniqueEvents[event_index]


        if kwargs['sumto1']== True:
            proba_cluster = [proba_cluster[i]/sum(proba_cluster) for i in range(len(proba_cluster))]
        
        log_proba_clusterTIMESproba_cluster = [0 for i in range(len(cluster_list))] 
        for cluster in cluster_list:
            log_proba_clusterTIMESproba_cluster[cluster] += np.log(proba_cluster[cluster])*proba_cluster[cluster]

        entropie_raw = - np.sum(log_proba_clusterTIMESproba_cluster)

        if kwargs['entropie_normalize']== "raw":
            return entropie_raw

        if kwargs['entropie_normalize']== "log":
            if len(cluster_list) == 1:
                return np.float64(0)
            else:
                return entropie_raw/np.log(len(cluster_list))
    

