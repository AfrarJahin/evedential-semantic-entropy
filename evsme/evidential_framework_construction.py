
import copy
import networkx as nx
from itertools import chain
import numpy as np
import logging


from evsme.evidence_theory_utils import myBin2dec
from evsme.evidence_theory_utils import vector_to_set_expression
from semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids


def all_entailements(strings_list, model, example=None):
    """Group list of predictions into semantic meaning."""

    n = m = len(strings_list)
    M = [[-1] * n for _ in range(m)]

    for i, string1 in enumerate(strings_list):
        for j, string2 in enumerate(strings_list):
            if string1 == string2:
                M[i][j] = 2
            else:
                M[i][j], _  = model.check_implication(string1, string2, example=example)

    return M



def get_semantic_ids_and_hierarchy(strings_list, entailements_full_matrix):
    """Group list of predictions into semantic meaning."""

    n = len(strings_list)
    # print("entailements_full_matrix" + str(entailements_full_matrix))

    M = np.array(entailements_full_matrix)
    Mprim = M.copy()


    # Initialise strings beloging to n diferent cluster (one cluster for one string).
    semantic_set_ids = list(range(n))

    Mprim = M.copy()
    nprim = n

    while True:

        
        relations = [(i, j) for i, row in enumerate(Mprim) for j, value in enumerate(row) if value == 2 and i !=j]
        G = nx.DiGraph(relations)

        try:
            cycle = nx.find_cycle(G, orientation='original')
        
        except:
            break
        
        

        [i,j] = [cycle[0][0], cycle[0][1]]

        
        for k in range(n):
            if semantic_set_ids[k] == j:
                semantic_set_ids[k] = i
        for k in range(nprim):
            if not Mprim[i][k] == Mprim[j][k]:
                Mprim[i][k] = -1
            if not Mprim[k][i] == Mprim[k][j]:
                Mprim[k][i] = -1

        # Update Mprim and semantic_set_ids (remove info related to obsolete cluster j)
        Mprim = np.delete(Mprim, j, axis=0)
        Mprim = np.delete(Mprim, j, axis=1)

        for k in range(n):
            if semantic_set_ids[k] > j:
                semantic_set_ids[k] = semantic_set_ids[k] - 1

        nprim = Mprim.shape[0]
        

    R = []
    for i in range(nprim):
        for j in range(nprim):
            if i == j:
                continue
            if Mprim[i][j]==2:
                R.append([i,j])
                # R_string.append([strings_list[semantic_set_ids[i]], strings_list[semantic_set_ids[j]]])
                # R_string.append([strings_list[semantic_set_ids.index(i)], strings_list[semantic_set_ids.index(j)]])

    Results = {"c": semantic_set_ids, "R": R}

    return(Results)




def eliminate_redundant_relations(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Initialize a list to store non-redundant edges
    non_redundant_edges = []

    for u, v in list(G.edges()):
        # Temporarily remove the edge to check for redundant path
        G.remove_edge(u, v)
        
        # Check if there is still a path from u to v
        if not nx.has_path(G, u, v):
            # If not, the edge was non-redundant, add it back
            non_redundant_edges.append([u, v])
        
        # Add the edge back
        G.add_edge(u, v)
    
    return non_redundant_edges





def build_hierarchy(entailements_full_matrix, unique_responses):
        
    Res = get_semantic_ids_and_hierarchy(unique_responses, entailements_full_matrix)

    # print("Res: " + str(Res))
    # print("unique_responses : " + str(unique_responses))
    relations = copy.deepcopy(Res["R"])
    # print("relations : " + str(relations))
    # print("Res[R] :" +str(Res["R"]))
    cluster_belonging = copy.deepcopy(Res['c'])
    # print("cluster_belonging :" + str(cluster_belonging))
    Gtemp = nx.DiGraph(relations)
    relations_add_single_root = copy.deepcopy(relations)
    # we add relations to the unique root

    set_Res = set(Res['c'])
    list_set_Res = list(set_Res)
    clusters_list = list(set_Res)


    stringsInrel = list(set(list(chain.from_iterable(Res['R']))))


    root_nodes = [node for node, in_deg in Gtemp.out_degree() if in_deg == 0] + [k for k in clusters_list if k not in stringsInrel]

    # print("root_nodes : " + str(root_nodes))

    for root in root_nodes:
        relations_add_single_root.append([root, -1])

    cluster2letters = {}
    cluster2responses = {}
    cluster2names = {}

    for cluster in clusters_list:
        # temp = [chr(i + 97) for i, value in enumerate(cluster_belonging) if value == cluster]
        temp =  [unique_responses[i] for i, value in enumerate(cluster_belonging) if value == cluster]
        cluster2letters[str(cluster)] = temp
        cluster2names[str(cluster)] = '\n or \n'.join(temp)
        cluster2responses[str(cluster)] = [unique_responses[i] for i, value in enumerate(cluster_belonging) if value == cluster]

    relations = eliminate_redundant_relations(relations_add_single_root)


    # add a child node to each parent with single child
    # Let us close the world by adding a filling nodes to the parents that own only one child 
    hierarchy = copy.deepcopy(relations)
    Parents = []
    filling_nodes = []
    filling_nodes_names = []
    for r in hierarchy:
        if r[1] not in Parents:
            Parents.append(r[1])

    IsArelListClosed = hierarchy[:]

    nodes2names = copy.deepcopy(cluster2names)
    nodes2names["-1"] = "Ignorance"

    k = 1
    for parent in Parents:
        if parent ==-1:
            continue
        childCount = 0
        for r in hierarchy:
            if r[1] == parent: # we suppose no repetition in the is-a list
                childCount += 1
                if childCount > 1:
                    break
                single_child = r[0]
        if childCount > 1:
            continue

        IsArelListClosed.append([max(cluster_belonging) + k, parent])
        filling_node = str(nodes2names[str(parent)]) + "\n but not \n " + str(nodes2names[str(single_child)])
        filling_nodes.append([filling_node])
        nodes2names[str(max(cluster_belonging) + k)] = filling_node
        k = k + 1
    
    # we a add a final node to account for the the tail part wich has no semantic similarity with the observed sampled
    IsArelListClosed.append([max(cluster_belonging) + k, -1])
    filling_node = str("semantically unrelated tail" )
    filling_nodes.append([filling_node])
    nodes2names[str(max(cluster_belonging) + k)] = filling_node

    
    hierarchy_info = {'relation' : IsArelListClosed, 'filling_nodes' : filling_nodes, 'cluster_belonging' : cluster_belonging, 'unique_responses' : unique_responses, 'nodes2names' : nodes2names}
    return(hierarchy_info)




def build_frame_of_discernement(hierarchy_info):

    discernment_info = {}
    C = list(hierarchy_info['nodes2names'].keys())
    discernment_info["C"] = C

    # print("C : " + str(C))
    Omega = [] # identification of the elemeents that constitute the frame of discernement
    for c in C:
        cIsEl = True
        for rel in hierarchy_info['relation']:
            if str(rel[1]) == c:
                cIsEl = False
                break
        if cIsEl:
            Omega.append(c)

    # print("Omega : " + str(Omega))
    discernment_info["Omega"] = Omega

    C_bin = np.zeros( (len(Omega), len(C)), dtype=int)
    treated = [] # class which representation is completed
    Totreat = C[:]
    # the singletons are simply on-hot vectors:
    for i in range(len(Omega)):
        C_bin[i, C.index(Omega[i])] = int(1)
        treated.append(C.index(Omega[i]))
        Totreat.remove(Omega[i])


    # Representation  of the sets that are not singletons
    l = 0
    k = 0
    while len(Totreat) > 0: 
        candidate2treatment = Totreat[k]
        # is all class c, such that c is-a candidate class, already treated?
        # print("candidate2treatment " + str(candidate2treatment))
        treat = True
        childs = []
        for r in hierarchy_info['relation']:
            if str(r[1]) == candidate2treatment:
                if str(r[0]) in Totreat:
                    treat = False 
                    break
                else:
                    childs.append(r[0])

        # print("childs " + str(childs))
        # if yes than we can treat candidate2treatment 
        if treat:
            k = 0 
            Totreat.remove(candidate2treatment)

            for child in childs:
                indices = [i for i, x in enumerate(C_bin[:, C.index(str(child))].tolist()) if x == 1]

                for index in indices:
                    C_bin[index, C.index(candidate2treatment)] = int(1)

        # if no we cannot and so we try to treat the next one
        else:
            k = k + 1

    # print("C_bin : " + str(C_bin))
    discernment_info["C_bin"] = np.array(C_bin)

    C_dec = [] # index of each class in the nat order

    for i in range(len(C)):
        bin_vect = list(C_bin[:,i])
        C_dec.append(myBin2dec(bin_vect))

    # print("C_dec: " + str(C_dec))
    discernment_info["C_dec"] = C_dec

    discernment_info["Omega_dec"] = max(C_dec)
    discernment_info["Omega_size"] = discernment_info['C_bin'].shape[0]

    discernment_info["C_dec2C_bin"] = {}
    discernment_info["C_dec2set_name"] = {}
    for i in range(len(C)):
        discernment_info["C_dec2C_bin"][C_dec[i]] = C_bin[:,i]
        discernment_info["C_dec2set_name"][C_dec[i]] = vector_to_set_expression(C_bin[:,i])



    return discernment_info




def mass_assignment(log_liks_with_eos, responses, hierarchy_info, discernment_info):
    F = []
    M = []
    F_names = []

    sequence_logprob = [np.sum(log_lik) for log_lik in log_liks_with_eos]

    sequence_prob = [np.exp(sequence_logprob[i]) for i in range(len(sequence_logprob))]

    sequence_prob_uniqueEvents = [sequence_prob[responses.index(response)] for response in hierarchy_info['unique_responses']]

    # print("sequence_prob_uniqueEvents is " + str(sequence_prob_uniqueEvents))

    if sum(sequence_prob_uniqueEvents) > 1: # because the sequence probability are not perfect this case may occur
                sequence_prob_uniqueEvents = [sequence_prob_uniqueEvents[i]/sum(sequence_prob_uniqueEvents) for i in range(len(sequence_prob_uniqueEvents))]
                logging.warning("sum sequence_prob_uniqueEvents exceeded 1. New sequence_prob_uniqueEvents is " + str(sequence_prob_uniqueEvents))


    cluster_list = list(set(hierarchy_info['cluster_belonging']))

    proba_cluster = [0 for i in range(len(cluster_list))] 
    for event_index in range(len(hierarchy_info['cluster_belonging'])):
        cluster_index = hierarchy_info['cluster_belonging'][event_index]
        proba_cluster[cluster_index] += sequence_prob_uniqueEvents[event_index]


    for focal_cluster in cluster_list:
        F.append(discernment_info['C_dec'][focal_cluster])
        M.append(proba_cluster[focal_cluster])
        F_names.append(hierarchy_info['nodes2names'][str(focal_cluster)])

    # add the rest to the total set 
    if sum(proba_cluster)<0.999:
        F.append(discernment_info['Omega_dec'])
        M.append(1 - sum(proba_cluster))
        F_names.append(hierarchy_info['nodes2names'][str(-1)])

    return {"F":F, "M":M, "F_names:" : F_names}


#------ For ablation studies



def build_hierarchy_ablation2(unique_responses, entailment_model, args, example):
        
    # Res = get_semantic_ids_and_hierarchy(unique_responses, entailements_full_matrix)
    Res = {}
    semantic_ids = get_semantic_ids(
    unique_responses, model=entailment_model,
    strict_entailment=args.strict_entailment, example=example)
    Res['c'] = semantic_ids

    Res['R']=[]

    # print("Res: " + str(Res))
    # print("unique_responses : " + str(unique_responses))
    relations = copy.deepcopy(Res["R"])
    # print("relations : " + str(relations))
    # print("Res[R] :" +str(Res["R"]))
    cluster_belonging = copy.deepcopy(Res['c'])
    # print("cluster_belonging :" + str(cluster_belonging))
    Gtemp = nx.DiGraph(relations)
    relations_add_single_root = copy.deepcopy(relations)
    # we add relations to the unique root

    set_Res = set(Res['c'])
    list_set_Res = list(set_Res)
    clusters_list = list(set_Res)


    stringsInrel = list(set(list(chain.from_iterable(Res['R']))))


    root_nodes = [node for node, in_deg in Gtemp.out_degree() if in_deg == 0] + [k for k in clusters_list if k not in stringsInrel]

    # print("root_nodes : " + str(root_nodes))

    for root in root_nodes:
        relations_add_single_root.append([root, -1])

    cluster2letters = {}
    cluster2responses = {}
    cluster2names = {}

    for cluster in clusters_list:
        # temp = [chr(i + 97) for i, value in enumerate(cluster_belonging) if value == cluster]
        temp =  [unique_responses[i] for i, value in enumerate(cluster_belonging) if value == cluster]
        cluster2letters[str(cluster)] = temp
        cluster2names[str(cluster)] = '\n or \n'.join(temp)
        cluster2responses[str(cluster)] = [unique_responses[i] for i, value in enumerate(cluster_belonging) if value == cluster]

    relations = eliminate_redundant_relations(relations_add_single_root)

    # relations_add_single_root_not_redundant : [['england', 'Ignorance'], ['wales', 'Ignorance']]


    # print("relations_add_single_root_not_redundant : " + str(relations_add_single_root_not_redundant))


    # add a child node to each parent with single child
    # Let us close the world by adding a filling nodes to the parents that own only one child 
    hierarchy = copy.deepcopy(relations)
    Parents = []
    filling_nodes = []
    filling_nodes_names = []
    for r in hierarchy:
        if r[1] not in Parents:
            Parents.append(r[1])

    IsArelListClosed = hierarchy[:]

    nodes2names = copy.deepcopy(cluster2names)
    nodes2names["-1"] = "Ignorance"

    k = 1
    for parent in Parents:
        if parent ==-1:
            continue
        childCount = 0
        for r in hierarchy:
            if r[1] == parent: # we suppose no repetition in the is-a list
                childCount += 1
                if childCount > 1:
                    break
                single_child = r[0]
        if childCount > 1:
            continue

        IsArelListClosed.append([max(cluster_belonging) + k, parent])
        filling_node = str(nodes2names[str(parent)]) + "\n but not \n " + str(nodes2names[str(single_child)])
        filling_nodes.append([filling_node])
        nodes2names[str(max(cluster_belonging) + k)] = filling_node
        k = k + 1
    
    # we a add a final node to account for the the tail part wich has no semantic similarity with the observed sampled
    IsArelListClosed.append([max(cluster_belonging) + k, -1])
    filling_node = str("semantically unrelated tail" )
    filling_nodes.append([filling_node])
    nodes2names[str(max(cluster_belonging) + k)] = filling_node

    
    hierarchy_info = {'relation' : IsArelListClosed, 'filling_nodes' : filling_nodes, 'cluster_belonging' : cluster_belonging, 'unique_responses' : unique_responses, 'nodes2names' : nodes2names}
    return(hierarchy_info)





def build_hierarchy_ablation1(entailements_full_matrix, unique_responses):
        
    Res = get_semantic_ids_and_hierarchy(unique_responses, entailements_full_matrix)
    Res['R']=[]

    # print("Res: " + str(Res))
    # print("unique_responses : " + str(unique_responses))
    relations = copy.deepcopy(Res["R"])
    # print("relations : " + str(relations))
    # print("Res[R] :" +str(Res["R"]))
    cluster_belonging = copy.deepcopy(Res['c'])
    # print("cluster_belonging :" + str(cluster_belonging))
    Gtemp = nx.DiGraph(relations)
    relations_add_single_root = copy.deepcopy(relations)
    # we add relations to the unique root

    set_Res = set(Res['c'])
    list_set_Res = list(set_Res)
    clusters_list = list(set_Res)


    stringsInrel = list(set(list(chain.from_iterable(Res['R']))))


    root_nodes = [node for node, in_deg in Gtemp.out_degree() if in_deg == 0] + [k for k in clusters_list if k not in stringsInrel]

    # print("root_nodes : " + str(root_nodes))

    for root in root_nodes:
        relations_add_single_root.append([root, -1])

    cluster2letters = {}
    cluster2responses = {}
    cluster2names = {}

    for cluster in clusters_list:
        # temp = [chr(i + 97) for i, value in enumerate(cluster_belonging) if value == cluster]
        temp =  [unique_responses[i] for i, value in enumerate(cluster_belonging) if value == cluster]
        cluster2letters[str(cluster)] = temp
        cluster2names[str(cluster)] = '\n or \n'.join(temp)
        cluster2responses[str(cluster)] = [unique_responses[i] for i, value in enumerate(cluster_belonging) if value == cluster]

    relations = eliminate_redundant_relations(relations_add_single_root)

    # relations_add_single_root_not_redundant : [['england', 'Ignorance'], ['wales', 'Ignorance']]


    # print("relations_add_single_root_not_redundant : " + str(relations_add_single_root_not_redundant))


    # add a child node to each parent with single child
    # Let us close the world by adding a filling nodes to the parents that own only one child 
    hierarchy = copy.deepcopy(relations)
    Parents = []
    filling_nodes = []
    filling_nodes_names = []
    for r in hierarchy:
        if r[1] not in Parents:
            Parents.append(r[1])

    IsArelListClosed = hierarchy[:]

    nodes2names = copy.deepcopy(cluster2names)
    nodes2names["-1"] = "Ignorance"

    k = 1
    for parent in Parents:
        if parent ==-1:
            continue
        childCount = 0
        for r in hierarchy:
            if r[1] == parent: # we suppose no repetition in the is-a list
                childCount += 1
                if childCount > 1:
                    break
                single_child = r[0]
        if childCount > 1:
            continue

        IsArelListClosed.append([max(cluster_belonging) + k, parent])
        filling_node = str(nodes2names[str(parent)]) + "\n but not \n " + str(nodes2names[str(single_child)])
        filling_nodes.append([filling_node])
        nodes2names[str(max(cluster_belonging) + k)] = filling_node
        k = k + 1
    
    # we a add a final node to account for the the tail part wich has no semantic similarity with the observed sampled
    IsArelListClosed.append([max(cluster_belonging) + k, -1])
    filling_node = str("semantically unrelated tail" )
    filling_nodes.append([filling_node])
    nodes2names[str(max(cluster_belonging) + k)] = filling_node

    
    hierarchy_info = {'relation' : IsArelListClosed, 'filling_nodes' : filling_nodes, 'cluster_belonging' : cluster_belonging, 'unique_responses' : unique_responses, 'nodes2names' : nodes2names}
    return(hierarchy_info)

