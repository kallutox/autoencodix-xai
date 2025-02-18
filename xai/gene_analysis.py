from helper_functions import *
from gprofiler import GProfiler
import scipy.stats as stats
import re
import mygene
import ast
import gseapy as gp
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_gene_lists_cf(beta):
    if beta == 0.01:
        beta = '001'
    # dls genes
    with open(f"cf_reports/cleaned_features_{beta}_deepliftshap.txt", "r") as file:
        dls_list = [line.strip() for line in file if line.strip()]
    dls_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in dls_list[1:]
    ]

    # lime genes
    with open(f"cf_reports/cleaned_features_{beta}_lime.txt", "r") as file:
        lime_list = [line.strip() for line in file if line.strip()]
    lime_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in lime_list[1:]
    ]

    # ig genes
    with open(f"cf_reports/cleaned_features_{beta}_integrated_gradients.txt", "r") as file:
        ig_list = [line.strip() for line in file if line.strip()]
    ig_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in ig_list[1:]
    ]

    return dls_genes, lime_genes, ig_genes


def get_gene_lists_tcga(beta, cancer_type):
    if beta == 0.01:
        beta = '001'

    # dls genes
    with open(f"tcga_reports/all_features_{cancer_type}_{beta}_deepliftshap.txt", "r") as file:
        dls_list = [line.strip() for line in file if line.strip()]

    dls_genes_with_mod = [line.split(',')[0] for line in dls_list[1:]]
    dls_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in dls_list[1:]
    ]

    # lime genes
    with open(f"tcga_reports/all_features_{cancer_type}_{beta}_lime.txt", "r") as file:
        lime_list = [line.strip() for line in file if line.strip()]

    lime_genes_with_mod = [line.split(',')[0] for line in lime_list[1:]]
    lime_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in lime_list[1:]
    ]

    # ig genes
    with open(f"tcga_reports/all_features_{cancer_type}_{beta}_integrated_gradients.txt", "r") as file:
        ig_list = [line.strip() for line in file if line.strip()]

    ig_genes_with_mod = [line.split(',')[0] for line in ig_list[1:]]
    ig_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in ig_list[1:]
    ]

    return dls_genes, lime_genes, ig_genes, dls_genes_with_mod, lime_genes_with_mod, ig_genes_with_mod


def calculate_average_gene_rank(candidate_genes, ranked_genes):
    """
    For each gene in candidate_genes, find its 1-indexed rank in ranked_genes.
    Print the rank for each gene (or indicate if the gene is not found), and
    compute the average rank of all genes that are present in ranked_genes.

    Parameters:
    - candidate_genes (list): List of gene names to look for.
    - ranked_genes (list): Ranked list of gene names.

    Returns:
    - float or None: The average rank of found genes, or None if no genes were found.
    """
    positions = []
    for gene in candidate_genes:
        if gene in ranked_genes:
            position = ranked_genes.index(gene) + 1  # convert to 1-indexed rank
            positions.append(position)

    if positions:
        average_rank = sum(positions) / len(positions)
        print(f"Average gene rank: {average_rank:.2f}")
        return average_rank
    else:
        print("No genes found in the ranked list.")
        return None


def get_all_gene_ranks_cf(beta, gene_list):
    dls_genes, lime_genes, ig_genes = get_gene_lists_cf(beta)

    print('\nDeepLiftShap Results: ')
    for gene in gene_list:
        get_gene_rank(gene, dls_genes)
    calculate_average_gene_rank(gene_list, dls_genes)

    print('\nLIME Results: ')
    for gene in gene_list:
        get_gene_rank(gene, lime_genes)
    calculate_average_gene_rank(gene_list, lime_genes)

    print('\nIntegradient Gradient Results: ')
    for gene in gene_list:
        get_gene_rank(gene, ig_genes)
    calculate_average_gene_rank(gene_list, ig_genes)


def collect_gene_ranks(disease, method, beta, gene_list):
    """
    Returns a list of ranks for each gene in gene_list for the given method and beta.
    If a gene is not found, assigns a rank of (len(ranking_list) + 1).
    """
    # Get the ranking list based on the method and beta value
    if disease == 'cf':
        if method == "DeepLiftShap":
            ranking_list = get_gene_lists_cf(beta)[0]
        elif method == "LIME":
            ranking_list = get_gene_lists_cf(beta)[1]
        elif method == "IntegratedGradients":
            ranking_list = get_gene_lists_cf(beta)[2]
        else:
            raise ValueError("Unknown method specified")
    else:
        cancer = disease
        if method == "DeepLiftShap":
            ranking_list = get_gene_lists_tcga(beta, cancer)[0]
        elif method == "LIME":
            ranking_list = get_gene_lists_tcga(beta, cancer)[1]
        elif method == "IntegratedGradients":
            ranking_list = get_gene_lists_tcga(beta, cancer)[2]
        else:
            raise ValueError("Unknown method specified")

    ranks = []
    for gene in gene_list:
        rank = get_gene_rank(gene, ranking_list)
        if rank is not None:
            ranks.append(rank + 1)
        else:
            ranks.append(len(ranking_list) + 1)
    return ranks


def get_all_gene_ranks_tcga(cancer_type, beta, gene_list):
    # gene list -> include mut./meth.
    dls_genes = get_gene_lists_tcga(beta, cancer_type)[0]
    lime_genes = get_gene_lists_tcga(beta, cancer_type)[1]
    ig_genes = get_gene_lists_tcga(beta, cancer_type)[2]

    print('\nDeepLiftShap Results: ')
    for gene in gene_list:
        get_gene_rank(gene, dls_genes)
    calculate_average_gene_rank(gene_list, dls_genes)

    print('\nLIME Results: ')
    for gene in gene_list:
        get_gene_rank(gene, lime_genes)
    calculate_average_gene_rank(gene_list, lime_genes)

    print('\nIntegradient Gradient Results: ')
    for gene in gene_list:
        get_gene_rank(gene, ig_genes)
    calculate_average_gene_rank(gene_list, ig_genes)


def get_gene_info_cf(beta, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_cf(beta)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]


    mg = mygene.MyGeneInfo()
    print("DeepLiftShap - Genes:")
    dls_gene_info = mg.querymany(dls_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in dls_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("LIME - Genes:")
    lime_gene_info = mg.querymany(lime_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in lime_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("Integrated Gradients - Genes:")
    ig_gene_info = mg.querymany(ig_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in ig_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")


def get_gene_info_tcga(beta, cancer_type, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_tcga(beta, cancer_type)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]

    # Retrieve gene information using mygene
    mg = mygene.MyGeneInfo()
    print("DeepLiftShap - Genes:")
    dls_gene_info = mg.querymany(dls_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in dls_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("LIME - Genes:")
    lime_gene_info = mg.querymany(lime_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in lime_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("Integrated Gradients - Genes:")
    ig_gene_info = mg.querymany(ig_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in ig_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")


def pathway_enrichment_analysis_cf(beta, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_cf(beta)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]

    if beta == 0.01:
        beta = '001'

    background = get_interim_data(f"cf_{beta}_1")
    feature_names = list(background.columns)
    ensembl_ids = [gene_id.replace("RNA_", "") for gene_id in feature_names]
    gene_metadata = get_cf_metadata(ensembl_ids)
    all_genes = [gene_metadata[id]['feature_name'] for id in ensembl_ids if id in gene_metadata]

    gp = GProfiler(return_dataframe=True)

    print("DeepLiftShap - Genes:")
    dls_enrichment_results = gp.profile(organism='hsapiens', query=dls_genes, background=all_genes)
    print(dls_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])
    #sorted_results = dls_enrichment_results.sort_values(by='intersection_size', ascending=False)
    #print(sorted_results[['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    print("LIME - Genes:")
    lime_enrichment_results = gp.profile(organism='hsapiens', query=lime_genes, background=all_genes)
    print(lime_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    print("Integrated Gradients - Genes:")
    ig_enrichment_results = gp.profile(organism='hsapiens', query=ig_genes, background=all_genes)
    print(ig_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    dls_enrichment_results.to_csv(f'evaluation_reports/cf_dls_enrichment_{beta}_top{gene_n}.csv', index=False)
    lime_enrichment_results.to_csv(f'evaluation_reports/cf_lime_enrichment_{beta}_top{gene_n}.csv', index=False)
    ig_enrichment_results.to_csv(f'evaluation_reports/cf_ig_enrichment_{beta}_top{gene_n}.csv', index=False)


def pathway_enrichment_analysis_gsea_cf(beta, gene_n):
    dls_genes_all, lime_genes, ig_genes = get_gene_lists_cf(beta)
    dls_genes = dls_genes_all[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]

    if beta == 0.01:
        beta = '001'

    # unranked list
    msigdb_gmt = 'materials/c2.all.v2024.1.Hs.symbols.gmt'
    enr = gp.enrichr(gene_list=dls_genes,
                     gene_sets=msigdb_gmt,
                     organism='Human',
                     outdir='evaluation_reports',
                     cutoff=0.05)
    sorted_results = enr.results.sort_values('Adjusted P-value', ascending=True)

    print(sorted_results[['Term', 'Overlap', 'Adjusted P-value', 'Genes']])

    # on ranked list
    N = len(dls_genes_all)
    pseudo_scores = {gene: N - i for i, gene in enumerate(dls_genes_all)}
    df_rank = pd.DataFrame(list(pseudo_scores.items()), columns=['gene', 'score'])
    df_rank = df_rank.sort_values(by='score', ascending=False)
    df_rank.to_csv(f'materials/dls_{beta}_ranked_list.rnk', sep='\t', header=False, index=False)

    gsea_results = gp.prerank(rnk=f'materials/dls_{beta}_ranked_list.rnk',
                           gene_sets=msigdb_gmt,
                           outdir='evaluation_reports',
                           permutation_num=1000,
                           min_size=15,
                           max_size=500,
                           seed=42)

    print(gsea_results.res2d)


def pathway_enrichment_analysis_tcga(beta, cancer_type, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_tcga(beta, cancer_type)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]

    if beta == 0.01:
        beta = '001'

    background = get_interim_data(f"tcga_{beta}_1")
    entrez_ids = list(background.columns)
    gene_metadata = get_tcga_metadata(entrez_ids)
    background = [gene_metadata.get(gene_id, gene_id) for gene_id in background.columns]
    all_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in background[1:]
    ]

    gp = GProfiler(return_dataframe=True)

    print("DeepLiftShap - Genes:")
    dls_enrichment_results = gp.profile(organism='hsapiens', query=dls_genes, background=all_genes)
    print(dls_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])
    #print(dls_enrichment_results.columns.to_list())

    print("LIME - Genes:")
    lime_enrichment_results = gp.profile(organism='hsapiens', query=lime_genes, background=all_genes)
    print(lime_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    print("Integrated Gradients - Genes:")
    ig_enrichment_results = gp.profile(organism='hsapiens', query=ig_genes, background=all_genes)
    print(ig_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    dls_enrichment_results.to_csv(f'evaluation_reports/tcga_dls_enrichment_{cancer_type}_{beta}_top{gene_n}.csv', index=False)
    lime_enrichment_results.to_csv(f'evaluation_reports/tcga_lime_enrichment_{cancer_type}_{beta}_top{gene_n}.csv', index=False)
    ig_enrichment_results.to_csv(f'evaluation_reports/tcga_ig_enrichment_{cancer_type}_{beta}_top{gene_n}.csv', index=False)


def pairwise_test_ranks(disease, associated_genes):
    # stat testing
    methods = ['DeepLiftShap', 'LIME', 'IntegratedGradients']
    for method in methods:
        ranks_beta001 = collect_gene_ranks(disease, method, 0.01, associated_genes)
        ranks_beta1 = collect_gene_ranks(disease, method, 1, associated_genes)

        t_stat, p_value = stats.ttest_rel(ranks_beta001, ranks_beta1)
        print(f"{method} paired t-test: t = {t_stat:.3f}, p = {p_value:.3f}")


if __name__ == "__main__":
    # cystic fibrosis
    # gene list from malacards
    cf_associated_genes = ['CFTR', 'TGFB1', 'FCGR2A', 'CLN6', 'SERPINA1', 'HFE', 'SCNN1B', 'SCNN1A', 'ELN', 'CAV1',
                              'DNAH5', 'DNAH11', 'CCDC40', 'DNAI1', 'CCDC39', 'ABCC6', 'RSPH1']

    # tcga
    # gene lists from malacard
    laml_associated_genes = ['CEBPA', 'DNMT3A', 'JAK2', 'GATA2', 'TERT', 'FLT3', 'RUNX1', 'NPM1', 'KIT', 'KRAS',
                                'ETV6', 'LPP', 'CHIC2', 'MLLT10', 'NUP214']
    prad_associated_genes = ['TP53', 'SPOP', 'PTEN', 'AKT1', 'PIK3CA', 'ERBB2', 'CTNNB1', 'BRAF', 'CDKN2A', 'IDH1',
                                'SMAD4', 'HRAS', 'MAP2K1', 'MED12', 'XPO1', 'CNOT9', 'LRRC56', 'APC', 'PXMP4', 'WDR19'
                             'BAX', 'PART1', 'MIEN1', 'ITGA2B', 'KLK3', 'MIR145', 'AR', 'MIR221', 'NKX3-1']
    thca_associated_genes = ['RET', 'HRAS', 'KRAS', 'LRRC56', 'BRAF', 'NRAS', 'PTEN', 'APC', 'TP53', 'PRKAR1A',
                                'KIT', 'DICER1', 'WRN', 'PTCSC3', 'MALAT1', 'PVT1', 'GAS5', 'HOTAIR', 'NEAT1', 'H19']

    cancer_types = ['LAML', 'PRAD', 'THCA']

    # CF
    # print('Ranks of disease associated genes:')
    # print('Beta = 0.01:')
    # get_all_gene_ranks_cf(beta=0.01, gene_list=cf_associated_genes)
    # print('\nBeta = 1:')
    # get_all_gene_ranks_cf(beta=1, gene_list=cf_associated_genes)
    #pairwise_test_ranks('cf', cf_associated_genes)

    # LAML
    # print('Ranks of disease associated genes - LAML:')
    # print('Beta = 0.01:')
    # get_all_gene_ranks_tcga('LAML', 0.01, laml_associated_genes)
    # print('\nBeta = 1:')
    # get_all_gene_ranks_tcga('LAML', 1, laml_associated_genes)
    # pairwise_test_ranks('LAML', laml_associated_genes)

    # PRAD
    # print('Ranks of disease associated genes - PRAD:')
    # print('Beta = 0.01:')
    # get_all_gene_ranks_tcga('PRAD', 0.01, prad_associated_genes)
    # print('\nBeta = 1:')
    # get_all_gene_ranks_tcga('PRAD', 1, prad_associated_genes)
    #pairwise_test_ranks('PRAD', prad_associated_genes)

    # THCA
    # print('Ranks of disease associated genes - THCA:')
    # print('Beta = 0.01:')
    # get_all_gene_ranks_tcga('THCA', 0.01, thca_associated_genes)
    # print('\nBeta = 1:')
    # get_all_gene_ranks_tcga('THCA', 1, thca_associated_genes)
    #pairwise_test_ranks('THCA', thca_associated_genes)


    # functional gene analysis
    # CF
    print('\n\nFunctional Gene Analysis:')
    print('Beta = 0.01:')
    get_gene_info_cf(beta=0.01, gene_n=15)
    print('\nBeta = 1:')
    get_gene_info_cf(beta=1, gene_n=15)

    #get_gene_info_tcga(beta=0.01, cancer_type='LAML', gene_n=15)

    # pathway enrichment analysis
    # print('\n\nPathway Enrichment Analysis:')
    # print('Beta = 0.01:')
    # pathway_enrichment_analysis_cf(0.01, 100)
    # print('Beta = 1:')
    # pathway_enrichment_analysis_cf(1, 100)

    # GSEA path enrichment analysis - unranked (top 100) and ranked (all genes)
    #pathway_enrichment_analysis_gsea_cf(0.01, 100)

    #pathway_enrichment_analysis_tcga(0.01, 'THCA', 100)


